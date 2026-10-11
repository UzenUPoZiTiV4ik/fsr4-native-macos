#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent FSR 4.1.1 lossy coefficient and quantization transforms.

The numerical policy follows zgauthier2000's dll-2026-10-07 release at the
pinned revision below. This module operates on AMD's MIT model data and our
MIT HLSL import. It neither vendors nor invokes that project's GPL shader
rewriters. Its dense matrix output changes the image; zero weights alone do
not reduce the work of a dense Metal matrix multiply.
"""

import hashlib
import re


REFERENCE_REPOSITORY = "https://github.com/zgauthier2000/fsr4-shader-performance-overrides"
REFERENCE_COMMIT = "8d72457ab6302dab8e4f918f6bf83a1af3900288"
REFERENCE_RELEASE = "dll-2026-10-07"
FOLD_WORDS = {1: 4, 5: 18, 10: 18, 12: 20}
ROUNDING_PASSES = frozenset((1, 2, 4, 5, 7, 8, 9, 10, 12))
SHARED_ROUNDING_PASSES = frozenset((7, 8, 9))
ROUND_HALF_UP_FLAG = 4


def rounding_passes(preset="normal"):
    if preset == "normal":
        return ROUNDING_PASSES
    if preset == "ultraperformance":
        # These three shaders have the same original container identity in
        # both models. The upstream lossy build replaces that shared shader.
        return SHARED_ROUNDING_PASSES
    raise ValueError("expected normal or ultraperformance preset")


def matrix_sha256(matrix):
    return hashlib.sha256(bytes(value & 255 for value in matrix)).hexdigest()


def fold_conv3x3(matrix, drop_words):
    """Return a folded copy of one [9 taps][16 inputs][16 outputs] matrix.

    For each output independently, rank the 36 packed int8x4 contributions
    by squared norm, then row-major tap and input quartet. Remove at most K
    nonzero contributions, retaining at least one tap per quartet. Add each
    removed vector to the closest retained tap of the same quartet; prefer
    its original norm on distance ties, then its row-major position. Addition
    saturates each signed byte after every transfer, in selection order.

    Statistics report saturation explicitly: the response to a spatially
    constant input is preserved only where no signed byte was saturated.
    Neither the input sequence nor any initializer bytes are modified.
    """
    if len(matrix) != 9 * 16 * 16:
        raise ValueError("folding expects 2304 signed weights in [tap][input][output] order")
    if any(type(value) is not int or not -128 <= value <= 127 for value in matrix):
        raise ValueError("folding expects signed int8 weights")
    if type(drop_words) is not int or not 0 <= drop_words <= 32:
        raise ValueError("drop_words must be an integer between 0 and 32")

    folded = list(matrix)
    per_channel = []
    saturated = 0
    changed = 0
    original_zero = 0
    resulting_zero = 0
    for output in range(16):
        vectors = []
        positions = []
        for tap in range(9):
            for quartet in range(4):
                indices = [(tap * 16 + quartet * 4 + component) * 16 + output
                           for component in range(4)]
                positions.append(indices)
                vectors.append([matrix[index] for index in indices])
        magnitude = [sum(component * component for component in vector) for vector in vectors]
        original_zero += magnitude.count(0)
        ordered = sorted(range(36), key=lambda index: (magnitude[index], index))
        survivors = [True] * 36
        remaining = [9] * 4
        removed = []
        for index in ordered:
            if len(removed) >= drop_words:
                break
            quartet = index % 4
            if magnitude[index] and remaining[quartet] > 1:
                survivors[index] = False
                remaining[quartet] -= 1
                removed.append(index)

        def distance(source, target):
            source_y, source_x = divmod(source // 4, 3)
            target_y, target_x = divmod(target // 4, 3)
            return (source_y - target_y) ** 2 + (source_x - target_x) ** 2

        for source in removed:
            choices = [index for index in range(source % 4, 36, 4) if survivors[index]]
            target = min(choices, key=lambda index: (distance(source, index), -magnitude[index], index))
            for component in range(4):
                value = vectors[target][component] + vectors[source][component]
                limited = min(127, max(-128, value))
                saturated += limited != value
                vectors[target][component] = limited
                vectors[source][component] = 0
        for vector, indices in zip(vectors, positions):
            resulting_zero += not any(vector)
            changed += vector != [matrix[index] for index in indices]
            for value, index in zip(vector, indices):
                folded[index] = value
        per_channel.append(len(removed))
    stats = {
        "drop_words_per_channel": drop_words,
        "dropped_words": sum(per_channel),
        "dropped_per_channel": per_channel,
        "changed_words": changed,
        "zero_words_before": original_zero,
        "zero_words_after": resulting_zero,
        "saturated_components": saturated,
        "input_sha256": matrix_sha256(matrix),
        "output_sha256": matrix_sha256(folded),
    }
    return folded, stats


def literal_to_matrix(words):
    """Decode the MIT CNB/FNB 36x16 table into one native 16-channel matrix."""
    if len(words) != 576 or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words):
        raise ValueError("expected 576 uint32 literal words")
    result = []
    for tap in range(9):
        for input_channel in range(16):
            for output_channel in range(16):
                row = (output_channel // 4) * 9 + tap
                column = (output_channel % 4) * 4 + input_channel // 4
                unsigned = (words[row * 16 + column] >> (8 * (input_channel % 4))) & 255
                result.append(unsigned - 256 if unsigned >= 128 else unsigned)
    return result


def matrix_to_literal(matrix):
    if len(matrix) != 2304 or any(type(value) is not int or not -128 <= value <= 127 for value in matrix):
        raise ValueError("expected 2304 signed int8 weights")
    words = [0] * 576
    for tap in range(9):
        for input_channel in range(16):
            for output_channel in range(16):
                row = (output_channel // 4) * 9 + tap
                column = (output_channel % 4) * 4 + input_channel // 4
                value = matrix[(tap * 16 + input_channel) * 16 + output_channel]
                words[row * 16 + column] |= (value & 255) << (8 * (input_channel % 4))
    return words


def apply_hlsl_lossy(source, logical_pass, preset="normal"):
    """Transform a generated source copy, retaining the imported exact files.

    Only the two intermediate requantizations use ties toward positive
    infinity. Floating output tails and POST's complete network stay exact.
    Unknown helper/table shapes fail instead of producing a partial profile.
    """
    selected = rounding_passes(preset)
    if logical_pass not in selected:
        return source
    marker = "// fsr411-generated-lossy: pass=%s preset=%s reference=%s\n" % (
        logical_pass, preset, REFERENCE_COMMIT)
    if source.startswith(marker):
        return source
    if "fsr411-generated-lossy:" in source:
        raise ValueError("source already carries a different lossy profile")
    helper = re.compile(
        r"(int requant_shift\(int x, uint shift, int add\)\s*\{)\s*"
        r"uint bit = \(uint\(x\) >> shift\) & 1u;\s*"
        r"return \(x \+ add \+ int\(bit\)\) >> int\(shift\);(?:\s*//[^\n]*)?\s*\}")
    source, count = helper.subn(
        r"\1\n    return (x + add + 1) >> int(shift); // lossy ties toward positive infinity\n}", source)
    if count != 1:
        raise ValueError("expected one known intermediate requantization helper")
    if preset == "normal" and logical_pass in FOLD_WORDS:
        name = "W1" if logical_pass in (1, 12) else "W0"
        array = re.compile(r"(static const uint " + name + r"\[36\]\[16\]\s*=\s*\{)(.*?)(\};)", re.S)
        matches = list(array.finditer(source))
        if len(matches) != 1:
            raise ValueError("expected one known 3x3 literal table")
        match = matches[0]
        literals = re.findall(r"0x([0-9a-fA-F]+)u", match[2])
        matrix = literal_to_matrix([int(value, 16) for value in literals])
        folded, _ = fold_conv3x3(matrix, FOLD_WORDS[logical_pass])
        replacements = iter(matrix_to_literal(folded))
        body = re.sub(r"0x[0-9a-fA-F]+u", lambda _: "0x%08xu" % next(replacements), match[2])
        source = source[:match.start(2)] + body + source[match.end(2):]
    return marker + source
