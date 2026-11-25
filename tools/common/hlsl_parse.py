#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small delimiter-parsing helpers shared by independently generated HLSL/Metal transforms."""
from __future__ import annotations


def require(ok, message):
    if not ok:
        raise ValueError("upstream oracle: " + message)


def balanced_end(text: str, start: int, opening: str = "(", closing: str = ")",
                 message: str = "unbalanced expression") -> int:
    """Index of the `closing` delimiter matching `opening` at text[start]."""
    require(text[start:start + 1] == opening, "opening boundary")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(message)


def argument_list(text: str) -> list[str]:
    """Comma-split call arguments at bracket depth zero; each part stripped."""
    depth, begin, parts = 0, 0, []
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[begin:index].strip())
            begin = index + 1
    return parts + [text[begin:].strip()]
