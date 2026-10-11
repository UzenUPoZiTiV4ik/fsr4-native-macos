"""Semantics-preserving Metal translations for the FSR 4.1.1 oracle shaders."""
import math
import re
import struct


def buffer_integer_constants(source):
    """Move literal integer arrays to a read-only buffer without reordering.

    The Metal driver's constant propagation blows up on the nested ML2Code
    weight tables even with unrolling disabled. Dynamic constant-buffer loads
    preserve every weight bit and every arithmetic instruction, while avoiding
    specialization of the entire network at pipeline creation time.
    """
    entry = re.search(r"kernel\s+void\s+\w+\s*\(", source)
    if not entry:
        raise ValueError("no Metal kernel entry")
    prefix, body = source[:entry.start()], source[entry.start():]
    aliases = []
    packed = bytearray()
    pattern = re.compile(r"^constant (spvUnsafeArray<.+>) (\w+) = (.+);$", re.M)

    def replace(match):
        typename, name, initializer = match.groups()
        if not re.search(r"spvUnsafeArray<(?:uint|int),\s*\d+>", typename):
            return match.group(0)
        # SPIRV-Cross redundantly emits each row and the complete table. Drop
        # the unreferenced row objects, but extract the table's full literals.
        if not re.search(r"\b" + re.escape(name) + r"\b", body):
            return ""
        values_text = re.sub(r"spvUnsafeArray<[^()]+>\(", "(", initializer)
        values = [int(x) & 0xffffffff for x in re.findall(r"(?<![\w.])(-?\d+)(?:u)?(?![\w.])", values_text)]
        count = math.prod(int(x) for x in re.findall(r",\s*(\d+)>\s*", typename))
        if not count or len(values) != count:
            raise ValueError(f"cannot extract integer constant {name}: {len(values)} != {count}")
        offset = (len(packed) + 15) & ~15
        packed.extend(b"\0" * (offset - len(packed)))
        packed.extend(struct.pack("<" + "I" * count, *values))
        aliases.append(f"    constant auto &{name} = *reinterpret_cast<constant {typename} *>(Fsr411Tables + {offset}u);")
        return ""

    prefix = pattern.sub(replace, prefix)
    if not aliases:
        return source, b""
    body = re.sub(r"(kernel\s+void\s+\w+\s*\()", r"\1constant uchar *Fsr411Tables [[buffer(7)]], ", body, count=1)
    first_brace = body.index("{") + 1
    body = body[:first_brace] + "\n" + "\n".join(aliases) + "\n" + body[first_brace:]
    return prefix + body, bytes(packed)
