#!/usr/bin/env python3
"""Create a local diagnostic graph copy that reads back one ANR buffer.

Only the output selection changes. The original kit/program and weights stay
unchanged. For the controlled test host use --image-output 0 with this copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import math
import subprocess


def create(source, destination, buffer, kernels=None):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("diagnostic destination exists")
    program = json.loads((source / "program.json").read_text())
    if buffer < 0 or buffer >= len(program["buffers"]):
        raise ValueError("unknown buffer")
    original = program["output"]
    program["images"].pop("output", None)
    program["stages"] = {name: stages for name, stages in program["stages"].items() if not name.endswith(":image")}
    # The test host reads an image-sized output regardless of the selected
    # buffer length. A final diagnostic copy makes that readback safe.
    output_bytes = program["width"] * program["height"] * 4
    source_bytes = program["buffers"][buffer]["size"]
    output_index = len(program["buffers"])
    program["buffers"].append({"name": "anr_probe_output", "index": output_index,
                               "size": output_bytes, "kind": "io", "data_offset": None})
    program["output"] = output_index
    for name, stages in program["stages"].items():
        if name.startswith("phase:"):
            stages.append({"op": "dispatch", "kernel": "anr_probe.slang:anr_probe_copy",
                           "groups": [math.ceil(output_bytes / 1024), 1, 1], "threads": [256, 1, 1],
                           "buffers": {"source": buffer, "output": output_index},
                           "uniforms": {"sourceBytes": source_bytes, "outputBytes": output_bytes},
                           "label": "probe_readback"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anr-diag-", dir=destination.parent) as directory:
        stage = Path(directory) / "graph"
        stage.mkdir()
        (stage / "program.json").write_text(json.dumps(program, indent=2) + "\n")
        (stage / "weights.bin").write_bytes((source / "weights.bin").read_bytes())
        kernel_dir = stage / "kernels/anr_probe"
        kernel_dir.mkdir(parents=True)
        if kernels is not None:
            import shutil
            shutil.copytree(kernels, stage / "kernels", dirs_exist_ok=True)
        tool_root = Path(__file__).resolve().parents[2]
        subprocess.run([str(tool_root / "tools/bin/dxc"), "-T", "cs_6_6", "-E", "anr_probe_copy",
                        str(tool_root / "tools/anr/probe_copy.hlsl"), "-Fo",
                        str(kernel_dir / "anr_probe.slang.anr_probe_copy.dxil")], check=True)
        fields = []
        for index, name in enumerate(("sourceBytes", "outputBytes")):
            fields.append({"name": name, "binding": {"kind": "uniform", "offset": index * 4, "size": 4},
                           "type": {"kind": "scalar", "scalarType": "uint32"}})
        for index, name in enumerate(("source", "output")):
            fields.append({"name": name, "binding": {"kind": "unorderedAccess", "index": index},
                           "type": {"kind": "resource", "baseShape": "byteAddressBuffer", "access": "readWrite"}})
        reflection = {"version": "1.1", "parameters": fields,
                      "globalScope": {"kind": "constantBuffer", "binding": {"kind": "constantBuffer", "index": 0},
                                      "parameters": fields},
                      "entryPoints": [{"name": "anr_probe_copy", "stage": "compute", "threadGroupSize": [256, 1, 1],
                                       "bindings": [{"name": name, "binding": {"kind": "unorderedAccess", "index": index, "used": 1}}
                                                    for index, name in enumerate(("source", "output"))]}]}
        (kernel_dir / "anr_probe.slang.anr_probe_copy.json").write_text(json.dumps(reflection))
        (stage / "probe.json").write_text(json.dumps({"original_output": original, "output": buffer,
            "changes": ["bounded diagnostic copy after buffer-output phase variants"],
            "original_program_sha256": hashlib.sha256((source / "program.json").read_bytes()).hexdigest()}, indent=2) + "\n")
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buffer", type=int, required=True)
    parser.add_argument("--kernels", type=Path, help="extracted kernel directory to copy locally")
    args = parser.parse_args()
    try:
        print(create(args.source, args.output, args.buffer, args.kernels))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "ANR probe graph: " + str(error) + "\n")


if __name__ == "__main__":
    main()
