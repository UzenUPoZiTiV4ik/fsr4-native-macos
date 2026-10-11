#!/usr/bin/env python3
"""FSR 4.0.2 GPU check for automatic selection, FP16 row bounds and fallback."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/tests/backends"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    testcase = OUT / "testcase"
    subprocess.run([str(ROOT / "tools/bin/gen_testcase"), str(testcase),
                    "quality", "1280", "720", "1920", "1080", "2"], check=True)
    info = json.loads(subprocess.check_output(
        [str(ROOT / "tools/bin/fsr4mac"), "--fsr-version", "4.0.2", "--backend-info"], text=True))
    runs = [
        ("auto", [], False),
        ("explicit", ["--fp16" if info["backend"] == "f16" else "--i8mpp"], False),
        ("fp16", ["--fp16"], False),
        ("fp16-repeat", ["--fp16"], False),
        ("fallback", ["--i8mpp"], True),
        ("last-flag", ["--i8mpp", "--fp16"], True),
    ]
    for name, flags, fail_native in runs:
        env = os.environ.copy()
        if fail_native:
            env["FSR4_I8_MFLAGS"] = "-fsr4-force-compiler-failure"
        else:
            env.pop("FSR4_I8_MFLAGS", None)
        with (OUT / f"{name}.log").open("w") as log:
            subprocess.run([str(ROOT / "tools/bin/fsr4mac"), str(testcase),
                            str(OUT / name), "--fsr-version", "4.0.2", *flags], cwd=ROOT, env=env,
                           stdout=log, stderr=log, check=True)
    for frame in range(2):
        name = f"f{frame:02d}_out.bin"
        auto = (OUT / "auto" / name).read_bytes()
        if auto != (OUT / "explicit" / name).read_bytes():
            raise AssertionError(f"automatic selection differs from {info['backend']}: {name}")
        expected = (OUT / "fp16" / name).read_bytes()
        for mode in ("fp16-repeat", "fallback", "last-flag"):
            if expected != (OUT / mode / name).read_bytes():
                raise AssertionError(f"{mode} differs from explicit FP16: {name}")
    if info["backend"] == "i8mpp":
        assert "falling back to FP16" in (OUT / "fallback.log").read_text()
    assert "native INT8 unavailable" not in (OUT / "last-flag.log").read_text()
    print(f"{info['device']}: auto={info['backend']}; repeated FP16, fallback and flag precedence are byte-exact")


if __name__ == "__main__":
    main()
