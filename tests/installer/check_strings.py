#!/usr/bin/env python3
"""Every localizable string of the installer app has a Russian translation
with the same format specifiers, and the table has no stale keys."""

import glob
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(str(p) for p in (ROOT / "tools/installer/Sources").glob("*.swift"))
TABLE = ROOT / "tools/installer/Resources/ru.lproj/Localizable.strings"
SPECIFIER = re.compile(r"%(?:\d+\$)?(?:lld|ld|d|@|f)")

# Xcode's Swift 6.4 driver does not always add the platform macro plugins to
# the default search path (macOS 27 SDK), so @State/@Bindable fail to expand.
# Point swiftc at the installed platform plugins when they are present.
def _macro_plugin_args() -> list:
    xcode = Path(subprocess.check_output(["xcode-select", "-p"], text=True).strip())
    plugins = xcode / "Platforms/MacOSX.platform/Developer/usr/lib/swift/host/plugins"
    return ["-plugin-path", str(plugins)] if plugins.is_dir() else []



def source_keys() -> set:
    with tempfile.TemporaryDirectory() as out:
        subprocess.check_call(["xcrun", "swiftc", "-parse-as-library", "-swift-version", "6",
                               "-target", "arm64-apple-macos26.0", "-module-name", "FSR4Installer",
                               *_macro_plugin_args(),
                               "-emit-localized-strings", "-emit-localized-strings-path", out,
                               "-c", *SOURCES], cwd=out)
        keys = set()
        for path in glob.glob(out + "/*.stringsdata"):
            for entries in json.load(open(path))["tables"].values():
                keys.update(entry["key"] for entry in entries)
        return keys


def main() -> int:
    table = json.loads(subprocess.check_output(["plutil", "-convert", "json", "-o", "-", str(TABLE)]))
    keys = source_keys()
    missing = sorted(keys - table.keys())
    stale = sorted(table.keys() - keys)
    mismatched = sorted(k for k in keys & table.keys()
                        if sorted(SPECIFIER.findall(k)) != sorted(SPECIFIER.findall(table[k])))
    for title, items in (("missing Russian translation", missing), ("not used in the sources", stale),
                         ("format specifiers differ", mismatched)):
        for item in items:
            print(f"{title}: {item!r}")
    if missing or stale or mismatched:
        return 1
    print(f"installer strings: {len(keys)} keys translated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
