// Tests for the installer app's non-UI code: make test-installer
import Darwin
import Foundation

var failures = 0

@MainActor
func check(_ condition: Bool, _ message: String, line: Int = #line) {
    if !condition {
        failures += 1
        print("FAIL line \(line): \(message)")
    }
}

func temporaryDirectory() -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("fsr4-installer-test-\(UUID().uuidString)")
    try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

func write(_ text: String, _ url: URL) {
    try! FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try! text.write(to: url, atomically: true, encoding: .utf8)
}

func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

/// Runs the main run loop until `done` or the timeout.
func wait(_ seconds: Double, until done: () -> Bool) {
    let deadline = Date().addingTimeInterval(seconds)
    while !done() && Date() < deadline { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
}

let scratch = temporaryDirectory()
defer { try? FileManager.default.removeItem(at: scratch) }

// MARK: cxbottle.conf parsing

do {
    let text = """
    [EnvironmentVariables]
    ;; "FSR4_PROFILE" = "commented"
      "D3DMETAL_FRAMEWORK_PATH" = "/Users/me/Application Support/x/D3DMetal"
    "FSR4_PROFILE" = "lossy"
    "FSR4_PROFILE" = "exact-fast"
    "WINEDLLOVERRIDES" = "a=b;c=n"
    "Broken" = unquoted
    """
    let values = Bottles.values(in: text)
    check(values["D3DMETAL_FRAMEWORK_PATH"] == "/Users/me/Application Support/x/D3DMetal", "path with spaces")
    check(values["FSR4_PROFILE"] == "exact-fast", "last occurrence wins")
    check(values["WINEDLLOVERRIDES"] == "a=b;c=n", "value containing '='")
    check(values["Broken"] == nil, "unquoted value is ignored")
    check(values.count == 3, "commented key is ignored: \(values)")
}

do {
    let bottles = scratch.appendingPathComponent("Bottles")
    write("\"FSR4_PANEL\" = \"0\"\n", bottles.appendingPathComponent("steam/cxbottle.conf"))
    write("""
    ;; fsr4-mac
    "D3DMETAL_FRAMEWORK_PATH" = "/x/D3DMetal"
    "FSR4_ROOT" = "/x"
    "FSR4_PROFILE" = "exact-fast"
    "FSR4_PANEL" = "1"
    """, bottles.appendingPathComponent("Игры/cxbottle.conf"))
    write("not a bottle", bottles.appendingPathComponent("Other/readme.txt"))
    let list = Bottles.list(in: bottles)
    check(Set(list.map(\.name)) == ["steam", "Игры"], "folders without cxbottle.conf skipped: \(list.map(\.name))")
    check(list.map(\.name) == list.map(\.name).sorted { $0.localizedStandardCompare($1) == .orderedAscending }, "sorted")
    let steam = list.first { $0.name == "steam" }
    check(steam?.installed == false && steam?.panel == false, "plain bottle")
    let games = list.first { $0.name == "Игры" }
    check(games?.installed == true && games?.root == "/x" && games?.profile == "exact-fast" && games?.panel == true,
          "installed bottle: \(String(describing: games))")
    let modern = scratch.appendingPathComponent("Modern/cxbottle.conf")
    write("""
    "FSR4_FRAMEWORK" = "/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/lib64/apple_gptk/external/D3DMetal.framework/Versions/A/D3DMetal"
    "FSR4_ROOT" = "/x"
    "FSR4_PROFILE" = "int8-lite"
    """, modern)
    check(Bottles.bottle(conf: modern)?.installed == true, "CrossOver 26 FSR4_FRAMEWORK marks installed")
    check(Bottles.bottle(conf: bottles.appendingPathComponent("steam/cxbottle.conf"))?.installed == false,
          "plain bottle stays not installed")
}

// MARK: External bottle discovery (issue #11)

do {
    let base = scratch.appendingPathComponent("config/Bottles")
    let external = scratch.appendingPathComponent("External SSD/My Bottles")
    let other = scratch.appendingPathComponent("Other Bottles")
    let app = scratch.appendingPathComponent("Custom CrossOver.app")
    let userConfig = base.deletingLastPathComponent().appendingPathComponent("CrossOver.conf")
    write("[CrossOver]\n\"BottlePath\" = \"\(external.path)\"\n",
          app.appendingPathComponent("Contents/SharedSupport/CrossOver/etc/CrossOver.conf"))
    write("", external.appendingPathComponent("Steam/cxbottle.conf"))
    write("", other.appendingPathComponent("Steam/cxbottle.conf"))
    check(Bottles.searchDirectories(base: base, apps: [app], environment: [:]).map(\.path) == [external.path], "app-config external root")
    write("[Other]\n\"BottlePath\" = \"/ignored\"\n[CrossOver]\n;\"BottlePath\" = \"/ignored\"\n\"BottlePath\" = \"${TEST_ROOT}\" ; SSD\n", userConfig)
    check(Bottles.searchDirectories(base: base, apps: [app], environment: ["TEST_ROOT": other.path]).map(\.path) == [other.path], "user overrides app, expands path, ignores comments/other sections")
    check(Bottles.searchDirectories(base: base, apps: [app], environment: ["CX_BOTTLE_PATH": external.path]).map(\.path) == [external.path], "environment overrides BottlePath")
    write("[CrossOver]\n\"BottlePath\" = \"\"\n", userConfig)
    check(Bottles.searchDirectories(base: base, apps: [app], environment: [:]).map(\.path) == [base.path], "empty config restores default root")
    let roots = Bottles.searchDirectories(base: base, apps: [], environment: ["CX_BOTTLE_PATH": "\(external.path):\(other.path):\(external.path)"])
    check(roots.map(\.path) == [external.path, other.path], "multiple search roots deduplicated")
    let list = Bottles.list(in: roots)
    check(list.count == 2 && Set(list.map(\.id)).count == 2, "same bottle name has distinct config IDs")
    try! FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    try! FileManager.default.createSymbolicLink(at: base.appendingPathComponent("Steam"), withDestinationURL: external.appendingPathComponent("Steam"))
    check(Bottles.list(in: [base, external]).count == 1, "external symlink is listed once")
}

// MARK: Payload unpack and update

@MainActor
func makeArchive(_ files: [String: String]) -> URL {
    let source = temporaryDirectory()
    for (path, text) in files { write(text, source.appendingPathComponent(path)) }
    let archive = scratch.appendingPathComponent("payload-\(UUID().uuidString).tar.gz")
    let tar = runCommand("/usr/bin/tar", ["-czf", archive.path, "-C", source.path, "."])
    check(tar.ok, "tar: \(tar.output)")
    try? FileManager.default.removeItem(at: source)
    return archive
}

do {
    let root = scratch.appendingPathComponent("Application Support/fsr4-native-macos")
    let v1 = BundledPayload(info: PayloadInfo(version: "v1", commit: "aaa"), archive: makeArchive([
        "tools/crossover/crossover_fsr4.py": "print('v1')\n",
        "src/old.metal": "old\n",
    ]))
    check((try? Payload.ensure(root: root, payload: v1)) == true, "fresh root is unpacked")
    check(exists(Payload.script(in: root)) && exists(root.appendingPathComponent("src/old.metal")), "v1 files")
    check(Payload.installed(at: root) == v1.info, "marker records v1")
    check((try? Payload.ensure(root: root, payload: v1)) == false, "same version is not unpacked again")

    write("cache\n", root.appendingPathComponent("build/shader.metallib"))
    write("log\n", root.appendingPathComponent("dist/crossover/hook.log"))
    let v2 = BundledPayload(info: PayloadInfo(version: "v2", commit: "bbb"), archive: makeArchive([
        "tools/crossover/crossover_fsr4.py": "print('v2')\n",
        "src/new.metal": "new\n",
    ]))
    check((try? Payload.ensure(root: root, payload: v2)) == true, "new version is unpacked")
    check(!exists(root.appendingPathComponent("src/old.metal")), "files of the old version are removed")
    check(exists(root.appendingPathComponent("src/new.metal")), "new files")
    check(exists(root.appendingPathComponent("build/shader.metallib")), "shader cache is kept")
    check(exists(root.appendingPathComponent("dist/crossover/hook.log")), "dist is kept")
    check((try? String(contentsOf: Payload.script(in: root), encoding: .utf8)) == "print('v2')\n", "script updated")

    // A folder without our marker is never cleaned.
    let foreign = scratch.appendingPathComponent("foreign")
    write("keep\n", foreign.appendingPathComponent("user-file.txt"))
    check((try? Payload.ensure(root: foreign, payload: v2)) == true, "unpacks into a folder without a marker")
    check(exists(foreign.appendingPathComponent("user-file.txt")), "unrelated files are not deleted")

    let broken = BundledPayload(info: PayloadInfo(version: "v3", commit: "ccc"), archive: scratch.appendingPathComponent("missing.tar.gz"))
    var threw = false
    do { try Payload.ensure(root: root, payload: broken) } catch { threw = true }
    check(threw, "a missing archive is reported")
}

// MARK: Shell helpers

do {
    let tricky = "it's a \"test\" $HOME `x`"
    let echo = runCommand("/bin/sh", ["-c", "printf %s " + shellQuote(tricky)])
    check(echo.output == tricky, "shellQuote round trip: \(echo.output)")
}

do {
    var output = ""
    var result: (Int32, Bool)?
    let command = StreamingCommand(["/bin/sh", "-c", "echo one; printf 'два\\nno newline'; exit 3"])
    try command.start(output: { output += $0 }, completion: { result = ($0, $1) })
    wait(10) { result != nil }
    check(result?.0 == 3 && result?.1 == false, "exit status: \(String(describing: result))")
    check(output == "one\nдва\nno newline\n", "streamed lines: \(output.debugDescription)")
}

do {
    // Cancel stops the whole process group, including spawned children.
    let pidFile = scratch.appendingPathComponent("child.pid")
    var result: (Int32, Bool)?
    let command = StreamingCommand(["/bin/sh", "-c", "sleep 60 & echo $! > \(shellQuote(pidFile.path)); wait"])
    try command.start(output: { _ in }, completion: { result = ($0, $1) })
    wait(5) { exists(pidFile) }
    command.cancel()
    wait(10) { result != nil }
    check(result?.1 == true, "cancelled: \(String(describing: result))")
    let child = Int32((try? String(contentsOf: pidFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") ?? 0
    check(child > 0, "child pid recorded")
    if child > 0 {
        wait(2) { kill(child, 0) != 0 }
        check(kill(child, 0) != 0, "child process was stopped")
        kill(child, SIGKILL)
    }
}

// MARK: Step names and requirements

check(InstallerModel.describe("python3 /x/tools/compile_fsr411.py --root /x") != nil, "shader step")
check(InstallerModel.describe("codesign -f -s - /x/dist/crossover/D3DMetal.framework/Versions/A/D3DMetal") != nil, "D3DMetal step")
check(InstallerModel.describe("arch -x86_64 /x/dist/crossover/live_test /x/testcases/t /x/ref") != nil, "live test step")
check(InstallerModel.describe("unrelated") == nil, "unknown command has no step")

// MARK: Finding CrossOver (issue #9: renamed copies in ~/Applications)

do {
    let system = scratch.appendingPathComponent("apps/system")
    let home = scratch.appendingPathComponent("apps/home")
    try! FileManager.default.createDirectory(at: system, withIntermediateDirectories: true)
    let patched = home.appendingPathComponent("CrossOver_patched.app")
    let dated = home.appendingPathComponent("CrossOver Preview-20261006.app")
    let nested = home.appendingPathComponent("Games/CrossOver.app")
    for app in [patched, dated, nested] {
        try! FileManager.default.createDirectory(at: app.appendingPathComponent(SystemCheck.crossoverMarker),
                                                 withIntermediateDirectories: true)
    }
    write("", home.appendingPathComponent("CrossOver/Game.app/Contents/Info.plist"))   // bottle launcher
    write("", home.appendingPathComponent("Steam.app/Contents/Info.plist"))
    try! FileManager.default.createDirectory(at: dated.appendingPathComponent(SystemCheck.d3dmetalPaths[0]),
                                             withIntermediateDirectories: true)
    try! FileManager.default.createDirectory(at: nested.appendingPathComponent(SystemCheck.d3dmetalPaths[1]),
                                             withIntermediateDirectories: true)
    let apps = SystemCheck.crossoverApps(in: [system, home])
    check(apps.map(\.lastPathComponent) == ["CrossOver Preview-20261006.app", "CrossOver_patched.app", "CrossOver.app"],
          "renamed copies found, launchers skipped: \(apps.map(\.path))")
    check(SystemCheck.d3dmetal(in: dated)?.path.hasSuffix("apple_gptk/external/D3DMetal.framework") == true, "GPTK 1 path")
    check(SystemCheck.d3dmetal(in: dated)?.path.contains("CrossOver/lib64/") == true, "CrossOver 26 lib64 path")
    check(SystemCheck.d3dmetal(in: nested)?.path.hasSuffix("apple_gptk3/external/D3DMetal.framework") == true, "GPTK 3 path")
    check(SystemCheck.d3dmetal(in: patched) == nil, "no D3DMetal")
    check(SystemCheck.d3dmetalPaths.count == 4, "lib64 and lib paths")
    let legacy = scratch.appendingPathComponent("apps/legacy/CrossOver_legacy.app")
    try! FileManager.default.createDirectory(at: legacy.appendingPathComponent(SystemCheck.crossoverMarker),
                                             withIntermediateDirectories: true)
    try! FileManager.default.createDirectory(at: legacy.appendingPathComponent(SystemCheck.d3dmetalPaths[2]),
                                             withIntermediateDirectories: true)
    check(SystemCheck.d3dmetal(in: legacy)?.path.contains("CrossOver/lib/") == true, "CrossOver 24/25 lib path")
    let standard = system.appendingPathComponent("CrossOver.app")
    try! FileManager.default.createDirectory(at: standard.appendingPathComponent(SystemCheck.crossoverMarker),
                                             withIntermediateDirectories: true)
    check(SystemCheck.crossoverApps(in: [system, home]).first?.path == standard.path, "standard name in /Applications first")
}

// MARK: Test-only CrossOver override (keeps the E2E run off /Applications)

do {
    let clone = scratch.appendingPathComponent("clone/CrossOver.app")
    try! FileManager.default.createDirectory(at: clone.appendingPathComponent(SystemCheck.crossoverMarker),
                                             withIntermediateDirectories: true)
    // Restore the inherited value: the E2E block below uses the same variable.
    let previous = ProcessInfo.processInfo.environment[SystemCheck.crossoverOverrideVariable]
    setenv(SystemCheck.crossoverOverrideVariable, clone.path, 1)
    defer {
        if let previous {
            setenv(SystemCheck.crossoverOverrideVariable, previous, 1)
        } else {
            unsetenv(SystemCheck.crossoverOverrideVariable)
        }
    }
    check(SystemCheck.crossoverOverride?.path == clone.path, "FSR4_INSTALLER_CROSSOVER selects a copy")
    check(SystemCheck.crossoverApps.first?.path == clone.path, "override comes first")
}

let requirements = SystemCheck.all()
check(requirements.map(\.kind) == [.xcode, .metal, .rosetta, .crossover], "requirement order")
for requirement in requirements {
    print("  \(requirement.ok ? "ok  " : "todo") \(requirement.title): \(requirement.detail)")
    check(requirement.ok || requirement.kind == .metal && !requirements[0].ok || requirement.fix != nil,
          "\(requirement.title) offers a fix")
    if requirement.kind == .crossover {
        check(requirement.ok == (requirement.location.map { SystemCheck.d3dmetal(in: $0) != nil } ?? false),
              "CrossOver location: \(String(describing: requirement.location))")
    }
}
print("  chip: \(Chip.name), neural accelerators: \(Chip.hasNeuralAccelerators)")

// MARK: End to end (opt-in, a few minutes)
//   FSR4_E2E="build/.../FSR4 Installer.app/Contents/Resources/payload.tar.gz" \
//   FSR4_INSTALLER_CROSSOVER="/path/to/a/CrossOver copy.app" build/tests/installer_test
// Runs InstallerModel.install() and restore() on a fake bottle. Set
// FSR4_INSTALLER_ROOT and FSR4_INSTALLER_BOTTLES to temporary folders.
// FSR4_INSTALLER_CROSSOVER is required: on CrossOver 26 the install patches
// D3DMetal inside the selected app, so the test must use a copy, never the
// app in /Applications.

if let archive = ProcessInfo.processInfo.environment["FSR4_E2E"] {
    if SystemCheck.crossoverOverride == nil {
        print("  end to end: skipped (set FSR4_INSTALLER_CROSSOVER to a CrossOver copy)")
    } else {
        let bottles = Bottles.directory
        let conf = bottles.appendingPathComponent("E2E Bottle/cxbottle.conf")
        write("[EnvironmentVariables]\n\"WINEDEBUG\" = \"-all\"\n", conf)
        let model = InstallerModel(payload: BundledPayload(info: PayloadInfo(version: "e2e", commit: "test"),
                                                           archive: URL(fileURLWithPath: archive)))
        model.refresh()
        wait(60) { !model.checking }
        model.selectedID = model.bottles.first { $0.name == "E2E Bottle" }?.id
        check(model.canInstall, "can install: requirements \(model.requirementsMet)")
        model.profile = .full
        model.panel = false
        model.install()
        var steps = Set<String>()
        wait(1800) { if !model.step.isEmpty { steps.insert(model.step) }; return !model.busy }
        check(model.banner?.style == .success, "install: \(String(describing: model.banner))\n\(model.log.suffix(2000))")
        print("  steps: \(steps.sorted())")
        let installed = Bottles.list(in: bottles).first { $0.name == "E2E Bottle" }
        check(installed?.installed == true && installed?.profile == "exact-fast" && installed?.panel == false,
              "bottle after install: \(String(describing: installed))")
        check(installed.map(model.ownsInstall) == true, "installed from the app's root")
        check(Payload.installed(at: model.root)?.version == "e2e", "payload marker")
        let installedValues = Bottles.values(in: (try? String(contentsOf: conf, encoding: .utf8)) ?? "")
        if let framework = installedValues["FSR4_FRAMEWORK"] {
            check(FileManager.default.fileExists(atPath: framework + ".fsr4-original"),
                  "in-place backup next to \(framework)")
            check(installedValues["D3DMETAL_FRAMEWORK_PATH"] == nil, "no redirect key on CrossOver 26")
        }
        if let installed {
            model.restore(installed)
            wait(120) { !model.busy }
            check(model.banner?.style == .success, "restore: \(String(describing: model.banner))")
            check(Bottles.list(in: bottles).first { $0.name == "E2E Bottle" }?.installed == false, "bottle after restore")
            let restored = Bottles.values(in: (try? String(contentsOf: conf, encoding: .utf8)) ?? "")
            check(restored["FSR4_FRAMEWORK"] == nil, "framework key removed")
        }
        print("  end to end: done")
    }
}

if failures > 0 {
    print("installer tests: \(failures) failed")
    exit(1)
}
print("installer tests: ok")
