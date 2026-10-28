import AppKit
import Foundation
import Metal

/// One row of the requirements list.
struct Requirement: Identifiable, Sendable {
    enum Kind: String, Sendable { case xcode, metal, rosetta, crossover }

    /// How the user (or the app) fixes a missing requirement.
    enum Fix: Sendable {
        case open(URL)
        /// Shell command run with administrator privileges.
        case administrator(String)
        /// Command run without privileges, output streamed to the log.
        case stream([String])
        /// Let the user pick a CrossOver.app that was not found automatically.
        case chooseCrossOver
    }

    let kind: Kind
    let title: String
    let ok: Bool
    let detail: String
    var fix: Fix? = nil
    var fixTitle: String? = nil
    var secondaryFix: Fix? = nil
    var secondaryFixTitle: String? = nil
    /// The app the check found (CrossOver.app); passed on to the install script.
    var location: URL? = nil

    var id: Kind { kind }
}

enum SystemCheck {
    static let minimumXcode = 26
    static let xcodeStore = URL(string: "macappstore://apps.apple.com/app/xcode/id497799835")!
    static let crossoverSite = URL(string: "https://www.codeweavers.com/crossover")!
    static let crossoverBundleID = "com.codeweavers.CrossOver"
    static let crossoverMarker = "Contents/SharedSupport/CrossOver"
    /// D3DMetal inside a CrossOver.app, in order of preference (as in crossover_fsr4.py).
    /// CrossOver 26 moved GPTK 3 to lib64; earlier builds ship it under lib.
    static let d3dmetalPaths = ["Contents/SharedSupport/CrossOver/lib64/apple_gptk/external/D3DMetal.framework",
                                "Contents/SharedSupport/CrossOver/lib64/apple_gptk3/external/D3DMetal.framework",
                                "Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework",
                                "Contents/SharedSupport/CrossOver/lib/apple_gptk3/external/D3DMetal.framework"]
    static let chosenCrossOverKey = "CrossOverApp"
    static let crossoverOverrideVariable = "FSR4_INSTALLER_CROSSOVER"

    /// A CrossOver.app under any name: copies get renamed (CrossOver_patched.app,
    /// CrossOver Preview-20261006.app).
    static func isCrossOver(_ app: URL) -> Bool {
        var directory: ObjCBool = false
        return app.pathExtension == "app"
            && FileManager.default.fileExists(atPath: app.appendingPathComponent(crossoverMarker).path, isDirectory: &directory)
            && directory.boolValue
    }

    static func d3dmetal(in app: URL) -> URL? {
        d3dmetalPaths.map(app.appendingPathComponent).first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// CrossOver copies in `folders`: the standard names first, then any renamed
    /// copy directly in a folder or one subfolder down. Same order as
    /// crossover_apps() in tools/crossover/crossover_fsr4.py.
    static func crossoverApps(in folders: [URL]) -> [URL] {
        let fm = FileManager.default
        var found: [URL] = []
        func add(_ app: URL) {
            if !found.contains(where: { $0.path == app.path }) && isCrossOver(app) { found.append(app) }
        }
        func apps(in folder: URL) -> [String] {
            ((try? fm.contentsOfDirectory(atPath: folder.path)) ?? []).sorted { $0.lowercased() < $1.lowercased() }
        }
        for folder in folders {
            for name in ["CrossOver.app", "CrossOver Preview.app"] { add(folder.appendingPathComponent(name)) }
        }
        for folder in folders {
            for name in apps(in: folder) where !name.hasPrefix(".") {
                let entry = folder.appendingPathComponent(name)
                if entry.pathExtension == "app" {
                    add(entry)
                } else {
                    for sub in apps(in: entry) where sub.hasSuffix(".app") { add(entry.appendingPathComponent(sub)) }
                }
            }
        }
        return found
    }

    /// Every CrossOver this Mac knows about: the Applications folders, then
    /// copies LaunchServices has seen elsewhere (other disks). The copy the
    /// user picked with Choose… comes first.
    static var crossoverApps: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        var apps = crossoverApps(in: [URL(fileURLWithPath: "/Applications"), home.appendingPathComponent("Applications")])
        for app in NSWorkspace.shared.urlsForApplications(withBundleIdentifier: crossoverBundleID)
        where !apps.contains(where: { $0.path == app.path }) && isCrossOver(app) {
            apps.append(app)
        }
        if let override = crossoverOverride {
            apps.removeAll { $0.path == override.path }
            apps.insert(override, at: 0)
        }
        if let chosen = chosenCrossOver {
            apps.removeAll { $0.path == chosen.path }
            apps.insert(chosen, at: 0)
        }
        return apps
    }

    /// Test-only: FSR4_INSTALLER_CROSSOVER selects a CrossOver copy outside
    /// /Applications, so the end-to-end test never patches a real install.
    static var crossoverOverride: URL? {
        guard let path = ProcessInfo.processInfo.environment[crossoverOverrideVariable], !path.isEmpty else { return nil }
        let app = URL(fileURLWithPath: path)
        return isCrossOver(app) ? app : nil
    }

    /// CrossOver.app picked with Choose…, while it still exists.
    static var chosenCrossOver: URL? {
        get {
            guard let path = UserDefaults.standard.string(forKey: chosenCrossOverKey) else { return nil }
            let app = URL(fileURLWithPath: path)
            return isCrossOver(app) ? app : nil
        }
        set { UserDefaults.standard.set(newValue?.path, forKey: chosenCrossOverKey) }
    }

    static func all() -> [Requirement] {
        let xcode = checkXcode()
        return [xcode, checkMetal(xcodeReady: xcode.ok), checkRosetta(), checkCrossOver()]
    }

    // MARK: Xcode

    static func installedXcodes() -> [URL] {
        let fm = FileManager.default
        let folders = [URL(fileURLWithPath: "/Applications"), fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
        let apps = folders.flatMap { folder in
            ((try? fm.contentsOfDirectory(atPath: folder.path)) ?? [])
                .filter { $0.hasPrefix("Xcode") && $0.hasSuffix(".app") }
                .map { folder.appendingPathComponent($0) }
        }
        // Prefer a plain Xcode.app over betas.
        return apps.sorted { ($0.lastPathComponent != "Xcode.app" ? 1 : 0, $0.path) < ($1.lastPathComponent != "Xcode.app" ? 1 : 0, $1.path) }
    }

    static func xcodeMajor(_ app: URL) -> Int? {
        let plist = NSDictionary(contentsOf: app.appendingPathComponent("Contents/Info.plist"))
        guard let version = plist?["CFBundleShortVersionString"] as? String else { return nil }
        return Int(version.split(separator: ".").first ?? "")
    }

    /// The Xcode.app that xcode-select points at, if any.
    static func selectedXcode() -> URL? {
        let selected = runCommand("/usr/bin/xcode-select", ["-p"])
        let path = selected.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard selected.ok, let range = path.range(of: ".app/Contents/Developer") else { return nil }
        let app = URL(fileURLWithPath: String(path[..<range.lowerBound]) + ".app")
        return FileManager.default.fileExists(atPath: app.path) ? app : nil
    }

    static func setupCommand(_ app: URL) -> String {
        let developer = shellQuote(app.appendingPathComponent("Contents/Developer").path)
        let xcodebuild = shellQuote(app.appendingPathComponent("Contents/Developer/usr/bin/xcodebuild").path)
        return "/usr/bin/xcode-select -s \(developer) && \(xcodebuild) -license accept && \(xcodebuild) -runFirstLaunch"
    }

    static func checkXcode() -> Requirement {
        let title = String(localized: "Xcode \(minimumXcode)+")
        let installed = installedXcodes()
        let suitable = installed.filter { (xcodeMajor($0) ?? 0) >= minimumXcode }
        guard let candidate = suitable.first else {
            let detail = installed.isEmpty
                ? String(localized: "Needed to build the hook and FSR shaders. Free in the App Store.")
                : String(localized: "Installed Xcode is too old. Update it in the App Store.")
            return Requirement(kind: .xcode, title: title, ok: false, detail: detail,
                               fix: .open(xcodeStore), fixTitle: String(localized: "Open App Store"))
        }
        let setup = Requirement(kind: .xcode, title: title, ok: false,
                                detail: String(localized: "\(candidate.lastPathComponent) is installed but not set up for command-line tools."),
                                fix: .administrator(setupCommand(candidate)), fixTitle: String(localized: "Set Up…"))
        guard let selected = selectedXcode(), (xcodeMajor(selected) ?? 0) >= minimumXcode else { return setup }
        let xcodebuild = selected.appendingPathComponent("Contents/Developer/usr/bin/xcodebuild").path
        // Functional check: -checkFirstLaunchStatus can report pending extras
        // while the compilers already work.
        guard runCommand(xcodebuild, ["-license", "check"]).ok,
              runCommand("/usr/bin/xcrun", ["--sdk", "macosx", "--show-sdk-path"]).ok,
              runCommand("/usr/bin/xcrun", ["clang", "--version"]).ok else {
            return Requirement(kind: .xcode, title: title, ok: false,
                               detail: String(localized: "Accept the Xcode license and finish its first-launch setup."),
                               fix: .administrator(setupCommand(selected)), fixTitle: String(localized: "Set Up…"))
        }
        let version = (NSDictionary(contentsOf: selected.appendingPathComponent("Contents/Info.plist"))?["CFBundleShortVersionString"] as? String) ?? ""
        return Requirement(kind: .xcode, title: title, ok: true, detail: "\(selected.lastPathComponent) \(version)")
    }

    // MARK: Metal Toolchain

    static func checkMetal(xcodeReady: Bool) -> Requirement {
        let title = String(localized: "Metal Toolchain")
        guard xcodeReady else {
            return Requirement(kind: .metal, title: title, ok: false, detail: String(localized: "Set up Xcode first."))
        }
        let metal = runCommand("/usr/bin/xcrun", ["metal", "--version"])
        if metal.ok, let line = metal.output.split(separator: "\n").first, line.contains("metal version") {
            return Requirement(kind: .metal, title: title, ok: true, detail: String(line))
        }
        return Requirement(kind: .metal, title: title, ok: false,
                           detail: String(localized: "Xcode component that compiles FSR shaders for each game."),
                           fix: .stream(["/usr/bin/xcodebuild", "-downloadComponent", "MetalToolchain"]),
                           fixTitle: String(localized: "Download"))
    }

    // MARK: Rosetta

    static func checkRosetta() -> Requirement {
        let title = String(localized: "Rosetta 2")
        if FileManager.default.fileExists(atPath: "/Library/Apple/usr/libexec/oah/libRosettaRuntime") {
            return Requirement(kind: .rosetta, title: title, ok: true, detail: String(localized: "Installed"))
        }
        return Requirement(kind: .rosetta, title: title, ok: false,
                           detail: String(localized: "CrossOver runs Windows games as Intel code."),
                           fix: .administrator("/usr/sbin/softwareupdate --install-rosetta --agree-to-license"),
                           fixTitle: String(localized: "Install"))
    }

    // MARK: CrossOver

    static func checkCrossOver() -> Requirement {
        let title = String(localized: "CrossOver 24+")
        let apps = crossoverApps
        let choose = String(localized: "Choose…")
        // A copy picked by hand is used as is, never swapped for another one.
        let candidates = chosenCrossOver.map { [$0] } ?? apps
        if let app = candidates.first(where: { d3dmetal(in: $0) != nil }) {
            let version = (NSDictionary(contentsOf: app.appendingPathComponent("Contents/Info.plist"))?["CFBundleShortVersionString"] as? String).map { " \($0)" } ?? ""
            let folder = (app.deletingLastPathComponent().path as NSString).abbreviatingWithTildeInPath
            var requirement = Requirement(kind: .crossover, title: title, ok: true,
                                          detail: "\(app.deletingPathExtension().lastPathComponent)\(version) · \(folder)",
                                          location: app)
            if apps.count > 1 {
                requirement.fix = .chooseCrossOver
                requirement.fixTitle = String(localized: "Change…")
            }
            return requirement
        }
        let detail = apps.isEmpty
            ? String(localized: "Runs the Windows game. Not found in /Applications or ~/Applications: install it, or choose where it is.")
            : String(localized: "This CrossOver has no D3DMetal. Update to version 24 or newer, or choose another copy.")
        return Requirement(kind: .crossover, title: title, ok: false, detail: detail,
                           fix: .chooseCrossOver, fixTitle: choose,
                           secondaryFix: .open(crossoverSite), secondaryFixTitle: String(localized: "Download"))
    }

    /// A wineserver runs while any CrossOver bottle is open.
    static func bottleRunning() -> Bool {
        runCommand("/usr/bin/pgrep", ["-x", "wineserver"]).ok
    }
}

/// The Mac's chip and which FSR backend it gets (see src/fsr4_device.h).
enum Chip {
    static var name: String {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "Mac" }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "Mac" }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    /// Apple GPU family 10 (M5 and newer) runs FSR's INT8 network natively.
    static var hasNeuralAccelerators: Bool {
        guard let device = MTLCreateSystemDefaultDevice(),
              let family = MTLGPUFamily(rawValue: 1010) else { return false }
        return device.supportsFamily(family)
    }
}
