import AppKit
import Observation

enum FSRProfile: String, CaseIterable, Identifiable {
    case lite = "int8-lite"
    case lossy = "lossy2"
    case full = "exact-fast"

    var id: String { rawValue }
    var version: String { self == .lite ? "4.1.2" : "4.1.1" }
}

struct Banner: Equatable {
    enum Style { case success, failure, info }

    let style: Style
    let title: String
    let message: String
}

@MainActor
@Observable
final class InstallerModel {
    private(set) var requirements: [Requirement] = []
    private(set) var bottles: [Bottle] = []
    private(set) var checking = false
    var selectedID: String? = nil
    var profile: FSRProfile = .lite
    var panel = true

    /// Title of the running action and its current step; nil when idle.
    private(set) var activity: String? = nil
    private(set) var step = ""
    private(set) var cancelling = false
    var banner: Banner? = nil
    private(set) var log = ""
    var showLog = false

    let root = Payload.installRoot
    let payload: BundledPayload?
    private var command: StreamingCommand?
    private var runStart: String.Index?
    private var extraBottleConfigs: [URL] = []

    init(payload: BundledPayload? = Payload.bundled()) {
        self.payload = payload
    }

    var selected: Bottle? { bottles.first { $0.id == selectedID } }
    var requirementsMet: Bool { !requirements.isEmpty && requirements.allSatisfy(\.ok) }
    var busy: Bool { activity != nil }
    var canInstall: Bool { requirementsMet && payload != nil && selected != nil && !busy }
    var logURL: URL { root.appendingPathComponent("dist/crossover/hook.log") }
    var panelApp: URL { root.appendingPathComponent("dist/crossover/FSR4 Panel.app") }

    // MARK: State

    func refresh() {
        guard !checking else { return }
        checking = true
        Task.detached(priority: .userInitiated) {
            let requirements = SystemCheck.all()
            let bottles = Bottles.list()
            await MainActor.run {
                self.requirements = requirements
                self.apply(bottles)
                self.checking = false
            }
        }
    }

    private func reloadBottles() { apply(Bottles.list()) }

    private func apply(_ list: [Bottle]) {
        let previous = selectedID
        let added = extraBottleConfigs.compactMap(Bottles.bottle(conf:))
        var seen: Set<String> = []
        bottles = (list + added).filter { seen.insert($0.conf.resolvingSymlinksInPath().path).inserted }
        if selected == nil {
            selectedID = (bottles.first(where: \.installed) ?? bottles.first)?.id
        }
        if selectedID != previous { syncOptions() }
    }

    /// Shows the selected bottle's current mode and panel setting.
    func syncOptions() {
        guard let bottle = selected, bottle.installed else { return }
        if let value = bottle.profile, let mode = FSRProfile(rawValue: value) { profile = mode }
        panel = bottle.panel
    }

    func ownsInstall(_ bottle: Bottle) -> Bool {
        guard let path = bottle.root else { return false }
        // crossover_fsr4.py writes FSR4_ROOT with symlinks resolved.
        return URL(fileURLWithPath: path).resolvingSymlinksInPath().path == root.resolvingSymlinksInPath().path
    }

    // MARK: Actions

    func install() {
        guard canInstall, let bottle = selected else { return }
        var arguments = ["install", "--bottle", bottle.conf.path, "--fsr-version", profile.version, "--profile", profile.rawValue]
        if panel { arguments.append("--panel") }
        // The same CrossOver the requirements list shows, not the script's own pick.
        if let app = requirements.first(where: { $0.kind == .crossover })?.location {
            arguments += ["--crossover", app.path]
        }
        runInstaller(String(localized: "Installing into “\(bottle.name)”"), arguments) { [weak self] ok, cancelled in
            guard let self else { return }
            if ok {
                self.banner = Banner(style: .success, title: String(localized: "FSR 4 is installed in “\(bottle.name)”"),
                                     message: String(localized: "Start the game in CrossOver and select DLSS in its graphics settings. The first launch compiles shaders for the game's resolution, which can take a minute."))
            } else {
                self.reportFailure(cancelled: cancelled, String(localized: "Installation failed"))
            }
        }
    }

    func restore(_ bottle: Bottle) {
        guard !busy, payload != nil else { return }
        runInstaller(String(localized: "Removing from “\(bottle.name)”"), ["restore", "--bottle", bottle.conf.path]) { [weak self] ok, cancelled in
            guard let self else { return }
            if ok {
                self.banner = Banner(style: .success, title: String(localized: "FSR 4 is removed from “\(bottle.name)”"),
                                     message: String(localized: "The bottle uses CrossOver's own D3DMetal again from its next launch."))
            } else {
                self.reportFailure(cancelled: cancelled, String(localized: "Removal failed"))
            }
        }
    }

    func fix(_ requirement: Requirement) {
        guard let fix = requirement.fix else { return }
        self.fix(fix, for: requirement)
    }

    func fix(_ fix: Requirement.Fix, for requirement: Requirement) {
        guard !busy else { return }
        switch fix {
        case .open(let url):
            NSWorkspace.shared.open(url)
        case .chooseCrossOver:
            chooseCrossOver()
        case .administrator(let command):
            begin(requirement.title)
            step = String(localized: "Waiting for your password…")
            appendLog("$ sudo \(command)\n")
            Task.detached {
                let result = runAdministrator(command)
                await MainActor.run {
                    self.appendLog(result.output.isEmpty ? "" : result.output + "\n")
                    self.finish()
                    // -128: the password prompt was cancelled
                    if !result.ok && !result.output.contains("-128") {
                        self.reportFailure(cancelled: false, String(localized: "\(requirement.title): setup failed"))
                    }
                    self.refresh()
                }
            }
        case .stream(let argv):
            begin(requirement.title)
            step = String(localized: "Downloading…")
            stream(argv, directory: nil) { [weak self] ok, cancelled in
                guard let self else { return }
                if !ok { self.reportFailure(cancelled: cancelled, String(localized: "\(requirement.title): download failed")) }
                self.refresh()
            }
        }
    }

    func cancel() {
        guard let command, !cancelling else { return }
        cancelling = true
        step = String(localized: "Cancelling…")
        command.cancel()
    }

    /// Picks the CrossOver.app to copy D3DMetal from, wherever it is and
    /// whatever it is called. Remembered until that app is gone.
    func chooseCrossOver() {
        let panel = NSOpenPanel()
        panel.message = String(localized: "Choose the CrossOver app you start your games with.")
        panel.prompt = String(localized: "Choose")
        panel.allowedContentTypes = [.applicationBundle]
        panel.canChooseDirectories = false
        panel.treatsFilePackagesAsDirectories = false
        panel.directoryURL = (SystemCheck.crossoverApps.first?.deletingLastPathComponent())
            ?? URL(fileURLWithPath: "/Applications")
        guard panel.runModal() == .OK, let app = panel.url else { return }
        guard SystemCheck.isCrossOver(app) else {
            banner = Banner(style: .failure, title: String(localized: "Not a CrossOver app"),
                            message: String(localized: "\(app.lastPathComponent) has no CrossOver inside. Choose CrossOver.app or its renamed copy."))
            return
        }
        SystemCheck.chosenCrossOver = app
        refresh()
    }

    func chooseBottle() {
        let panel = NSOpenPanel()
        panel.message = String(localized: "Choose the bottle folder containing cxbottle.conf, including on an external disk.")
        panel.prompt = String(localized: "Choose")
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.directoryURL = URL(fileURLWithPath: "/Volumes")
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        let conf = folder.appendingPathComponent("cxbottle.conf")
        guard let bottle = Bottles.bottle(conf: conf) else {
            banner = Banner(style: .failure, title: String(localized: "Not a CrossOver bottle"),
                            message: String(localized: "The selected folder has no readable cxbottle.conf."))
            return
        }
        extraBottleConfigs.append(conf)
        reloadBottles()
        selectedID = bottles.first { $0.conf.resolvingSymlinksInPath() == bottle.conf.resolvingSymlinksInPath() }?.id
        syncOptions()
    }

    func openPanel() { NSWorkspace.shared.open(panelApp) }
    func openLog() { NSWorkspace.shared.open(logURL) }
    func revealFolder() { NSWorkspace.shared.activateFileViewerSelecting([root]) }
    func copyLog() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(log, forType: .string)
    }

    // MARK: Running commands

    private func begin(_ title: String) {
        banner = nil
        activity = title
        step = ""
        cancelling = false
        if !log.isEmpty { log += "\n" }
        runStart = log.endIndex
    }

    private func finish() {
        activity = nil
        step = ""
        cancelling = false
        command = nil
    }

    /// Unpacks the bundled FSR files (if needed), then runs crossover_fsr4.py.
    private func runInstaller(_ title: String, _ arguments: [String], completion: @escaping @MainActor (Bool, Bool) -> Void) {
        guard let payload else { return }
        begin(title)
        step = String(localized: "Unpacking FSR files…")
        let root = root
        Task.detached {
            let result = Result { try Payload.ensure(root: root, payload: payload) }
            await MainActor.run {
                switch result {
                case .failure(let error):
                    self.appendLog(error.localizedDescription + "\n")
                    self.finish()
                    completion(false, false)
                case .success(let unpacked):
                    if unpacked { self.appendLog("FSR \(payload.info.version) → \(root.path)\n") }
                    let script = Payload.script(in: root).path
                    self.stream(["/usr/bin/python3", script] + arguments, directory: root, completion: completion)
                }
            }
        }
    }

    private func stream(_ argv: [String], directory: URL?, completion: @escaping @MainActor (Bool, Bool) -> Void) {
        appendLog("$ " + argv.map { $0.contains(" ") ? shellQuote($0) : $0 }.joined(separator: " ") + "\n")
        let command = StreamingCommand(argv, directory: directory)
        self.command = command
        do {
            try command.start(output: { [weak self] text in
                self?.received(text)
            }, completion: { [weak self] status, cancelled in
                guard let self else { return }
                self.finish()
                self.reloadBottles()
                completion(status == 0 && !cancelled, cancelled)
            })
        } catch {
            appendLog(error.localizedDescription + "\n")
            finish()
            completion(false, false)
        }
    }

    private func received(_ text: String) {
        appendLog(text)
        guard !cancelling else { return }
        for line in text.split(separator: "\n") where line.hasPrefix("+ ") {
            if let next = Self.describe(line.dropFirst(2)) { step = next }
        }
    }

    private func appendLog(_ text: String) {
        log += text
        // Keep the text view responsive: drop the oldest output past ~400 KB.
        if log.utf8.count > 400_000 {
            log = String(log.suffix(300_000))
            runStart = nil
        }
    }

    private func reportFailure(cancelled: Bool, _ title: String) {
        if cancelled {
            banner = Banner(style: .info, title: String(localized: "Cancelled"),
                            message: String(localized: "Nothing was changed in the bottle."))
            return
        }
        let output = runStart.map { String(log[$0...]) } ?? log
        let last = output.split(separator: "\n").last { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        banner = Banner(style: .failure, title: title,
                        message: last.map { String($0.prefix(300)) } ?? String(localized: "See the log for details."))
        showLog = true
    }

    /// Friendly name of a crossover_fsr4.py step from its "+ command" echo.
    static func describe(_ command: Substring) -> String? {
        if command.contains("compile_fsr411") || (command.contains("--profile") && command.contains("fsr4mac")) {
            return String(localized: "Compiling FSR shaders (a few minutes the first time)…")
        }
        if command.contains("live_test") { return String(localized: "Checking the result against the reference…") }
        if command.contains("FSR4 Panel") || command.contains("iconutil") { return String(localized: "Building the FSR 4 panel…") }
        if command.contains("D3DMetal") { return String(localized: "Patching a copy of D3DMetal…") }
        if command.contains("smoke") || command.contains("dlopen_test") { return String(localized: "Testing the hook…") }
        if command.contains("make") || command.contains("--backend-info") { return String(localized: "Preparing the FSR tool…") }
        if command.hasPrefix("clang") || command.contains("libfsr4hook") { return String(localized: "Building the hook…") }
        return nil
    }

    /// Whether a bottle is open; its settings are only read at start.
    func bottleRunning() async -> Bool {
        await Task.detached { SystemCheck.bottleRunning() }.value
    }
}
