import AppKit
import SwiftUI

struct ContentView: View {
    @Bindable var model: InstallerModel
    @State private var confirmRunning = false
    @State private var confirmRemove: Bottle?

    static var guideURL: URL {
        let russian = Locale.preferredLanguages.first?.hasPrefix("ru") == true
        return URL(string: russian
            ? "https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/blob/main/docs/GUIDE.md"
            : "https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos#readme")!
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Form {
                requirementsSection
                bottlesSection
                optionsSection
                logSection
            }
            .formStyle(.grouped)
            Divider()
            // Pinned so the main action and its result never scroll away.
            VStack(spacing: 10) {
                if let banner = model.banner {
                    BannerView(banner: banner) { model.banner = nil }
                }
                actionRow
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            footer
        }
        .alert("A CrossOver bottle is running", isPresented: $confirmRunning) {
            Button("Install Anyway") { model.install() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Quit the game and close its bottle first: a bottle reads its settings only when it starts.")
        }
        .alert(item: $confirmRemove) { bottle in
            Alert(title: Text("Remove FSR 4 from “\(bottle.name)”?"),
                  message: Text("The bottle goes back to CrossOver's own D3DMetal. Games and saves are not touched."),
                  primaryButton: .destructive(Text("Remove")) { model.restore(bottle) },
                  secondaryButton: .cancel())
        }
        .onChange(of: model.selectedID) { model.syncOptions() }
        .task { model.refresh() }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 14) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 56, height: 56)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text("FSR 4 for CrossOver")
                    .font(.title2.bold())
                Text("AMD FSR 4.1.2 instead of DLSS in Windows games")
                    .foregroundStyle(.secondary)
                Text(chipLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var chipLine: String {
        let version = model.payload?.info.version ?? "—"
        let backend = Chip.hasNeuralAccelerators
            ? String(localized: "native INT8 on the neural accelerators")
            : String(localized: "half-matrix INT8 backend")
        return "\(version) · \(Chip.name) · \(backend)"
    }

    // MARK: Requirements

    private var requirementsSection: some View {
        Section {
            if model.requirements.isEmpty {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Checking this Mac…").foregroundStyle(.secondary)
                }
            }
            ForEach(model.requirements) { requirement in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    StatusIcon(ok: requirement.ok)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(requirement.title)
                        Text(requirement.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    if let fix = requirement.secondaryFix, let title = requirement.secondaryFixTitle {
                        Button(title) { model.fix(fix, for: requirement) }
                            .disabled(model.busy)
                    }
                    if let fix = requirement.fix, let title = requirement.fixTitle {
                        Button(title) { model.fix(fix, for: requirement) }
                            .disabled(model.busy)
                    }
                }
                .accessibilityElement(children: .combine)
            }
        } header: {
            HStack {
                Text("Requirements")
                Spacer()
                Button {
                    model.refresh()
                } label: {
                    Label("Check Again", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .labelStyle(.iconOnly)
                .help("Check Again")
                .disabled(model.checking || model.busy)
            }
        }
    }

    // MARK: Bottles

    private var bottlesSection: some View {
        Section("CrossOver bottle") {
            if model.bottles.isEmpty {
                Text("No bottles yet. Create one in CrossOver, install the game there, then come back.")
                    .foregroundStyle(.secondary)
            } else {
                Picker("Bottle", selection: $model.selectedID) {
                    ForEach(model.bottles) { bottle in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(bottle.name)
                            Text(bottle.conf.deletingLastPathComponent().path)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(status(of: bottle))
                                .font(.caption)
                                .foregroundStyle(bottle.installed ? (model.ownsInstall(bottle) ? .green : .orange) : .secondary)
                        }
                        .tag(Optional(bottle.id))
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
                .disabled(model.busy)
            }
            Button("Choose bottle folder…") { model.chooseBottle() }
                .disabled(model.busy)
        }
    }

    private func status(of bottle: Bottle) -> String {
        guard bottle.installed else { return String(localized: "FSR 4 not installed") }
        guard model.ownsInstall(bottle) else {
            return String(localized: "Installed from another folder (\(bottle.root ?? "?")). Reinstall to switch to this app.")
        }
        switch bottle.profile {
        case FSRProfile.full.rawValue: return String(localized: "Installed · FSR 4.1")
        case "exact": return String(localized: "Installed · FSR 4.1 (bit-exact)")
        case "fsr4": return String(localized: "Installed · FSR 4")
        case FSRProfile.lossy.rawValue: return String(localized: "Installed · FSR 4.1 Lossy")
        case FSRProfile.lite.rawValue: return String(localized: "Installed · FSR 4.1.2 Lite")
        default: return String(localized: "Installed · Previous FSR model")
        }
    }

    // MARK: Options and actions

    private var optionsSection: some View {
        Section {
            Picker("Starting mode", selection: $model.profile) {
                Text("FSR 4.1.2 Lite").tag(FSRProfile.lite)
                Text("FSR 4.1 Lossy").tag(FSRProfile.lossy)
                Text("FSR 4.1").tag(FSRProfile.full)
            }
            .pickerStyle(.segmented)
            .disabled(model.busy)
            Group {
                if model.profile == .lite {
                    Text("AMD's compact INT8 model. Runs the network every frame.")
                } else if model.profile == .lossy {
                    Text("Formerly Lite 2. Uses trained layers and reuses inference on suitable frames.")
                } else {
                    Text("Full network every frame, best quality.")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            Toggle("Show the FSR 4 panel next to the game", isOn: $model.panel)
                .disabled(model.busy)
            if model.panel {
                Text("Switch modes and sharpness while playing, no restart needed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var actionRow: some View {
        if let activity = model.activity {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text(activity).bold()
                    if !model.step.isEmpty {
                        Text(model.step).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Button("Cancel") { model.cancel() }
                    .disabled(model.cancelling)
            }
            .accessibilityElement(children: .combine)
        } else {
            HStack {
                if let hint = installHint {
                    Text(hint).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let bottle = model.selected, bottle.installed {
                    Button("Remove…", role: .destructive) { confirmRemove = bottle }
                }
                Button(installTitle) { startInstall() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!model.canInstall)
            }
        }
    }

    private var installTitle: String {
        guard let bottle = model.selected else { return String(localized: "Install") }
        return bottle.installed
            ? String(localized: "Reinstall in “\(bottle.name)”")
            : String(localized: "Install in “\(bottle.name)”")
    }

    private var installHint: String? {
        if model.payload == nil { return String(localized: "This build has no FSR files. Build it with tools/installer/build_app.py.") }
        if model.requirements.isEmpty { return nil }
        if !model.requirementsMet { return String(localized: "Fix the requirements above first.") }
        if model.selected == nil { return String(localized: "Pick a bottle.") }
        return nil
    }

    private func startInstall() {
        Task {
            if await model.bottleRunning() {
                confirmRunning = true
            } else {
                model.install()
            }
        }
    }

    // MARK: Log

    private var logSection: some View {
        Section {
            DisclosureGroup("Log", isExpanded: $model.showLog) {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(model.log.isEmpty ? String(localized: "Nothing yet.") : model.log)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .padding(6)
                        Color.clear.frame(height: 1).id("end")
                    }
                    .frame(minHeight: 140, maxHeight: 260)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 6))
                    .onChange(of: model.log) { proxy.scrollTo("end", anchor: .bottom) }
                    .onAppear { proxy.scrollTo("end", anchor: .bottom) }
                }
                HStack {
                    Spacer()
                    Button("Copy Log") { model.copyLog() }
                        .disabled(model.log.isEmpty)
                }
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: 12) {
            Button("Open FSR 4 Panel") { model.openPanel() }
                .disabled(!FileManager.default.fileExists(atPath: model.panelApp.path))
            Button("Game Log") { model.openLog() }
                .disabled(!FileManager.default.fileExists(atPath: model.logURL.path))
            Button("Show Files") { model.revealFolder() }
                .disabled(!FileManager.default.fileExists(atPath: model.root.path))
            Spacer()
            Link("Guide", destination: Self.guideURL)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }
}

private struct StatusIcon: View {
    let ok: Bool

    var body: some View {
        Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
            .foregroundStyle(ok ? .green : .orange)
            .accessibilityLabel(ok ? Text("Ready") : Text("Needs attention"))
    }
}

private struct BannerView: View {
    let banner: Banner
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .font(.title3)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(banner.title).bold()
                Text(banner.message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Button(action: dismiss) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(Text("Dismiss"))
        }
        .padding(10)
        .background(color.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .combine)
    }

    private var icon: String {
        switch banner.style {
        case .success: "checkmark.circle.fill"
        case .failure: "xmark.octagon.fill"
        case .info: "info.circle.fill"
        }
    }

    private var color: Color {
        switch banner.style {
        case .success: .green
        case .failure: .red
        case .info: .blue
        }
    }
}
