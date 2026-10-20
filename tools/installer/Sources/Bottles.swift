import Foundation

/// A CrossOver bottle and the fsr4-mac keys found in its cxbottle.conf.
struct Bottle: Identifiable, Hashable, Sendable {
    let name: String
    let conf: URL
    /// D3DMETAL_FRAMEWORK_PATH is set: the bottle loads the patched D3DMetal.
    let installed: Bool
    /// FSR4_ROOT: the folder the bottle's FSR files come from.
    let root: String?
    /// FSR4_PROFILE: lossy (FSR 4.1 Lite), exact-fast (FSR 4.1), ...
    let profile: String?
    let panel: Bool

    var id: String { conf.path }
}

enum Bottles {
    /// Set FSR4_INSTALLER_BOTTLES to list another folder (used by tests).
    static var directory: URL {
        if let custom = ProcessInfo.processInfo.environment["FSR4_INSTALLER_BOTTLES"], !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/CrossOver/Bottles")
    }

    /// The same BottlePath / CX_BOTTLE_PATH search used by CrossOver itself.
    static func searchDirectories(base: URL, apps: [URL], environment: [String: String]) -> [URL] {
        var environment = environment
        var bottlePath = ""
        let configs = apps.prefix(1).map {
            $0.appendingPathComponent("Contents/SharedSupport/CrossOver/etc/CrossOver.conf")
        } + [base.deletingLastPathComponent().appendingPathComponent("CrossOver.conf")]
        for config in configs {
            guard let text = try? String(contentsOf: config, encoding: .utf8) else { continue }
            var section = ""
            for line in text.split(whereSeparator: \.isNewline) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix(";") { continue }
                if trimmed.hasPrefix("["), let end = trimmed.firstIndex(of: "]") {
                    section = String(trimmed[trimmed.index(after: trimmed.startIndex)..<end]).lowercased()
                    continue
                }
                guard let field = configField(trimmed) else { continue }
                if section == "crossover", field.0.lowercased() == "bottlepath" {
                    bottlePath = field.1
                } else if section == "environmentvariables" {
                    environment[field.0] = expand(field.1, environment: environment)
                }
            }
        }
        let search = environment["CX_BOTTLE_PATH"].flatMap { $0.isEmpty ? nil : $0 } ?? bottlePath
        if search.isEmpty { return [base] }
        var paths: [URL] = []
        for item in search.split(separator: ":") {
            let path = (expand(String(item), environment: environment) as NSString).expandingTildeInPath
            guard path.hasPrefix("/") else { continue }
            let url = URL(fileURLWithPath: path)
            if !paths.contains(url) { paths.append(url) }
        }
        return paths
    }

    private static func expand(_ text: String, environment: [String: String]) -> String {
        let pattern = try! NSRegularExpression(pattern: #"\$\{([^}]+)\}"#)
        var result = text
        for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let whole = Range(match.range, in: result), let key = Range(match.range(at: 1), in: text) else { continue }
            result.replaceSubrange(whole, with: environment[String(text[key])] ?? "")
        }
        return result
    }

    /// Quoted CrossOver fields may contain escaped quotes/backslashes and a trailing comment.
    private static func configField(_ line: String) -> (String, String)? {
        let pattern = try! NSRegularExpression(pattern: #"^"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:;.*)?$"#)
        guard let match = pattern.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
              let key = Range(match.range(at: 1), in: line), let value = Range(match.range(at: 2), in: line) else { return nil }
        func unescape(_ text: String) -> String {
            text.replacingOccurrences(of: #"\""#, with: "\"").replacingOccurrences(of: #"\\"#, with: #"\"#)
        }
        return (unescape(String(line[key])), unescape(String(line[value])))
    }

    /// `"KEY" = "VALUE"` lines of a cxbottle.conf. The last occurrence wins.
    static func values(in text: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("\"") else { continue }
            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, let key = unquote(parts[0]), let value = unquote(parts[1]) else { continue }
            result[key] = value
        }
        return result
    }

    static func bottle(conf: URL) -> Bottle? {
        guard let data = try? Data(contentsOf: conf) else { return nil }
        let values = values(in: String(decoding: data, as: UTF8.self))
        return Bottle(
            name: conf.deletingLastPathComponent().lastPathComponent,
            conf: conf,
            installed: values["D3DMETAL_FRAMEWORK_PATH"] != nil,
            root: values["FSR4_ROOT"],
            profile: values["FSR4_PROFILE"],
            panel: values["FSR4_PANEL"] == "1")
    }

    static func list(in directory: URL? = nil) -> [Bottle] {
        let environment = ProcessInfo.processInfo.environment
        let roots: [URL]
        if let directory {
            roots = [directory]
        } else if environment["FSR4_INSTALLER_BOTTLES"]?.isEmpty == false {
            roots = [self.directory]
        } else {
            roots = searchDirectories(base: self.directory, apps: SystemCheck.crossoverApps, environment: environment)
        }
        return list(in: roots)
    }

    static func list(in directories: [URL]) -> [Bottle] {
        var seen: Set<String> = []
        var result: [Bottle] = []
        for directory in directories {
            let items = (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
            for item in items {
                let conf = item.appendingPathComponent("cxbottle.conf")
                if let bottle = bottle(conf: conf), seen.insert(conf.resolvingSymlinksInPath().path).inserted {
                    result.append(bottle)
                }
            }
        }
        return result.sorted {
            let order = $0.name.localizedStandardCompare($1.name)
            return order == .orderedSame ? $0.conf.path < $1.conf.path : order == .orderedAscending
        }
    }

    private static func unquote(_ text: Substring) -> String? {
        let value = text.trimmingCharacters(in: .whitespaces)
        guard value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") else { return nil }
        return String(value.dropFirst().dropLast())
    }
}
