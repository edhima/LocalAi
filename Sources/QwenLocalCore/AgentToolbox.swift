//
//  LocalAi — AI locale su Apple Silicon (chat, agente, RAG, training, immagini)
//  © 2026 Eridon Dhima · e.dhima@alpha-soft.al · +355 69 600 0666
//

import Darwin
import Foundation
import MLXLMCommon

/// Gli strumenti che l'agente può usare, confinati a una cartella di lavoro (workspace).
/// Le descrizioni sono in inglese: è la lingua su cui i modelli rendono meglio con le tool call.
public struct AgentToolbox: Sendable {
    public let workspace: URL
    /// Se impostato, write_file ed edit_file possono toccare SOLO questa cartella
    /// (dentro il workspace). La lettura resta libera in tutto il workspace.
    public let writeBoundary: URL?
    /// Ricerca semantica sui documenti indicizzati (RAG), se un indice è disponibile.
    let semanticSearch: (@Sendable (String) async throws -> String)?

    public init(
        workspace: URL,
        writeBoundary: URL? = nil,
        semanticSearch: (@Sendable (String) async throws -> String)? = nil
    ) {
        self.workspace = workspace
        self.writeBoundary = writeBoundary
        self.semanticSearch = semanticSearch
    }

    public struct ToolboxError: LocalizedError {
        let message: String
        public init(_ message: String) { self.message = message }
        public var errorDescription: String? { message }
    }

    /// Limite dei caratteri restituiti al modello da un singolo tool.
    private static let maxOutput = 16_000
    private static let excludedDirs: Set<String> = [".git", ".build", "node_modules", ".DerivedData", ".localai"]

    // MARK: - Input dei tool (decodificati dagli argomenti JSON del modello)

    struct ReadInput: Codable {
        var path: String
        var offset: Int?
        var limit: Int?
    }
    struct WriteInput: Codable {
        var path: String
        var content: String
    }
    struct EditInput: Codable {
        var path: String
        var old_string: String
        var new_string: String
        var replace_all: Bool?
    }
    struct ListInput: Codable {
        var path: String?
    }
    struct GlobInput: Codable {
        var pattern: String
    }
    struct GrepInput: Codable {
        var pattern: String
        var path: String?
    }
    struct BashInput: Codable {
        var command: String
    }
    struct SearchInput: Codable {
        var query: String
    }

    // MARK: - Catalogo tool

    /// Tool il cui uso modifica lo stato (file o sistema): richiedono approvazione.
    public static let toolsNeedingApproval: Set<String> = ["write_file", "edit_file", "bash"]

    public static func icon(for toolName: String) -> String {
        switch toolName {
        case "read_file": "doc.text"
        case "write_file": "square.and.pencil"
        case "edit_file": "pencil.line"
        case "list_directory": "folder"
        case "glob": "doc.text.magnifyingglass"
        case "grep": "magnifyingglass"
        case "bash": "terminal"
        case "search_documents": "sparkle.magnifyingglass"
        default: "wrench.and.screwdriver"
        }
    }

    private struct Entry: @unchecked Sendable {
        let name: String
        let schema: ToolSpec
        let run: @Sendable (ToolCall) async throws -> String
    }

    private var entries: [Entry] {
        let workspace = self.workspace

        let read = Tool<ReadInput, String>(
            name: "read_file",
            description:
                "Read a text file from the workspace. Returns the content with line numbers. Use offset/limit for large files.",
            parameters: [
                .required("path", type: .string, description: "File path relative to the workspace root"),
                .optional("offset", type: .int, description: "1-based line number to start reading from"),
                .optional("limit", type: .int, description: "Maximum number of lines to return (default 2000)"),
            ]
        ) { input in
            let url = try Self.resolve(input.path, in: workspace)
            let text = try String(contentsOf: url, encoding: .utf8)
            let lines = text.components(separatedBy: "\n")
            let start = max((input.offset ?? 1) - 1, 0)
            let count = min(input.limit ?? 2000, 5000)
            guard start < lines.count else {
                throw ToolboxError("Offset \(start + 1) beyond end of file (\(lines.count) lines).")
            }
            let slice = lines[start ..< min(start + count, lines.count)]
            let numbered = slice.enumerated()
                .map { String(format: "%5d\t%@", start + $0.offset + 1, $0.element) }
                .joined(separator: "\n")
            return Self.truncate(numbered)
        }

        let write = Tool<WriteInput, String>(
            name: "write_file",
            description:
                "Create or overwrite a file in the workspace with the given content. Creates intermediate directories if needed.",
            parameters: [
                .required("path", type: .string, description: "File path relative to the workspace root"),
                .required("content", type: .string, description: "Full content to write to the file"),
            ]
        ) { [writeBoundary] input in
            let url = try Self.resolve(input.path, in: workspace)
            try Self.checkWriteBoundary(url, boundary: writeBoundary)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try input.content.write(to: url, atomically: true, encoding: .utf8)
            return "Wrote \(input.content.utf8.count) bytes to \(input.path)."
        }

        let edit = Tool<EditInput, String>(
            name: "edit_file",
            description:
                "Replace an exact string in a file. The old_string must match exactly (including whitespace) and must be unique in the file unless replace_all is true.",
            parameters: [
                .required("path", type: .string, description: "File path relative to the workspace root"),
                .required("old_string", type: .string, description: "Exact text to find"),
                .required("new_string", type: .string, description: "Replacement text"),
                .optional("replace_all", type: .bool, description: "Replace every occurrence (default false)"),
            ]
        ) { [writeBoundary] input in
            let url = try Self.resolve(input.path, in: workspace)
            try Self.checkWriteBoundary(url, boundary: writeBoundary)
            let text = try String(contentsOf: url, encoding: .utf8)
            let occurrences = text.components(separatedBy: input.old_string).count - 1
            guard occurrences > 0 else {
                throw ToolboxError("old_string not found in \(input.path). Read the file again and retry with the exact text.")
            }
            guard occurrences == 1 || input.replace_all == true else {
                throw ToolboxError("old_string occurs \(occurrences) times in \(input.path). Provide more context to make it unique, or set replace_all.")
            }
            let updated = text.replacingOccurrences(of: input.old_string, with: input.new_string)
            try updated.write(to: url, atomically: true, encoding: .utf8)
            return "Replaced \(occurrences) occurrence(s) in \(input.path)."
        }

        let list = Tool<ListInput, String>(
            name: "list_directory",
            description: "List files and subdirectories of a directory in the workspace.",
            parameters: [
                .optional("path", type: .string, description: "Directory path relative to the workspace root (default: the root)")
            ]
        ) { input in
            let url = try Self.resolve(input.path ?? ".", in: workspace)
            let items = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: [.isDirectoryKey], options: [])
            let listing = items
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
                .prefix(500)
                .map { item -> String in
                    let isDir = (try? item.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                    return item.lastPathComponent + (isDir ? "/" : "")
                }
                .joined(separator: "\n")
            return listing.isEmpty ? "(empty directory)" : Self.truncate(listing)
        }

        let glob = Tool<GlobInput, String>(
            name: "glob",
            description:
                "Find files by name pattern (e.g. \"*.swift\" or \"Sources/**/*.swift\"). Returns paths relative to the workspace root.",
            parameters: [
                .required("pattern", type: .string, description: "Glob pattern matched against relative paths and file names")
            ]
        ) { input in
            let matches = Self.globMatches(pattern: input.pattern, in: workspace)
            return matches.isEmpty
                ? "No files match \(input.pattern)."
                : Self.truncate(matches.prefix(200).joined(separator: "\n"))
        }

        let grep = Tool<GrepInput, String>(
            name: "grep",
            description:
                "Search file contents with a regular expression (grep -rn). Returns matching lines as path:line:text.",
            parameters: [
                .required("pattern", type: .string, description: "Regular expression to search for"),
                .optional("path", type: .string, description: "File or directory to search in (default: whole workspace)"),
            ]
        ) { input in
            let target = try Self.resolve(input.path ?? ".", in: workspace)
            var args = ["-rn", "-I", "-e", input.pattern]
            for dir in Self.excludedDirs { args.append("--exclude-dir=\(dir)") }
            args.append(target.path)
            let (status, output) = try await Self.runProcess("/usr/bin/grep", args, cwd: workspace, timeout: 30)
            if status == 1 { return "No matches." }
            guard status == 0 else { throw ToolboxError("grep failed: \(output)") }
            // Riporta i path relativi al workspace per coerenza con gli altri tool.
            let relative = output.replacingOccurrences(of: workspace.path + "/", with: "")
            return Self.truncate(relative)
        }

        let bash = Tool<BashInput, String>(
            name: "bash",
            description:
                "Run a shell command (zsh) with the workspace as the working directory. Returns stdout and stderr. Timeout: 120 seconds.",
            parameters: [
                .required("command", type: .string, description: "The shell command to execute")
            ]
        ) { input in
            let (status, output) = try await Self.runProcess(
                "/bin/zsh", ["-c", input.command], cwd: workspace, timeout: 120)
            var result = output.isEmpty ? "(no output)" : output
            if status != 0 {
                result += "\n[exit code: \(status)]"
            }
            return Self.truncate(result)
        }

        var tools = [
            wrap(read), wrap(write), wrap(edit), wrap(list), wrap(glob), wrap(grep), wrap(bash),
        ]

        if let semanticSearch {
            let search = Tool<SearchInput, String>(
                name: "search_documents",
                description:
                    "Semantic search over the indexed workspace documents. Finds relevant passages by MEANING, not exact words — prefer this over grep for questions about document content. Returns the most relevant passages with their file paths.",
                parameters: [
                    .required("query", type: .string, description: "Natural-language description of what to find")
                ]
            ) { input in
                try await semanticSearch(input.query)
            }
            tools.append(wrap(search))
        }

        return tools
    }

    private func wrap<I: Codable, O: Codable>(_ tool: Tool<I, O>) -> Entry {
        Entry(name: tool.name, schema: tool.schema) { call in
            let output = try await call.execute(with: tool)
            return output as? String ?? String(describing: output)
        }
    }

    /// Gli schemi JSON da passare a `ChatSession(tools:)`.
    public var specs: [ToolSpec] {
        entries.map(\.schema)
    }

    /// Esegue la tool call richiesta dal modello.
    public func dispatch(_ call: ToolCall) async throws -> String {
        guard let entry = entries.first(where: { $0.name == call.function.name }) else {
            throw ToolboxError("Unknown tool: \(call.function.name)")
        }
        return try await entry.run(call)
    }

    // MARK: - Helpers

    /// Risolve un path (relativo o assoluto) e verifica che resti dentro il workspace.
    /// I symlink vanno risolti su entrambi i lati del confronto (es. /var → /private/var),
    /// altrimenti lo stesso percorso può sembrare fuori dal workspace.
    static func resolve(_ path: String, in workspace: URL) throws -> URL {
        // La base DEVE avere lo slash finale (isDirectory: true), altrimenti
        // i path relativi vengono risolti rispetto alla directory padre.
        let base = URL(fileURLWithPath: workspace.path, isDirectory: true)
        let url = URL(fileURLWithPath: path, relativeTo: base)
            .standardizedFileURL.resolvingSymlinksInPath()
        let root = base.standardizedFileURL.resolvingSymlinksInPath().path
        guard url.path == root || url.path.hasPrefix(root + "/") else {
            throw ToolboxError("Path '\(path)' is outside the workspace. Only paths inside \(root) are allowed.")
        }
        return url
    }

    /// Verifica che una scrittura ricada nel confine autorizzato, se impostato.
    static func checkWriteBoundary(_ url: URL, boundary: URL?) throws {
        guard let boundary else { return }
        let root = boundary.standardizedFileURL.resolvingSymlinksInPath().path
        let target = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard target == root || target.hasPrefix(root + "/") else {
            throw ToolboxError(
                "Writes are only allowed inside '\(boundary.lastPathComponent)/' (\(root)). Reading is allowed everywhere in the workspace.")
        }
    }

    static func truncate(_ text: String) -> String {
        guard text.count > maxOutput else { return text }
        return text.prefix(maxOutput) + "\n… [output truncated: \(text.count) characters total]"
    }

    static func globMatches(pattern: String, in workspace: URL) -> [String] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: workspace,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var results: [String] = []
        let rootPath = workspace.standardizedFileURL.path
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            if excludedDirs.contains(name) {
                enumerator.skipDescendants()
                continue
            }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                continue
            }
            var relative = url.standardizedFileURL.path
            if relative.hasPrefix(rootPath + "/") {
                relative = String(relative.dropFirst(rootPath.count + 1))
            }
            // Confronta sia il path relativo sia il solo nome file; "**/" è trattato
            // come prefisso qualsiasi (fnmatch non supporta il globstar).
            let simplified = pattern.replacingOccurrences(of: "**/", with: "*")
            if fnmatch(simplified, relative, 0) == 0 || fnmatch(simplified, name, 0) == 0 {
                results.append(relative)
            }
            if results.count >= 1000 { break }
        }
        return results.sorted()
    }

    /// Esegue un processo esterno con timeout, unendo stdout e stderr.
    static func runProcess(
        _ executable: String, _ arguments: [String], cwd: URL, timeout: TimeInterval
    ) async throws -> (Int32, String) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = arguments
                process.currentDirectoryURL = cwd
                var environment = ProcessInfo.processInfo.environment
                environment["TERM"] = "dumb"
                process.environment = environment

                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe
                process.standardInput = FileHandle.nullDevice

                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }

                // Watchdog: termina il processo se supera il timeout.
                let deadline = Date().addingTimeInterval(timeout)
                DispatchQueue.global(qos: .utility).async {
                    while process.isRunning, Date() < deadline {
                        Thread.sleep(forTimeInterval: 0.2)
                    }
                    if process.isRunning {
                        process.terminate()
                    }
                }

                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                var output = String(data: data, encoding: .utf8) ?? ""
                if Date() >= deadline {
                    output += "\n[terminated: timeout after \(Int(timeout))s]"
                }
                continuation.resume(returning: (process.terminationStatus, output))
            }
        }
    }
}