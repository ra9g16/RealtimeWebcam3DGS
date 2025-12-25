import Foundation
import os

/// Client for communicating with the SHARP Python server via Unix Domain Socket
actor SHARPClient {
    private static let log = Logger(subsystem: "com.metalsplatter.webcam3dgs", category: "SHARPClient")

    private let socketPath: String
    private var serverProcess: Process?

    struct GenerationResult: Codable, Sendable {
        let success: Bool
        let plyPath: String?
        let gaussianCount: Int
        let processingTimeMs: Double
        let metadata: Metadata?
        let error: String?

        struct Metadata: Codable, Sendable {
            let imageWidth: Int
            let imageHeight: Int
            let focalLengthPx: Double

            enum CodingKeys: String, CodingKey {
                case imageWidth = "image_width"
                case imageHeight = "image_height"
                case focalLengthPx = "focal_length_px"
            }
        }

        enum CodingKeys: String, CodingKey {
            case success
            case plyPath = "ply_path"
            case gaussianCount = "gaussian_count"
            case processingTimeMs = "processing_time_ms"
            case metadata
            case error
        }
    }

    /// Result for direct socket transfer (PLY data returned inline instead of file path)
    struct DirectGenerationResult: Sendable {
        let success: Bool
        let plyData: Data?
        let plySize: Int
        let gaussianCount: Int
        let processingTimeMs: Double
        let metadata: GenerationResult.Metadata?
        let error: String?
    }

    enum SHARPError: Error, LocalizedError {
        case serverNotRunning
        case connectionFailed(String)
        case invalidResponse
        case generationFailed(String)
        case timeout

        var errorDescription: String? {
            switch self {
            case .serverNotRunning:
                return "SHARP server is not running"
            case .connectionFailed(let message):
                return "Connection failed: \(message)"
            case .invalidResponse:
                return "Invalid response from server"
            case .generationFailed(let message):
                return "Generation failed: \(message)"
            case .timeout:
                return "Request timed out"
            }
        }
    }

    init(socketPath: String = "/tmp/webcam_3dgs/server.sock") {
        self.socketPath = socketPath
    }

    /// Check if the SHARP server is running
    func isServerRunning() async -> Bool {
        do {
            let response = try await sendCommand(["command": "ping"])
            return response["success"] as? Bool ?? false
        } catch {
            return false
        }
    }

    /// Generate a PLY file from an image
    func generatePLY(from imagePath: URL, outputPath: URL) async throws -> GenerationResult {
        Self.log.info("Requesting PLY generation from \(imagePath.path)")

        let request: [String: Any] = [
            "command": "generate",
            "input_path": imagePath.path,
            "output_path": outputPath.path,
            "options": ["device": "mps"]
        ]

        let response = try await sendCommand(request)

        let jsonData = try JSONSerialization.data(withJSONObject: response)
        let result = try JSONDecoder().decode(GenerationResult.self, from: jsonData)

        if !result.success, let error = result.error {
            throw SHARPError.generationFailed(error)
        }

        Self.log.info("PLY generated: \(result.gaussianCount) Gaussians in \(result.processingTimeMs)ms")
        return result
    }

    /// Generate PLY and receive data directly via socket (no file I/O)
    /// This eliminates file write/read overhead, saving ~100-200ms per generation.
    func generatePLYDirect(from imagePath: URL) async throws -> DirectGenerationResult {
        Self.log.info("Requesting PLY generation (direct) from \(imagePath.path)")

        let request: [String: Any] = [
            "command": "generate_direct",
            "input_path": imagePath.path,
        ]

        let response = try await sendCommand(request)

        let success = response["success"] as? Bool ?? false

        if !success {
            let errorMessage = response["error"] as? String ?? "Unknown error"
            throw SHARPError.generationFailed(errorMessage)
        }

        // Decode Base64 PLY data
        guard let plyBase64 = response["ply_data"] as? String,
              let plyData = Data(base64Encoded: plyBase64) else {
            throw SHARPError.invalidResponse
        }

        let plySize = response["ply_size"] as? Int ?? plyData.count
        let gaussianCount = response["gaussian_count"] as? Int ?? 0
        let processingTimeMs = response["processing_time_ms"] as? Double ?? 0.0

        // Parse metadata
        var metadata: GenerationResult.Metadata?
        if let metadataDict = response["metadata"] as? [String: Any] {
            let metadataJson = try JSONSerialization.data(withJSONObject: metadataDict)
            metadata = try JSONDecoder().decode(GenerationResult.Metadata.self, from: metadataJson)
        }

        Self.log.info("PLY generated (direct): \(gaussianCount) Gaussians, \(plySize) bytes in \(processingTimeMs)ms")

        return DirectGenerationResult(
            success: true,
            plyData: plyData,
            plySize: plySize,
            gaussianCount: gaussianCount,
            processingTimeMs: processingTimeMs,
            metadata: metadata,
            error: nil
        )
    }

    /// Get server status
    func getServerStatus() async throws -> [String: Any] {
        return try await sendCommand(["command": "status"])
    }

    /// Start the SHARP server process
    func startServer(pythonPath: String = "python3", scriptPath: String? = nil) async throws {
        Self.log.info("Starting SHARP server...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")

        let serverScript = scriptPath ?? findServerScript()
        guard let serverScript = serverScript else {
            throw SHARPError.connectionFailed("Could not find sharp_server.py")
        }

        process.arguments = [pythonPath, serverScript, "--socket", socketPath]

        // Set up environment
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        // Capture output for debugging
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        do {
            try process.run()
            serverProcess = process

            // Wait for server to start
            for _ in 0..<30 {
                try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds
                if await isServerRunning() {
                    Self.log.info("SHARP server started successfully")
                    return
                }
            }
            throw SHARPError.timeout
        } catch {
            throw SHARPError.connectionFailed("Failed to start server: \(error.localizedDescription)")
        }
    }

    /// Stop the SHARP server
    func stopServer() async {
        if let process = serverProcess, process.isRunning {
            // Try graceful shutdown first
            do {
                _ = try await sendCommand(["command": "shutdown"])
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch {
                // Force terminate if graceful shutdown fails
                process.terminate()
            }
            serverProcess = nil
            Self.log.info("SHARP server stopped")
        }
    }

    // MARK: - Private Methods

    private func findServerScript() -> String? {
        // Look for sharp_server.py relative to the app bundle or current directory
        let possiblePaths = [
            Bundle.main.bundlePath + "/../sharp_server.py",
            Bundle.main.resourcePath.map { $0 + "/sharp_server.py" },
            FileManager.default.currentDirectoryPath + "/sharp_server.py",
            FileManager.default.currentDirectoryPath + "/RealtimeWebcam3DGS/sharp_server.py"
        ].compactMap { $0 }

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return nil
    }

    private func sendCommand(_ command: [String: Any]) async throws -> [String: Any] {
        // Create socket
        let socket = socket(AF_UNIX, SOCK_STREAM, 0)
        guard socket >= 0 else {
            throw SHARPError.connectionFailed("Failed to create socket")
        }
        defer { close(socket) }

        // Connect
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)

        let pathBytes = socketPath.utf8CString
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: Int(104)) { sunPath in
                for (i, byte) in pathBytes.enumerated() {
                    sunPath[i] = byte
                }
            }
        }

        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(socket, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }

        guard connectResult == 0 else {
            throw SHARPError.serverNotRunning
        }

        // Send request
        let jsonData = try JSONSerialization.data(withJSONObject: command)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        _ = jsonString.withCString { ptr in
            send(socket, ptr, strlen(ptr), 0)
        }

        // Shutdown write side to signal end of request
        shutdown(socket, SHUT_WR)

        // Receive response
        var responseData = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)

        while true {
            let bytesRead = recv(socket, &buffer, buffer.count, 0)
            if bytesRead <= 0 { break }
            responseData.append(contentsOf: buffer[0..<bytesRead])
        }

        guard !responseData.isEmpty else {
            throw SHARPError.invalidResponse
        }

        guard let response = try JSONSerialization.jsonObject(with: responseData) as? [String: Any] else {
            throw SHARPError.invalidResponse
        }

        return response
    }
}
