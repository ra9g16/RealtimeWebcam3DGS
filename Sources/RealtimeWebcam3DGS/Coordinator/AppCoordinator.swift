import Foundation
import Combine
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import AppKit
import os

/// Simple async semaphore for limiting concurrent operations
actor AsyncSemaphore {
    private var count: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(count: Int) {
        self.count = count
    }

    func wait() async {
        if count > 0 {
            count -= 1
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func signal() {
        if let waiter = waiters.first {
            waiters.removeFirst()
            waiter.resume()
        } else {
            count += 1
        }
    }

    var availableCount: Int {
        count
    }
}

/// Main coordinator for the webcam 3DGS pipeline
@MainActor
class AppCoordinator: ObservableObject {
    private static let log = Logger(subsystem: "com.metalsplatter.webcam3dgs", category: "AppCoordinator")

    // MARK: - App Mode
    enum AppMode: String, CaseIterable {
        case webcam = "Webcam"
        case staticImage = "Static"
    }

    @Published var appMode: AppMode = .staticImage {
        didSet {
            if appMode == .webcam {
                // Switching to webcam mode - clear imported image
                importedImage = nil
                importedImagePath = nil
            } else {
                // Switching to static mode - stop webcam pipeline
                stopPipeline()
            }
        }
    }

    /// The imported image for static mode
    @Published var importedImage: CGImage?
    /// Path to the imported image
    @Published var importedImagePath: URL?

    // MARK: - Components
    let cameraManager: CameraCaptureManager
    let sharpClient: SHARPClient
    let renderManager: SplatRenderManager
    let faceTrackingManager: FaceTrackingManager

    // MARK: - Settings
    @Published var captureIntervalSeconds: Double = 5.0 {
        didSet {
            cameraManager.captureInterval = captureIntervalSeconds
        }
    }
    @Published var autoCapture: Bool = true
    /// Enable dynamic capture interval based on processing completion
    @Published var dynamicCaptureInterval: Bool = true
    /// Minimum interval between captures when using dynamic mode (seconds)
    @Published var minimumCaptureInterval: Double = 0.5
    /// Enable pipeline parallelization (capture during inference)
    @Published var pipelineParallelization: Bool = true
    /// Maximum number of concurrent processing tasks
    @Published var maxConcurrentProcessing: Int = 2
    /// Enable direct socket transfer (avoid PLY file I/O)
    /// This eliminates ~100-200ms of file write/read overhead per frame
    @Published var useDirectSocketTransfer: Bool = true
    /// Automatically start capture and generate 3DGS on app launch
    @Published var autoCaptureOnLaunch: Bool = true
    /// Crop captured images to square (center crop) for optimal SHARP model input
    /// SHARP uses 1536x1536 internally, so square input avoids aspect ratio distortion
    @Published var cropToSquare: Bool = true {
        didSet {
            cameraManager.cropToSquare = cropToSquare
        }
    }
    /// Enable head-tracked parallax view (uses webcam face detection)
    /// Defaults to on - works in both Static and Webcam modes
    @Published var useHeadTracking: Bool = true {
        didSet {
            renderManager.useHeadTracking = useHeadTracking
            if useHeadTracking {
                faceTrackingManager.startTracking()
                // Start camera for face tracking if in static mode (webcam mode already has camera running)
                if appMode == .staticImage && !cameraManager.isCapturing {
                    cameraManager.startCapture(onImageCaptured: nil)
                }
            } else {
                faceTrackingManager.stopTracking()
                renderManager.headPosition = .zero
                // Stop camera if in static mode (only used for face tracking)
                if appMode == .staticImage && cameraManager.isCapturing {
                    cameraManager.stopCapture()
                }
            }
        }
    }
    /// Sensitivity for head tracking parallax effect
    @Published var headTrackingSensitivity: Float = 0.5 {
        didSet {
            renderManager.headTrackingSensitivity = headTrackingSensitivity
        }
    }

    // MARK: - State
    @Published var appState: AppState = .idle
    @Published var processingQueue: [ProcessingItem] = []
    @Published var serverRunning: Bool = false
    @Published var errorMessage: String?
    @Published var secondsUntilNextCapture: Int = 0
    @Published var lastProcessingTimeMs: Double = 0
    @Published var activeProcessingCount: Int = 0
    /// Whether a face is currently detected (mirrors faceTrackingManager.faceDetected)
    @Published var faceDetected: Bool = false

    private var countdownTimer: Timer?
    private var frameCounter: Int = 0
    private var cancellables = Set<AnyCancellable>()
    private var recentProcessingTimes: [Double] = []
    private let maxProcessingTimeSamples = 5
    /// Track the last successfully rendered frame number
    private var lastRenderedFrameNumber: Int = 0
    /// Semaphore to limit concurrent processing
    private var processingSemaphore: AsyncSemaphore?

    // MARK: - File Management
    private let tempDirectory: URL
    private let capturesDirectory: URL
    private let outputsDirectory: URL

    enum AppState: Equatable {
        case idle
        case capturing
        case generating
        case rendering
        case error(String)

        var description: String {
            switch self {
            case .idle: return "Idle"
            case .capturing: return "Capturing"
            case .generating: return "Generating 3DGS..."
            case .rendering: return "Rendering"
            case .error(let msg): return "Error: \(msg)"
            }
        }

        static func == (lhs: AppState, rhs: AppState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.capturing, .capturing), (.generating, .generating), (.rendering, .rendering):
                return true
            case (.error(let a), .error(let b)):
                return a == b
            default:
                return false
            }
        }
    }

    struct ProcessingItem: Identifiable {
        let id = UUID()
        let frameNumber: Int
        let imagePath: URL
        let outputPath: URL
        var status: ProcessingStatus = .pending

        enum ProcessingStatus {
            case pending
            case processing
            case completed
            case failed(String)
        }
    }

    // MARK: - Initialization

    init() {
        // Set up temp directories
        let basePath = FileManager.default.temporaryDirectory.appendingPathComponent("webcam_3dgs")
        tempDirectory = basePath
        capturesDirectory = basePath.appendingPathComponent("captures")
        outputsDirectory = basePath.appendingPathComponent("outputs")

        // Create directories
        try? FileManager.default.createDirectory(at: capturesDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: outputsDirectory, withIntermediateDirectories: true)

        // Initialize components
        cameraManager = CameraCaptureManager()
        sharpClient = SHARPClient()
        renderManager = SplatRenderManager()!
        faceTrackingManager = FaceTrackingManager()

        // Initialize semaphore for concurrent processing limit
        processingSemaphore = AsyncSemaphore(count: maxConcurrentProcessing)

        Self.log.info("AppCoordinator initialized")
        Self.log.info("Temp directory: \(self.tempDirectory.path)")
        Self.log.info("Pipeline parallelization enabled with max \(self.maxConcurrentProcessing) concurrent tasks")

        // Set up face tracking connection
        setupFaceTracking()
    }

    private func setupFaceTracking() {
        // Connect face tracking to camera frames
        cameraManager.onFrameForFaceTracking = { [weak self] pixelBuffer in
            self?.faceTrackingManager.processFrame(pixelBuffer)
        }

        // Bind face position to renderer
        faceTrackingManager.$facePosition
            .receive(on: DispatchQueue.main)
            .sink { [weak self] position in
                if let pos = position {
                    self?.renderManager.headPosition = pos
                }
            }
            .store(in: &cancellables)

        // Bind face detected state to coordinator (for UI observation)
        faceTrackingManager.$faceDetected
            .receive(on: DispatchQueue.main)
            .assign(to: &$faceDetected)

        // Start tracking if enabled by default
        if useHeadTracking {
            faceTrackingManager.startTracking()
            renderManager.useHeadTracking = true
            // Start camera for face tracking (static mode starts with camera off)
            if !cameraManager.isCapturing {
                cameraManager.startCapture(onImageCaptured: nil)
            }
        }
    }

    // MARK: - Lifecycle

    func initialize() async {
        Self.log.info("Initializing app...")

        // Check if server is already running
        serverRunning = await sharpClient.isServerRunning()

        if !serverRunning {
            Self.log.info("SHARP server not running. Will need to start manually or use the control panel.")
        }

        // Auto-start pipeline and capture on launch if enabled (only in webcam mode)
        if autoCaptureOnLaunch && appMode == .webcam {
            Self.log.info("Auto-capture on launch enabled, starting pipeline...")
            await startPipeline()

            // Wait briefly for camera to initialize, then trigger first capture
            if appState == .capturing {
                try? await Task.sleep(for: .milliseconds(500))
                Self.log.info("Triggering initial capture...")
                cameraManager.triggerCapture()
            }
        }
    }

    func cleanup() async {
        Self.log.info("Cleaning up...")
        stopPipeline()
        await stopServer()

        // Clean up temp files
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    // MARK: - Pipeline Control

    func startPipeline() async {
        guard appState == .idle else { return }

        Self.log.info("Starting pipeline...")

        // Ensure server is running
        if !serverRunning {
            await startServer()
            guard serverRunning else {
                appState = .error("Failed to start SHARP server")
                return
            }
        }

        // Start camera capture
        cameraManager.startCapture { [weak self] image in
            guard let self = self else { return }
            Task { @MainActor in
                await self.onImageCaptured(image)
            }
        }

        appState = .capturing
        startCountdownTimer()

        Self.log.info("Pipeline started")
    }

    func stopPipeline() {
        Self.log.info("Stopping pipeline...")

        countdownTimer?.invalidate()
        countdownTimer = nil
        cameraManager.stopCapture()
        appState = .idle
        secondsUntilNextCapture = 0

        Self.log.info("Pipeline stopped")
    }

    func captureAndGenerate() async {
        guard cameraManager.isCapturing else { return }

        cameraManager.triggerCapture()
    }

    // MARK: - Server Control

    func startServer() async {
        Self.log.info("Starting SHARP server...")

        do {
            try await sharpClient.startServer()
            serverRunning = true
            errorMessage = nil
            Self.log.info("SHARP server started successfully")
        } catch {
            Self.log.error("Failed to start SHARP server: \(error.localizedDescription)")
            errorMessage = "Failed to start server: \(error.localizedDescription)"
            serverRunning = false
        }
    }

    func stopServer() async {
        Self.log.info("Stopping SHARP server...")
        await sharpClient.stopServer()
        serverRunning = false
    }

    // MARK: - Private Methods

    private func onImageCaptured(_ image: CGImage) async {
        frameCounter += 1
        let frameNumber = frameCounter

        Self.log.info("Image captured: frame \(frameNumber)")

        // Save image to file
        let imagePath = capturesDirectory.appendingPathComponent("frame_\(String(format: "%04d", frameNumber)).jpg")
        let outputPath = outputsDirectory.appendingPathComponent("frame_\(String(format: "%04d", frameNumber)).ply")

        guard saveImage(image, to: imagePath) else {
            Self.log.error("Failed to save captured image")
            return
        }

        // Add to processing queue
        let item = ProcessingItem(frameNumber: frameNumber, imagePath: imagePath, outputPath: outputPath)
        processingQueue.append(item)

        // Process in background with pipeline parallelization
        let itemCopy = item
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            await self.processQueueItemParallel(itemCopy)
        }

        // Pipeline parallelization: trigger next capture immediately if enabled
        if pipelineParallelization && autoCapture {
            // Schedule next capture without waiting for processing to complete
            Task {
                try? await Task.sleep(for: .seconds(minimumCaptureInterval))
                await MainActor.run {
                    if self.cameraManager.isCapturing && self.autoCapture {
                        Self.log.info("Pipeline: triggering next capture while processing frame \(frameNumber)")
                        self.cameraManager.triggerCapture()
                    }
                }
            }
        }

        // Reset countdown
        resetCountdown()
    }

    /// Process a queue item with pipeline parallelization support
    private func processQueueItemParallel(_ item: ProcessingItem) async {
        let imagePath = item.imagePath
        let outputPath = item.outputPath
        let frameNumber = item.frameNumber
        let startTime = CFAbsoluteTimeGetCurrent()

        // Wait for semaphore to limit concurrent processing
        if let semaphore = processingSemaphore {
            await semaphore.wait()
        }

        await MainActor.run {
            activeProcessingCount += 1
        }

        defer {
            Task {
                if let semaphore = self.processingSemaphore {
                    await semaphore.signal()
                }
                await MainActor.run {
                    self.activeProcessingCount -= 1
                }
            }
        }

        // Verify input file exists before processing
        guard FileManager.default.fileExists(atPath: imagePath.path) else {
            Self.log.error("Input image does not exist: \(imagePath.path)")
            await MainActor.run {
                var updatedItem = item
                updatedItem.status = .failed("Input image not found")
                self.updateQueueItem(updatedItem)
            }
            return
        }

        await MainActor.run {
            var updatedItem = item
            updatedItem.status = .processing
            self.updateQueueItem(updatedItem)
            self.appState = .generating
        }

        do {
            // Check which transfer mode to use
            let useDirectTransfer = await MainActor.run { self.useDirectSocketTransfer }

            if useDirectTransfer {
                // Direct socket transfer mode - PLY data returned inline, no file I/O
                let result = try await sharpClient.generatePLYDirect(from: imagePath)

                if result.success, let plyData = result.plyData {
                    await MainActor.run {
                        var updatedItem = item
                        updatedItem.status = .completed
                        self.updateQueueItem(updatedItem)
                    }

                    // Only render if this is a newer frame than the last rendered one
                    let shouldRender = await MainActor.run { () -> Bool in
                        if frameNumber > self.lastRenderedFrameNumber {
                            self.lastRenderedFrameNumber = frameNumber
                            return true
                        }
                        Self.log.info("Skipping render for frame \(frameNumber) (newer frame \(self.lastRenderedFrameNumber) already rendered)")
                        return false
                    }

                    if shouldRender {
                        await MainActor.run {
                            self.appState = .rendering
                        }

                        // Load PLY directly from memory data (no file I/O)
                        try await renderManager.loadPLY(from: plyData)

                        // Track processing time for dynamic interval calculation
                        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

                        await MainActor.run {
                            self.lastProcessingTimeMs = processingTime
                            self.updateProcessingTimeHistory(processingTime)
                        }

                        Self.log.info("Successfully rendered PLY (direct) for frame \(frameNumber) with \(result.gaussianCount) Gaussians in \(String(format: "%.0f", processingTime))ms")
                    }
                } else {
                    throw SHARPClient.SHARPError.generationFailed(result.error ?? "Unknown error")
                }
            } else {
                // File-based transfer mode - PLY written to file then read
                let result = try await sharpClient.generatePLY(from: imagePath, outputPath: outputPath)

                if result.success, let plyPath = result.plyPath {
                    await MainActor.run {
                        var updatedItem = item
                        updatedItem.status = .completed
                        self.updateQueueItem(updatedItem)
                    }

                    // Only render if this is a newer frame than the last rendered one
                    let shouldRender = await MainActor.run { () -> Bool in
                        if frameNumber > self.lastRenderedFrameNumber {
                            self.lastRenderedFrameNumber = frameNumber
                            return true
                        }
                        Self.log.info("Skipping render for frame \(frameNumber) (newer frame \(self.lastRenderedFrameNumber) already rendered)")
                        return false
                    }

                    if shouldRender {
                        await MainActor.run {
                            self.appState = .rendering
                        }

                        try await renderManager.loadPLY(from: URL(fileURLWithPath: plyPath))

                        // Track processing time for dynamic interval calculation
                        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

                        await MainActor.run {
                            self.lastProcessingTimeMs = processingTime
                            self.updateProcessingTimeHistory(processingTime)
                        }

                        Self.log.info("Successfully rendered PLY for frame \(frameNumber) with \(result.gaussianCount) Gaussians in \(String(format: "%.0f", processingTime))ms")
                    }
                } else {
                    throw SHARPClient.SHARPError.generationFailed(result.error ?? "Unknown error")
                }
            }
        } catch {
            Self.log.error("Failed to process frame \(frameNumber): \(error.localizedDescription)")
            await MainActor.run {
                var updatedItem = item
                updatedItem.status = .failed(error.localizedDescription)
                self.updateQueueItem(updatedItem)
                self.errorMessage = error.localizedDescription
            }
        }

        // Return to capturing state if still running
        await MainActor.run {
            if self.cameraManager.isCapturing {
                self.appState = .capturing
            }

            // Clean up old queue items
            self.cleanupQueue()
        }

        // Dynamic capture: trigger next capture after processing completes (if not using pipeline parallelization)
        if !pipelineParallelization && dynamicCaptureInterval && autoCapture {
            await triggerNextCaptureAfterDelay()
        }
    }

    /// Trigger next capture after minimum interval
    private func triggerNextCaptureAfterDelay() async {
        do {
            // Wait for minimum interval to prevent overwhelming the system
            try await Task.sleep(for: .seconds(minimumCaptureInterval))

            // Only trigger if still in capturing state
            if cameraManager.isCapturing && autoCapture {
                Self.log.info("Dynamic capture: triggering next capture")
                cameraManager.triggerCapture()
                resetCountdown()
            }
        } catch {
            // Task cancelled, ignore
        }
    }

    /// Update processing time history for adaptive interval calculation
    private func updateProcessingTimeHistory(_ time: Double) {
        recentProcessingTimes.append(time)
        if recentProcessingTimes.count > maxProcessingTimeSamples {
            recentProcessingTimes.removeFirst()
        }
    }

    /// Calculate adaptive capture interval based on recent processing times
    func getAdaptiveCaptureInterval() -> Double {
        guard !recentProcessingTimes.isEmpty else {
            return captureIntervalSeconds
        }
        let avgTime = recentProcessingTimes.reduce(0, +) / Double(recentProcessingTimes.count)
        // Use 1.2x average processing time as interval, with minimum bound
        return max(minimumCaptureInterval, avgTime / 1000.0 * 1.2)
    }

    private func updateQueueItem(_ item: ProcessingItem) {
        if let index = processingQueue.firstIndex(where: { $0.id == item.id }) {
            processingQueue[index] = item
        }
    }

    private func cleanupQueue() {
        // Keep only the last 10 items, but only remove completed or failed items
        let completedOrFailed = processingQueue.filter { item in
            switch item.status {
            case .completed, .failed:
                return true
            case .pending, .processing:
                return false
            }
        }

        if completedOrFailed.count > 10 {
            let itemsToRemove = completedOrFailed.prefix(completedOrFailed.count - 10)
            for item in itemsToRemove {
                // Delete temp files
                try? FileManager.default.removeItem(at: item.imagePath)
                try? FileManager.default.removeItem(at: item.outputPath)
                // Remove from queue
                processingQueue.removeAll { $0.id == item.id }
            }
        }
    }

    private func saveImage(_ image: CGImage, to url: URL) -> Bool {
        // Ensure directory exists
        let directory = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            Self.log.error("Failed to create image destination for \(url.path)")
            return false
        }

        CGImageDestinationAddImage(destination, image, [
            kCGImageDestinationLossyCompressionQuality: 0.9
        ] as CFDictionary)

        let success = CGImageDestinationFinalize(destination)

        // Verify file was actually written
        if success {
            let exists = FileManager.default.fileExists(atPath: url.path)
            if !exists {
                Self.log.error("Image save reported success but file does not exist: \(url.path)")
                return false
            }
            Self.log.info("Image saved successfully: \(url.path)")
        }

        return success
    }

    private func startCountdownTimer() {
        secondsUntilNextCapture = Int(captureIntervalSeconds)

        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self else { return }
                if self.secondsUntilNextCapture > 0 {
                    self.secondsUntilNextCapture -= 1
                }
            }
        }
    }

    private func resetCountdown() {
        secondsUntilNextCapture = Int(captureIntervalSeconds)
    }

    // MARK: - Static Image Mode

    /// Import an image file and convert to 3DGS
    func importImage() async {
        Self.log.info("Opening image import dialog...")

        // Show file picker (must run on main thread)
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.jpeg, .png, .heic, .tiff, .bmp, .gif]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.message = "Select an image to convert to 3D Gaussian Splatting"
        panel.prompt = "Import"

        guard panel.runModal() == .OK, let url = panel.url else {
            Self.log.info("Image import cancelled")
            return
        }

        await processImportedImage(from: url)
    }

    /// Process an imported image file
    func processImportedImage(from url: URL) async {
        Self.log.info("Importing image from: \(url.path)")

        // Load and display preview
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            errorMessage = "Failed to load image: \(url.lastPathComponent)"
            Self.log.error("Failed to load image from \(url.path)")
            return
        }

        importedImagePath = url
        importedImage = cgImage
        errorMessage = nil

        Self.log.info("Image loaded: \(cgImage.width)x\(cgImage.height)")

        // Ensure server is running
        if !serverRunning {
            Self.log.info("Starting SHARP server for image processing...")
            await startServer()
            guard serverRunning else {
                errorMessage = "Failed to start SHARP server"
                appState = .error("Failed to start SHARP server")
                return
            }
        }

        // Copy to temp location as JPEG (SHARP expects file path)
        let tempPath = tempDirectory.appendingPathComponent("imported_\(UUID().uuidString).jpg")
        guard saveImage(cgImage, to: tempPath) else {
            errorMessage = "Failed to save image for processing"
            Self.log.error("Failed to save imported image to temp location")
            return
        }

        // Process via SHARP
        appState = .generating
        Self.log.info("Processing image through SHARP...")

        do {
            let result = try await sharpClient.generatePLYDirect(from: tempPath)

            if result.success, let plyData = result.plyData {
                Self.log.info("SHARP processing complete: \(result.gaussianCount) Gaussians in \(String(format: "%.0f", result.processingTimeMs))ms")

                // Load into renderer
                try await renderManager.loadPLY(from: plyData)

                appState = .rendering
                lastProcessingTimeMs = result.processingTimeMs

                Self.log.info("Static image 3DGS rendering ready")
            } else {
                let errorMsg = result.error ?? "Unknown error"
                errorMessage = "Processing failed: \(errorMsg)"
                appState = .error(errorMsg)
                Self.log.error("SHARP processing failed: \(errorMsg)")
            }
        } catch {
            errorMessage = "Processing failed: \(error.localizedDescription)"
            appState = .error(error.localizedDescription)
            Self.log.error("Failed to process imported image: \(error.localizedDescription)")
        }

        // Clean up temp file
        try? FileManager.default.removeItem(at: tempPath)
    }
}
