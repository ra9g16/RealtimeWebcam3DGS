@preconcurrency import AVFoundation
import Combine
import CoreGraphics
import CoreImage
import os

/// Manages webcam capture and periodic image extraction
@MainActor
class CameraCaptureManager: NSObject, ObservableObject {
    private static let log = Logger(subsystem: "com.metalsplatter.webcam3dgs", category: "CameraCaptureManager")

    // MARK: - Configuration
    var captureInterval: TimeInterval = 5.0
    var captureResolution: CGSize = CGSize(width: 1280, height: 720)
    /// Crop captured images to square (center crop). This optimizes for SHARP's 1536x1536 input.
    var cropToSquare: Bool = true

    // MARK: - Published State
    @Published var isCapturing: Bool = false
    @Published var previewImage: CGImage?
    @Published var lastCapturedImage: CGImage?
    @Published var availableCameras: [AVCaptureDevice] = []
    @Published var selectedCamera: AVCaptureDevice?
    @Published var errorMessage: String?

    // MARK: - Private Properties
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var captureTimer: Timer?
    private let captureQueue = DispatchQueue(label: "com.metalsplatter.webcam3dgs.capture", qos: .userInitiated)
    private nonisolated let ciContext = CIContext()

    // Use a thread-safe storage for the last frame
    private let frameStorage = FrameStorage()
    private var onImageCaptured: ((CGImage) -> Void)?

    /// Callback for continuous frame processing (e.g., face tracking)
    /// This is called for every frame, not just periodic captures
    var onFrameForFaceTracking: ((CVPixelBuffer) -> Void)?

    /// Thread-safe storage for camera frames
    private final class FrameStorage: @unchecked Sendable {
        private let lock = NSLock()
        private var _frame: CMSampleBuffer?

        var frame: CMSampleBuffer? {
            get {
                lock.lock()
                defer { lock.unlock() }
                return _frame
            }
            set {
                lock.lock()
                defer { lock.unlock() }
                _frame = newValue
            }
        }
    }

    // MARK: - Initialization
    override init() {
        super.init()
        refreshAvailableCameras()
    }

    // MARK: - Public Methods

    /// Refresh the list of available cameras
    func refreshAvailableCameras() {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        availableCameras = discoverySession.devices
        if selectedCamera == nil, let firstCamera = availableCameras.first {
            selectedCamera = firstCamera
        }
        Self.log.info("Found \(self.availableCameras.count) cameras")
    }

    /// Select a specific camera by its unique ID
    func selectCamera(_ deviceID: String) {
        guard let device = availableCameras.first(where: { $0.uniqueID == deviceID }) else {
            Self.log.warning("Camera with ID \(deviceID) not found")
            return
        }
        selectedCamera = device
        if isCapturing {
            Task {
                stopCapture()
                try? await Task.sleep(nanoseconds: 100_000_000)
                startCapture(onImageCaptured: onImageCaptured)
            }
        }
    }

    /// Start capturing from the selected camera
    func startCapture(onImageCaptured: ((CGImage) -> Void)? = nil) {
        guard !isCapturing else { return }
        guard let camera = selectedCamera else {
            errorMessage = "No camera selected"
            return
        }

        self.onImageCaptured = onImageCaptured

        let session = AVCaptureSession()
        // Use 720p instead of 1080p - SHARP resizes to 1536x1536 anyway
        // This reduces capture overhead and data transfer size
        session.sessionPreset = .hd1280x720

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            guard session.canAddInput(input) else {
                errorMessage = "Cannot add camera input"
                return
            }
            session.addInput(input)
        } catch {
            errorMessage = "Failed to create camera input: \(error.localizedDescription)"
            Self.log.error("Failed to create camera input: \(error.localizedDescription)")
            return
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: captureQueue)
        output.alwaysDiscardsLateVideoFrames = true

        guard session.canAddOutput(output) else {
            errorMessage = "Cannot add video output"
            return
        }
        session.addOutput(output)

        captureSession = session
        videoOutput = output

        // Start session on capture queue
        let capturedSession = session
        captureQueue.async {
            capturedSession.startRunning()
        }

        isCapturing = true
        errorMessage = nil
        Self.log.info("Started capture from \(camera.localizedName)")

        // Start periodic capture timer
        startCaptureTimer()
    }

    /// Stop capturing
    func stopCapture() {
        guard isCapturing else { return }

        captureTimer?.invalidate()
        captureTimer = nil

        if let session = captureSession {
            captureQueue.async {
                session.stopRunning()
            }
        }

        captureSession = nil
        videoOutput = nil
        isCapturing = false
        onImageCaptured = nil

        Self.log.info("Stopped capture")
    }

    /// Capture an image immediately
    func captureNow() -> CGImage? {
        guard let sampleBuffer = frameStorage.frame else {
            Self.log.warning("No frame available for capture")
            return nil
        }
        return extractImage(from: sampleBuffer, cropToSquare: cropToSquare)
    }

    /// Trigger an immediate capture and callback
    func triggerCapture() {
        guard let image = captureNow() else { return }
        lastCapturedImage = image
        onImageCaptured?(image)
    }

    // MARK: - Private Methods

    private func startCaptureTimer() {
        captureTimer?.invalidate()
        captureTimer = Timer.scheduledTimer(withTimeInterval: captureInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.triggerCapture()
            }
        }
    }

    private nonisolated func extractImage(from sampleBuffer: CMSampleBuffer, cropToSquare: Bool) -> CGImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let rect: CGRect
        if cropToSquare && width != height {
            // Center crop to square
            let size = min(width, height)
            let x = (width - size) / 2
            let y = (height - size) / 2
            rect = CGRect(x: x, y: y, width: size, height: size)
        } else {
            rect = CGRect(x: 0, y: 0, width: width, height: height)
        }

        return ciContext.createCGImage(ciImage, from: rect)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraCaptureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Store the last frame for on-demand capture
        frameStorage.frame = sampleBuffer

        // Get pixel buffer for face tracking
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            // Call face tracking callback (runs on capture queue, face tracker handles threading)
            Task { @MainActor in
                self.onFrameForFaceTracking?(pixelBuffer)
            }
        }

        // Update preview on main thread (throttled)
        // Preview is not cropped to show the full camera view
        if let image = extractImage(from: sampleBuffer, cropToSquare: false) {
            Task { @MainActor in
                self.previewImage = image
            }
        }
    }
}
