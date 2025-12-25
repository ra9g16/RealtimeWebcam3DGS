import AVFoundation
import Combine
import os
import simd
import Vision

/// Manages face detection for head-tracked parallax view
/// Uses Vision framework to detect face position and estimate depth from face size
@MainActor
class FaceTrackingManager: ObservableObject {
    private static nonisolated let log = Logger(subsystem: "com.metalsplatter.webcam3dgs", category: "FaceTrackingManager")

    // MARK: - Published State

    /// Normalized face position (x, y: -1 to 1, z: estimated distance)
    @Published var facePosition: SIMD3<Float>?
    /// Whether face tracking is currently active
    @Published var isTracking: Bool = false
    /// Whether a face is currently detected
    @Published var faceDetected: Bool = false

    // MARK: - Configuration

    /// Sensitivity multiplier for head movement (higher = more responsive)
    var sensitivity: Float = 1.0
    /// Smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    var smoothing: Float = 0.7
    /// Reference distance for depth calculation (arbitrary units)
    var referenceDistance: Float = 2.0

    // MARK: - Private Properties

    private let processingQueue = DispatchQueue(label: "com.metalsplatter.webcam3dgs.facetracking", qos: .userInteractive)
    private var lastFacePosition: SIMD3<Float> = .zero
    private var frameSkipCounter: Int = 0
    /// Process every Nth frame to reduce CPU load
    private let frameSkipInterval: Int = 2

    /// Reference face width in normalized coordinates (used for depth estimation)
    /// When face appears this size, we consider it at referenceDistance
    private nonisolated let referenceFaceWidth: Float = 0.25

    // MARK: - Public Methods

    /// Start face tracking
    func startTracking() {
        isTracking = true
        Self.log.info("Face tracking started")
    }

    /// Stop face tracking
    func stopTracking() {
        isTracking = false
        faceDetected = false
        facePosition = nil
        Self.log.info("Face tracking stopped")
    }

    /// Process a camera frame for face detection
    /// This should be called from the camera capture callback
    /// - Parameter pixelBuffer: The camera frame to process
    nonisolated func processFrame(_ pixelBuffer: CVPixelBuffer) {
        // Skip frames to reduce CPU load
        let skipCounter = Task { @MainActor in
            self.frameSkipCounter += 1
            return self.frameSkipCounter
        }

        Task {
            let counter = await skipCounter.value
            if counter % frameSkipInterval != 0 {
                return
            }

            await processFrameInternal(pixelBuffer)
        }
    }

    // MARK: - Private Methods

    private func processFrameInternal(_ pixelBuffer: CVPixelBuffer) async {
        guard isTracking else { return }

        // Capture reference distance before entering background queue
        let refDistance = self.referenceDistance
        let refFaceWidth = self.referenceFaceWidth

        // Run face detection on background queue
        let result: SIMD3<Float>? = await withCheckedContinuation { continuation in
            processingQueue.async {
                // Create request in this scope (not stored as instance property)
                let request = VNDetectFaceRectanglesRequest()
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])

                do {
                    try handler.perform([request])

                    guard let results = request.results,
                          let face = results.first else {
                        continuation.resume(returning: nil)
                        return
                    }

                    // Extract face bounding box (normalized 0-1 coordinates)
                    let boundingBox = face.boundingBox

                    // Calculate face center (convert from Vision coordinates to centered coordinates)
                    // Vision uses bottom-left origin, we want center-origin with Y up
                    let centerX = Float(boundingBox.midX) * 2.0 - 1.0  // -1 to 1
                    let centerY = Float(boundingBox.midY) * 2.0 - 1.0  // -1 to 1 (Y up)

                    // Estimate depth from face width
                    // Larger face = closer, smaller face = farther
                    let faceWidth = Float(boundingBox.width)
                    let estimatedZ = refFaceWidth / max(faceWidth, 0.05) * refDistance

                    let position = SIMD3<Float>(centerX, centerY, estimatedZ)
                    continuation.resume(returning: position)

                } catch {
                    Self.log.error("Face detection failed: \(error.localizedDescription)")
                    continuation.resume(returning: nil)
                }
            }
        }

        // Update state on main actor
        if let newPosition = result {
            // Apply smoothing
            let smoothedPosition: SIMD3<Float>
            if lastFacePosition == .zero {
                smoothedPosition = newPosition
            } else {
                smoothedPosition = mix(newPosition, lastFacePosition, t: smoothing)
            }

            lastFacePosition = smoothedPosition
            facePosition = smoothedPosition * sensitivity
            faceDetected = true
        } else {
            faceDetected = false
            // Keep last position briefly to avoid jitter when face is temporarily lost
        }
    }
}

// MARK: - Helper Functions

/// Linear interpolation between two vectors
private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    return a * (1.0 - t) + b * t
}
