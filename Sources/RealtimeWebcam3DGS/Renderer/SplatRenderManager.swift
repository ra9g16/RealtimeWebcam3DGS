import Foundation
import Metal
import MetalKit
import MetalSplatter
import simd
import os

/// Manages 3D Gaussian Splat rendering using MetalSplatter
@MainActor
class SplatRenderManager: NSObject, ObservableObject, MTKViewDelegate {
    private static let log = Logger(subsystem: "com.metalsplatter.webcam3dgs", category: "SplatRenderManager")

    // MARK: - Constants
    private static let maxSimultaneousRenders = 3

    // MARK: - Metal Resources
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private var renderer: SplatRenderer?
    private var rendererInitialized = false

    // MARK: - Double Buffering
    /// Pre-loaded renderer ready for instant swap
    private var nextRenderer: SplatRenderer?
    /// URL of the pre-loaded PLY
    private var nextPLYPath: URL?
    /// Splat count for the pre-loaded renderer
    private var nextSplatCount: Int = 0
    /// Whether a preload is in progress
    @Published var isPreloading: Bool = false

    // MARK: - State
    @Published var currentPLYPath: URL?
    @Published var splatCount: Int = 0
    @Published var fps: Double = 0.0
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    // MARK: - Camera Control
    var cameraDistance: Float = 1.2  // Closer for larger initial display
    var cameraRotationX: Float = 0.0
    var cameraRotationY: Float = Float.pi  // Start viewing from front (180 degrees)
    var fieldOfView: Float = 60.0

    // MARK: - Head Tracking (Rotation-based Parallax)
    /// Head position from face tracking (x, y: -1 to 1, z: estimated distance)
    var headPosition: SIMD3<Float> = .zero
    /// Enable head-tracked parallax view
    var useHeadTracking: Bool = false
    /// Sensitivity multiplier for head tracking rotation (radians per unit head movement)
    var headTrackingSensitivity: Float = 0.3
    /// Maximum rotation angle for head tracking (radians)
    private let maxHeadTrackingRotation: Float = Float.pi / 6  // 30 degrees

    // MARK: - Rendering State
    private let inFlightSemaphore = DispatchSemaphore(value: maxSimultaneousRenders)
    private var drawableSize: CGSize = .zero
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0

    // MARK: - View Configuration
    private var colorFormat: MTLPixelFormat = .bgra8Unorm_srgb
    private var depthFormat: MTLPixelFormat = .depth32Float

    // MARK: - Initialization

    init?(device: MTLDevice? = nil) {
        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            Self.log.error("Failed to create Metal device")
            return nil
        }
        self.device = metalDevice

        guard let queue = metalDevice.makeCommandQueue() else {
            Self.log.error("Failed to create command queue")
            return nil
        }
        self.commandQueue = queue

        super.init()

        // Initialize renderer early so Metal library is loaded at startup
        initializeRenderer()
    }

    private func initializeRenderer() {
        guard !rendererInitialized else { return }

        do {
            renderer = try SplatRenderer(
                device: device,
                colorFormat: colorFormat,
                depthFormat: depthFormat,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: Self.maxSimultaneousRenders
            )
            rendererInitialized = true
            Self.log.info("SplatRenderer initialized successfully")
        } catch {
            Self.log.error("Failed to initialize SplatRenderer: \(error.localizedDescription)")
            errorMessage = "Failed to initialize renderer: \(error.localizedDescription)"
        }
    }

    // MARK: - Public Methods

    /// Configure the renderer for a specific MTKView
    func configure(for view: MTKView) async throws {
        view.device = device
        view.colorPixelFormat = colorFormat
        view.depthStencilPixelFormat = depthFormat
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        view.delegate = self

        colorFormat = view.colorPixelFormat
        depthFormat = view.depthStencilPixelFormat

        Self.log.info("Renderer configured for view")
    }

    /// Load a PLY file for rendering
    /// - Parameters:
    ///   - url: URL of the PLY file
    ///   - showLoadingState: Whether to show loading indicator (set to false for streaming updates)
    func loadPLY(from url: URL, showLoadingState: Bool = false) async throws {
        Self.log.info("Loading PLY from \(url.path)")
        if showLoadingState {
            isLoading = true
        }
        errorMessage = nil

        do {
            // Create a new renderer for each PLY to avoid buffer accumulation
            // SplatRenderer.read() appends to existing buffers, so we need a fresh instance
            let newRenderer = try SplatRenderer(
                device: device,
                colorFormat: colorFormat,
                depthFormat: depthFormat,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: Self.maxSimultaneousRenders
            )

            // Use memory-mapped I/O for faster file reading
            // This reduces disk read overhead by using virtual memory mapping
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            let fileExtension = url.pathExtension

            try await newRenderer.read(from: data, fileExtension: fileExtension)

            // Get splat count from renderer
            let count = newRenderer.splatCount

            // Replace renderer atomically
            renderer = newRenderer
            rendererInitialized = true
            currentPLYPath = url
            splatCount = count
            isLoading = false

            Self.log.info("PLY loaded successfully: \(count) splats")
        } catch {
            Self.log.error("Failed to load PLY: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
            isLoading = false
            throw error
        }
    }

    /// Load PLY directly from memory data (no file I/O)
    /// This is used for direct socket transfer, avoiding disk I/O overhead
    /// - Parameters:
    ///   - data: PLY file data
    ///   - fileExtension: File extension for format detection (default: "ply")
    ///   - showLoadingState: Whether to show loading indicator (set to false for streaming updates)
    func loadPLY(from data: Data, fileExtension: String = "ply", showLoadingState: Bool = false) async throws {
        Self.log.info("Loading PLY from memory data (\(data.count) bytes)")
        if showLoadingState {
            isLoading = true
        }
        errorMessage = nil

        do {
            // Create a new renderer for each PLY to avoid buffer accumulation
            let newRenderer = try SplatRenderer(
                device: device,
                colorFormat: colorFormat,
                depthFormat: depthFormat,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: Self.maxSimultaneousRenders
            )

            try await newRenderer.read(from: data, fileExtension: fileExtension)

            // Get splat count from renderer
            let count = newRenderer.splatCount

            // Replace renderer atomically
            renderer = newRenderer
            rendererInitialized = true
            currentPLYPath = nil  // No file path for in-memory data
            splatCount = count
            isLoading = false

            Self.log.info("PLY loaded from memory successfully: \(count) splats")
        } catch {
            Self.log.error("Failed to load PLY from memory: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
            isLoading = false
            throw error
        }
    }

    /// Clear the current scene
    func clearScene() {
        // Don't destroy the renderer, just clear its data if possible
        currentPLYPath = nil
        splatCount = 0
    }

    // MARK: - Double Buffering Methods

    /// Preload a PLY file in the background for instant swap later
    /// This enables seamless transitions between PLY files
    func preloadPLY(from url: URL) async throws {
        Self.log.info("Preloading PLY from \(url.path)")
        isPreloading = true

        do {
            // Create a new renderer for preloading
            let preloadRenderer = try SplatRenderer(
                device: device,
                colorFormat: colorFormat,
                depthFormat: depthFormat,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: Self.maxSimultaneousRenders
            )

            // Use memory-mapped I/O for faster file reading
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            let fileExtension = url.pathExtension

            try await preloadRenderer.read(from: data, fileExtension: fileExtension)

            // Get splat count from renderer
            let count = preloadRenderer.splatCount

            // Store for later swap
            nextRenderer = preloadRenderer
            nextPLYPath = url
            nextSplatCount = count
            isPreloading = false

            Self.log.info("PLY preloaded successfully: \(count) splats, ready for swap")
        } catch {
            Self.log.error("Failed to preload PLY: \(error.localizedDescription)")
            isPreloading = false
            throw error
        }
    }

    /// Preload PLY from memory data in the background for instant swap later
    /// This is used for direct socket transfer with double buffering
    func preloadPLY(from data: Data, fileExtension: String = "ply") async throws {
        Self.log.info("Preloading PLY from memory data (\(data.count) bytes)")
        isPreloading = true

        do {
            // Create a new renderer for preloading
            let preloadRenderer = try SplatRenderer(
                device: device,
                colorFormat: colorFormat,
                depthFormat: depthFormat,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: Self.maxSimultaneousRenders
            )

            try await preloadRenderer.read(from: data, fileExtension: fileExtension)

            // Get splat count from renderer
            let count = preloadRenderer.splatCount

            // Store for later swap
            nextRenderer = preloadRenderer
            nextPLYPath = nil  // No file path for in-memory data
            nextSplatCount = count
            isPreloading = false

            Self.log.info("PLY preloaded from memory successfully: \(count) splats, ready for swap")
        } catch {
            Self.log.error("Failed to preload PLY from memory: \(error.localizedDescription)")
            isPreloading = false
            throw error
        }
    }

    /// Swap to the preloaded renderer instantly (near-zero latency)
    /// Returns true if swap was successful, false if no preloaded renderer available
    @discardableResult
    func swapToPreloadedRenderer() -> Bool {
        guard let preloaded = nextRenderer else {
            Self.log.warning("No preloaded renderer available for swap")
            return false
        }

        // Instant swap
        renderer = preloaded
        rendererInitialized = true
        currentPLYPath = nextPLYPath
        splatCount = nextSplatCount

        // Clear preload state
        nextRenderer = nil
        nextPLYPath = nil
        nextSplatCount = 0

        Self.log.info("Swapped to preloaded renderer: \(self.splatCount) splats")
        return true
    }

    /// Check if a preloaded renderer is available
    var hasPreloadedRenderer: Bool {
        nextRenderer != nil
    }

    /// Load PLY with automatic double-buffering
    /// If a preloaded renderer matches the URL, swap instantly; otherwise load normally
    func loadPLYWithDoubleBuffering(from url: URL) async throws {
        // Check if we have a matching preloaded renderer
        if let preloadedPath = nextPLYPath, preloadedPath == url, nextRenderer != nil {
            Self.log.info("Using preloaded renderer for \(url.lastPathComponent)")
            swapToPreloadedRenderer()
            return
        }

        // Fall back to normal loading
        try await loadPLY(from: url)
    }

    // MARK: - Camera Control

    /// Pan the camera
    func pan(deltaX: Float, deltaY: Float) {
        cameraRotationY += deltaX * 0.01
        cameraRotationX += deltaY * 0.01
        cameraRotationX = max(-Float.pi / 2, min(Float.pi / 2, cameraRotationX))
    }

    /// Zoom the camera
    func zoom(delta: Float) {
        cameraDistance *= (1.0 - delta * 0.1)
        cameraDistance = max(0.5, min(10.0, cameraDistance))
    }

    /// Reset camera to default position
    func resetCamera() {
        cameraDistance = 2.0
        cameraRotationX = 0.0
        cameraRotationY = 0.0
    }

    // MARK: - MTKViewDelegate

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        Task { @MainActor in
            drawableSize = size
        }
    }

    nonisolated func draw(in view: MTKView) {
        Task { @MainActor in
            drawFrame(in: view)
        }
    }

    // MARK: - Private Methods

    private func drawFrame(in view: MTKView) {
        guard let renderer = renderer else { return }
        guard splatCount > 0 else { return }  // Don't render if no splats loaded
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: .distantFuture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        // Calculate viewport
        let viewport = calculateViewport()

        // Render
        do {
            try renderer.render(
                viewports: [viewport],
                colorTexture: view.multisampleColorTexture ?? drawable.texture,
                colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
        } catch {
            Self.log.error("Render error: \(error.localizedDescription)")
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        // Update FPS
        updateFPS()
    }

    private func calculateViewport() -> SplatRenderer.ViewportDescriptor {
        let aspectRatio = Float(drawableSize.width / drawableSize.height)
        let fovyRadians = fieldOfView * Float.pi / 180.0
        let nearZ: Float = 0.1
        let farZ: Float = 100.0

        // Standard perspective projection (no off-axis needed for rotation-based tracking)
        let projectionMatrix = matrix_perspective_right_hand(
            fovyRadians: fovyRadians,
            aspectRatio: aspectRatio,
            nearZ: nearZ,
            farZ: farZ
        )

        // Calculate head tracking rotation offsets
        var headRotationX: Float = 0.0
        var headRotationY: Float = 0.0

        if useHeadTracking && headPosition != .zero {
            // Convert head position to rotation angles
            // Head moves left -> camera rotates right (positive Y rotation)
            // Head moves up -> camera rotates down (negative X rotation)
            // Apply sensitivity and clamp to max rotation
            headRotationY = clamp(
                -headPosition.x * headTrackingSensitivity,
                min: -maxHeadTrackingRotation,
                max: maxHeadTrackingRotation
            )
            headRotationX = clamp(
                -headPosition.y * headTrackingSensitivity,
                min: -maxHeadTrackingRotation,
                max: maxHeadTrackingRotation
            )
        }

        // View matrix (camera orbiting around origin)
        // Combine base rotation with head tracking rotation
        let totalRotationX = cameraRotationX + headRotationX
        let totalRotationY = cameraRotationY + headRotationY

        let rotationX = simd_float4x4(rotationAbout: SIMD3<Float>(1, 0, 0), by: totalRotationX)
        let rotationY = simd_float4x4(rotationAbout: SIMD3<Float>(0, 1, 0), by: totalRotationY)
        let translation = simd_float4x4(translation: SIMD3<Float>(0, 0, -cameraDistance))

        // Flip to match common 3DGS coordinate conventions
        let flipZ = simd_float4x4(rotationAbout: SIMD3<Float>(0, 0, 1), by: Float.pi)

        let viewMatrix = translation * rotationX * rotationY * flipZ

        let viewport = MTLViewport(
            originX: 0,
            originY: 0,
            width: drawableSize.width,
            height: drawableSize.height,
            znear: 0,
            zfar: 1
        )

        return SplatRenderer.ViewportDescriptor(
            viewport: viewport,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            screenSize: SIMD2<Int>(Int(drawableSize.width), Int(drawableSize.height))
        )
    }

    /// Clamp a value between min and max
    private func clamp(_ value: Float, min minValue: Float, max maxValue: Float) -> Float {
        return Swift.min(Swift.max(value, minValue), maxValue)
    }

    private func updateFPS() {
        let currentTime = CACurrentMediaTime()
        frameCount += 1

        if currentTime - lastFrameTime >= 1.0 {
            fps = Double(frameCount) / (currentTime - lastFrameTime)
            frameCount = 0
            lastFrameTime = currentTime
        }
    }
}

// MARK: - Matrix Utilities

private func matrix_perspective_right_hand(
    fovyRadians: Float,
    aspectRatio: Float,
    nearZ: Float,
    farZ: Float
) -> simd_float4x4 {
    let ys = 1 / tanf(fovyRadians * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)

    return simd_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * nearZ, 0)
    ))
}

private extension simd_float4x4 {
    init(translation t: SIMD3<Float>) {
        self.init(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        ))
    }

    init(rotationAbout axis: SIMD3<Float>, by angle: Float) {
        let c = cosf(angle)
        let s = sinf(angle)
        let t = 1 - c

        let x = axis.x, y = axis.y, z = axis.z

        self.init(columns: (
            SIMD4<Float>(t*x*x + c,   t*x*y + z*s, t*x*z - y*s, 0),
            SIMD4<Float>(t*x*y - z*s, t*y*y + c,   t*y*z + x*s, 0),
            SIMD4<Float>(t*x*z + y*s, t*y*z - x*s, t*z*z + c,   0),
            SIMD4<Float>(0,           0,           0,           1)
        ))
    }
}
