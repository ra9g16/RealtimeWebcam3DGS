import SwiftUI
import MetalKit

/// SwiftUI wrapper for the MetalKit-based splat renderer
struct SplatRenderView: NSViewRepresentable {
    @ObservedObject var renderManager: SplatRenderManager

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = renderManager.device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        view.delegate = renderManager

        // Enable user interaction
        view.allowedTouchTypes = []

        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // View updates handled by delegate
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderManager: renderManager)
    }

    class Coordinator {
        let renderManager: SplatRenderManager

        init(renderManager: SplatRenderManager) {
            self.renderManager = renderManager
        }
    }
}

/// Interactive splat view with gesture controls
struct InteractiveSplatView: View {
    @ObservedObject var renderManager: SplatRenderManager
    @State private var lastDragLocation: CGPoint = .zero

    var body: some View {
        ZStack {
            SplatRenderView(renderManager: renderManager)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            let deltaX = Float(value.translation.width - lastDragLocation.x)
                            let deltaY = Float(value.translation.height - lastDragLocation.y)
                            renderManager.pan(deltaX: deltaX, deltaY: deltaY)
                            lastDragLocation = CGPoint(x: value.translation.width, y: value.translation.height)
                        }
                        .onEnded { _ in
                            lastDragLocation = .zero
                        }
                )
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            let delta = Float(value - 1.0)
                            renderManager.zoom(delta: delta)
                        }
                )

            // Loading overlay
            if renderManager.isLoading {
                Color.black.opacity(0.5)
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                        .padding()
                    Text("Loading PLY...")
                        .foregroundColor(.white)
                }
            }

            // Stats overlay
            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(String(format: "%.1f FPS", renderManager.fps))
                        if renderManager.splatCount > 0 {
                            Text("\(formatNumber(renderManager.splatCount)) splats")
                        }
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(4)
                    .padding(8)
                }
                Spacer()
            }
        }
        .cornerRadius(8)
    }

    private func formatNumber(_ n: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
