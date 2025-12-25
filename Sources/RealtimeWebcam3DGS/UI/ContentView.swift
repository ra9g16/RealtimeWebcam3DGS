import SwiftUI

/// Main content view with split layout
struct ContentView: View {
    @StateObject private var coordinator = AppCoordinator()

    var body: some View {
        HSplitView {
            // Left side: Camera/Image preview and controls
            VStack(spacing: 0) {
                // Mode-dependent preview
                if coordinator.appMode == .webcam {
                    CameraPreviewView(cameraManager: coordinator.cameraManager)
                        .frame(minHeight: 200)
                } else {
                    StaticImagePreviewView(coordinator: coordinator)
                        .frame(minHeight: 200)
                }

                Divider()

                // Status panel
                StatusPanelView(coordinator: coordinator)
                    .padding(.horizontal)

                Divider()

                // Control panel
                ScrollView {
                    ControlPanelView(coordinator: coordinator)
                }
            }
            .frame(minWidth: 300, idealWidth: 350, maxWidth: 400)

            // Right side: 3D Gaussian Splat renderer
            InteractiveSplatView(renderManager: coordinator.renderManager)
                .frame(minWidth: 400)
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            Task {
                await coordinator.initialize()
            }
        }
        .onDisappear {
            Task {
                await coordinator.cleanup()
            }
        }
    }
}

#Preview {
    ContentView()
}
