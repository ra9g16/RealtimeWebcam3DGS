import SwiftUI
import AVFoundation

/// Control panel for camera and capture settings
struct ControlPanelView: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Mode selection
            Picker("Mode", selection: $coordinator.appMode) {
                ForEach(AppCoordinator.AppMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Divider()

            // Mode-specific controls
            if coordinator.appMode == .webcam {
                // Camera selection
                CameraSelectionSection(coordinator: coordinator)

                Divider()

                // Capture controls
                CaptureControlsSection(coordinator: coordinator)
            } else {
                // Static image controls
                StaticModeSection(coordinator: coordinator)
            }

            Divider()

            // Server status (common)
            ServerStatusSection(coordinator: coordinator)

            Divider()

            // Head Tracking (common)
            HeadTrackingSection(coordinator: coordinator)

            Spacer()
        }
        .padding()
    }
}

struct CameraSelectionSection: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Camera")
                .font(.headline)

            Picker("Camera", selection: Binding(
                get: { coordinator.cameraManager.selectedCamera?.uniqueID ?? "" },
                set: { coordinator.cameraManager.selectCamera($0) }
            )) {
                ForEach(coordinator.cameraManager.availableCameras, id: \.uniqueID) { camera in
                    Text(camera.localizedName)
                        .tag(camera.uniqueID)
                }
            }
            .labelsHidden()

            Button(action: {
                coordinator.cameraManager.refreshAvailableCameras()
            }) {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
        }
    }
}

struct CaptureControlsSection: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Capture")
                .font(.headline)

            // Capture interval
            HStack {
                Text("Interval:")
                Slider(value: $coordinator.captureIntervalSeconds, in: 2...30, step: 1)
                Text("\(Int(coordinator.captureIntervalSeconds))s")
                    .monospacedDigit()
                    .frame(width: 30)
            }

            // Auto capture toggle
            Toggle("Auto Capture", isOn: $coordinator.autoCapture)

            // Control buttons
            HStack(spacing: 12) {
                Button(action: {
                    Task {
                        await coordinator.startPipeline()
                    }
                }) {
                    Label("Start", systemImage: "play.fill")
                }
                .disabled(coordinator.appState == .capturing || coordinator.appState == .generating)

                Button(action: {
                    coordinator.stopPipeline()
                }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .disabled(coordinator.appState == .idle)

                Button(action: {
                    Task {
                        await coordinator.captureAndGenerate()
                    }
                }) {
                    Label("Capture Now", systemImage: "camera.fill")
                }
                .disabled(!coordinator.cameraManager.isCapturing)
            }
        }
    }
}

struct ServerStatusSection: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SHARP Server")
                .font(.headline)

            HStack {
                Circle()
                    .fill(coordinator.serverRunning ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(coordinator.serverRunning ? "Running" : "Stopped")
            }

            if !coordinator.serverRunning {
                Button("Start Server") {
                    Task {
                        await coordinator.startServer()
                    }
                }
            } else {
                Button("Stop Server") {
                    Task {
                        await coordinator.stopServer()
                    }
                }
            }
        }
    }
}

struct StaticModeSection: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Static Image")
                .font(.headline)

            // Import button
            Button(action: {
                Task {
                    await coordinator.importImage()
                }
            }) {
                Label("Import Image", systemImage: "photo.on.rectangle")
            }
            .disabled(coordinator.appState == .generating)

            // Preview of imported image
            if let image = coordinator.importedImage {
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 120)
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )

                if let path = coordinator.importedImagePath {
                    Text(path.lastPathComponent)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            // Processing indicator
            if coordinator.appState == .generating {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Converting to 3DGS...")
                        .foregroundColor(.secondary)
                }
            }

            // Processing time
            if coordinator.appState == .rendering && coordinator.lastProcessingTimeMs > 0 {
                Text("Processed in \(String(format: "%.0f", coordinator.lastProcessingTimeMs))ms")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct HeadTrackingSection: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Head Tracking")
                .font(.headline)

            Toggle("Enable Parallax", isOn: $coordinator.useHeadTracking)

            if coordinator.useHeadTracking {
                // Face detection status
                HStack {
                    Circle()
                        .fill(coordinator.faceDetected ? Color.green : Color.orange)
                        .frame(width: 10, height: 10)
                    Text(coordinator.faceDetected ? "Face Detected" : "No Face")
                        .foregroundColor(.secondary)
                }

                // Sensitivity slider (rotation-based: 0.1 to 1.0 radians per head unit)
                HStack {
                    Text("Sensitivity:")
                    Slider(value: $coordinator.headTrackingSensitivity, in: 0.1...1.0, step: 0.1)
                    Text(String(format: "%.1f", coordinator.headTrackingSensitivity))
                        .monospacedDigit()
                        .frame(width: 30)
                }
            }
        }
    }
}

/// Status display panel
struct StatusPanelView: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Status")
                .font(.headline)

            // App state
            HStack {
                stateIndicator(for: coordinator.appState)
                Text(coordinator.appState.description)
            }

            // Countdown to next capture
            if coordinator.autoCapture && coordinator.appState != .idle {
                HStack {
                    Image(systemName: "clock")
                    Text("Next capture: \(coordinator.secondsUntilNextCapture)s")
                        .monospacedDigit()
                }
                .foregroundColor(.secondary)
            }

            // Processing queue
            if !coordinator.processingQueue.isEmpty {
                Text("Queue: \(coordinator.processingQueue.count) items")
                    .foregroundColor(.secondary)
            }

            // Error message
            if let error = coordinator.errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }

    @ViewBuilder
    private func stateIndicator(for state: AppCoordinator.AppState) -> some View {
        switch state {
        case .idle:
            Image(systemName: "circle")
                .foregroundColor(.gray)
        case .capturing:
            Image(systemName: "camera.fill")
                .foregroundColor(.blue)
        case .generating:
            ProgressView()
                .scaleEffect(0.7)
        case .rendering:
            Image(systemName: "cube.fill")
                .foregroundColor(.green)
        case .error:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
        }
    }
}
