#!/usr/bin/env python3
"""
SHARP Server - Unix Domain Socket server for 3D Gaussian Splatting generation

For licensing see accompanying LICENSE file.
Copyright (C) 2024

This server accepts image paths via Unix Domain Socket and generates PLY files
using the SHARP model (Apple's single-image 3DGS predictor).

Supports both PyTorch and Core ML backends for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add ml-sharp to path
ML_SHARP_PATH = Path(__file__).parent.parent / "ml-sharp" / "ml-sharp" / "src"
sys.path.insert(0, str(ML_SHARP_PATH))

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

# For direct socket transfer
import base64
import io as sysio
import struct

# Optional Core ML support
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_SOCKET_PATH = "/tmp/webcam_3dgs/server.sock"
DEFAULT_COREML_PATH = Path(__file__).parent / "models" / "sharp.mlpackage"


def importance_sampling(gaussians: Gaussians3D, ratio: float = 1.0) -> Gaussians3D:
    """
    Downsample Gaussians based on importance (opacity).

    Gaussians with higher opacity are considered more important and are
    prioritized for retention. This allows reducing the number of Gaussians
    while maintaining visual quality.

    Args:
        gaussians: Input Gaussians3D object (batch size 1 expected)
        ratio: Fraction of Gaussians to keep (0.0 to 1.0). Default 1.0 (no downsampling).

    Returns:
        Downsampled Gaussians3D object
    """
    if ratio >= 1.0:
        return gaussians

    # Get opacities and flatten
    opacities = gaussians.opacities[0]  # [N] or [N, 1]
    if opacities.ndim > 1:
        opacities = opacities.squeeze(-1)

    n = opacities.shape[0]
    k = max(1, int(n * ratio))

    if k >= n:
        return gaussians

    # Get indices of top-k highest opacity Gaussians
    _, indices = torch.topk(opacities, k)

    # Select Gaussians by indices
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, indices],
        singular_values=gaussians.singular_values[:, indices],
        quaternions=gaussians.quaternions[:, indices],
        colors=gaussians.colors[:, indices],
        opacities=gaussians.opacities[:, indices],
    )


def random_sampling(gaussians: Gaussians3D, ratio: float = 1.0) -> Gaussians3D:
    """
    Randomly downsample Gaussians.

    This is a simpler alternative to importance_sampling that doesn't
    prioritize by opacity.

    Args:
        gaussians: Input Gaussians3D object (batch size 1 expected)
        ratio: Fraction of Gaussians to keep (0.0 to 1.0). Default 1.0 (no downsampling).

    Returns:
        Downsampled Gaussians3D object
    """
    if ratio >= 1.0:
        return gaussians

    n = gaussians.mean_vectors.shape[1]
    k = max(1, int(n * ratio))

    if k >= n:
        return gaussians

    # Random permutation and select first k
    indices = torch.randperm(n, device=gaussians.mean_vectors.device)[:k]

    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, indices],
        singular_values=gaussians.singular_values[:, indices],
        quaternions=gaussians.quaternions[:, indices],
        colors=gaussians.colors[:, indices],
        opacities=gaussians.opacities[:, indices],
    )


class SHARPServer:
    """Unix Domain Socket server for PLY generation using SHARP model."""

    def __init__(
        self,
        socket_path: str,
        device: str = "mps",
        use_fp16: bool = False,
        use_coreml: bool = False,
        coreml_path: str | None = None,
        sampling_ratio: float = 1.0,
        sampling_method: str = "importance",
    ):
        """
        Initialize the SHARP server.

        Args:
            socket_path: Path for the Unix Domain Socket
            device: Device to run inference on ('cpu', 'mps', 'cuda')
            use_fp16: Enable FP16 (half precision) inference for faster processing
                      Note: Currently disabled by default due to compatibility issues
                      with the SHARP model architecture on MPS.
            use_coreml: Use Core ML backend for inference (faster on Apple Silicon)
            coreml_path: Path to Core ML model (uses default if not provided)
            sampling_ratio: Fraction of Gaussians to keep (0.0-1.0). Default 1.0 (no downsampling).
            sampling_method: Method for downsampling ('importance' or 'random'). Default 'importance'.
        """
        self.socket_path = socket_path
        self.device = self._resolve_device(device)
        self.predictor = None
        self.coreml_model = None
        # Original SHARP model resolution - required for model compatibility
        # Reducing to 1024x1024 causes tensor size mismatch errors
        self.internal_shape = (1536, 1536)
        # FP16 mode for additional speedup (~20-40% on supported hardware)
        # Currently disabled by default due to tensor size mismatch issues in SHARP model
        # TODO: Investigate SHARP model FP16 compatibility
        self.use_fp16 = use_fp16 and self.device in ("cuda", "mps")
        # Core ML mode for Apple Neural Engine acceleration
        self.use_coreml = use_coreml and COREML_AVAILABLE
        self.coreml_path = Path(coreml_path) if coreml_path else DEFAULT_COREML_PATH
        # Gaussian sampling settings for reducing output size
        self.sampling_ratio = min(1.0, max(0.01, sampling_ratio))
        self.sampling_method = sampling_method

    def _resolve_device(self, device: str) -> str:
        """Resolve the device string to an available device."""
        if device == "default":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_model(self, checkpoint_path: str | None = None):
        """Load the SHARP predictor model."""
        if self.use_coreml:
            self._load_coreml_model()
        else:
            self._load_pytorch_model(checkpoint_path)

    def _load_coreml_model(self):
        """Load the Core ML model for inference."""
        if not COREML_AVAILABLE:
            raise RuntimeError("coremltools is not installed. Install with: pip install coremltools")

        if not self.coreml_path.exists():
            raise FileNotFoundError(
                f"Core ML model not found at {self.coreml_path}. "
                f"Run convert_to_coreml.py to create it first."
            )

        LOGGER.info(f"Loading Core ML model from {self.coreml_path}")
        self.coreml_model = ct.models.MLModel(str(self.coreml_path))
        LOGGER.info("Core ML model loaded successfully (using Apple Neural Engine)")

    def _load_pytorch_model(self, checkpoint_path: str | None = None):
        """Load the PyTorch model for inference."""
        LOGGER.info(f"Loading SHARP model on device: {self.device}")

        if checkpoint_path is None:
            LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, progress=True
            )
        else:
            LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, weights_only=True)

        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)

        # Apply FP16 conversion for faster inference
        if self.use_fp16:
            self.predictor = self.predictor.half()
            LOGGER.info("FP16 (half precision) mode enabled for faster inference")

        LOGGER.info("PyTorch model loaded successfully")

    def _apply_sampling(self, gaussians: Gaussians3D) -> tuple[Gaussians3D, int, int]:
        """
        Apply Gaussian sampling if configured.

        Args:
            gaussians: Input Gaussians3D object

        Returns:
            Tuple of (sampled_gaussians, original_count, final_count)
        """
        original_count = gaussians.mean_vectors.shape[1]

        if self.sampling_ratio >= 1.0:
            return gaussians, original_count, original_count

        if self.sampling_method == "importance":
            sampled = importance_sampling(gaussians, self.sampling_ratio)
        else:
            sampled = random_sampling(gaussians, self.sampling_ratio)

        final_count = sampled.mean_vectors.shape[1]
        LOGGER.info(f"Sampling: {original_count:,} -> {final_count:,} Gaussians ({self.sampling_ratio*100:.0f}%, {self.sampling_method})")

        return sampled, original_count, final_count

    @torch.no_grad()
    def process_image(self, image_path: str, output_path: str) -> dict[str, Any]:
        """
        Generate PLY from an image.

        Args:
            image_path: Path to input image
            output_path: Path for output PLY file

        Returns:
            Result dictionary with success status and metadata
        """
        start_time = time.time()

        try:
            # Load image
            image, _, f_px = sharp_io.load_rgb(Path(image_path))
            height, width = image.shape[:2]

            LOGGER.info(f"Processing image: {width}x{height}, f_px={f_px:.2f}")

            # Run inference using appropriate backend
            if self.use_coreml:
                gaussians_ndc = self._run_coreml_inference(image, f_px, width)
            else:
                gaussians_ndc = self._run_pytorch_inference(image, f_px, width)

            # Build intrinsics (use FP32 for precision in matrix operations)
            intrinsics = (
                torch.tensor([
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ])
                .float()
                .to(self.device)
            )
            intrinsics_resized = intrinsics.clone()
            intrinsics_resized[0] *= self.internal_shape[0] / width
            intrinsics_resized[1] *= self.internal_shape[1] / height

            # Unproject to metric space
            gaussians = unproject_gaussians(
                gaussians_ndc,
                torch.eye(4).float().to(self.device),
                intrinsics_resized,
                self.internal_shape,
            )

            # Apply Gaussian sampling (if configured)
            gaussians, original_count, gaussian_count = self._apply_sampling(gaussians)

            # Save PLY
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_ply(gaussians, f_px, (height, width), Path(output_path))

            processing_time_ms = (time.time() - start_time) * 1000

            backend = "CoreML" if self.use_coreml else "PyTorch"
            LOGGER.info(
                f"Generated PLY with {gaussian_count:,} Gaussians in {processing_time_ms:.1f}ms ({backend})"
            )

            return {
                "success": True,
                "ply_path": output_path,
                "gaussian_count": gaussian_count,
                "original_gaussian_count": original_count,
                "processing_time_ms": processing_time_ms,
                "metadata": {
                    "image_width": width,
                    "image_height": height,
                    "focal_length_px": f_px,
                    "backend": backend,
                    "sampling_ratio": self.sampling_ratio,
                    "sampling_method": self.sampling_method,
                },
                "error": None,
            }

        except Exception as e:
            LOGGER.error(f"Failed to process image: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "ply_path": None,
                "gaussian_count": 0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "metadata": None,
                "error": str(e),
            }

    @torch.no_grad()
    def process_image_direct(self, image_path: str) -> dict[str, Any]:
        """
        Generate PLY from an image and return data directly via socket.

        This method returns PLY data as Base64-encoded binary instead of
        writing to a file, eliminating file I/O overhead (~100-200ms savings).

        Args:
            image_path: Path to input image

        Returns:
            Result dictionary with success status, PLY data, and metadata
        """
        start_time = time.time()

        try:
            # Load image
            image, _, f_px = sharp_io.load_rgb(Path(image_path))
            height, width = image.shape[:2]

            LOGGER.info(f"Processing image (direct): {width}x{height}, f_px={f_px:.2f}")

            # Run inference using appropriate backend
            if self.use_coreml:
                gaussians_ndc = self._run_coreml_inference(image, f_px, width)
            else:
                gaussians_ndc = self._run_pytorch_inference(image, f_px, width)

            # Build intrinsics (use FP32 for precision in matrix operations)
            intrinsics = (
                torch.tensor([
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ])
                .float()
                .to(self.device)
            )
            intrinsics_resized = intrinsics.clone()
            intrinsics_resized[0] *= self.internal_shape[0] / width
            intrinsics_resized[1] *= self.internal_shape[1] / height

            # Unproject to metric space
            gaussians = unproject_gaussians(
                gaussians_ndc,
                torch.eye(4).float().to(self.device),
                intrinsics_resized,
                self.internal_shape,
            )

            # Apply Gaussian sampling (if configured)
            gaussians, original_count, gaussian_count = self._apply_sampling(gaussians)

            # Generate PLY data in memory
            ply_data = self._generate_ply_bytes(gaussians, f_px, (height, width))

            processing_time_ms = (time.time() - start_time) * 1000

            backend = "CoreML" if self.use_coreml else "PyTorch"
            LOGGER.info(
                f"Generated PLY (direct) with {gaussian_count:,} Gaussians in {processing_time_ms:.1f}ms ({backend}), {len(ply_data)} bytes"
            )

            return {
                "success": True,
                "ply_data": base64.b64encode(ply_data).decode("ascii"),
                "ply_size": len(ply_data),
                "gaussian_count": gaussian_count,
                "original_gaussian_count": original_count,
                "processing_time_ms": processing_time_ms,
                "metadata": {
                    "image_width": width,
                    "image_height": height,
                    "focal_length_px": f_px,
                    "backend": backend,
                    "sampling_ratio": self.sampling_ratio,
                    "sampling_method": self.sampling_method,
                },
                "error": None,
            }

        except Exception as e:
            LOGGER.error(f"Failed to process image (direct): {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "ply_data": None,
                "ply_size": 0,
                "gaussian_count": 0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "metadata": None,
                "error": str(e),
            }

    def _generate_ply_bytes(
        self, gaussians: Gaussians3D, focal_length: float, image_shape: tuple[int, int]
    ) -> bytes:
        """
        Generate PLY file data in memory as bytes.

        This is a memory-based version of sharp.utils.gaussians.save_ply,
        avoiding file I/O overhead.
        """
        def _inverse_sigmoid(x):
            """Convert probability to logit."""
            return np.log(x / (1.0 - x))

        def _convert_rgb_to_spherical_harmonics(rgb):
            """Convert RGB to degree-0 spherical harmonics."""
            coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
            return (rgb - 0.5) / coeff_degree0

        # Extract data from Gaussians (move to CPU and convert to numpy)
        xyz = gaussians.mean_vectors[0].cpu().numpy()  # [N, 3]

        # Convert singular_values to log scale (same as save_ply)
        scale_logits = np.log(gaussians.singular_values[0].cpu().numpy())  # [N, 3]

        rotations = gaussians.quaternions[0].cpu().numpy()  # [N, 4]

        # Convert colors: linearRGB -> sRGB -> spherical harmonics (same as save_ply)
        colors_linear = gaussians.colors[0].cpu()
        colors_srgb = cs_utils.linearRGB2sRGB(colors_linear).numpy()
        colors_sh = _convert_rgb_to_spherical_harmonics(colors_srgb)  # [N, 3]

        # Convert opacity to logits (same as save_ply)
        opacities_raw = gaussians.opacities[0].cpu().numpy()  # [N, 1] or [N]
        if opacities_raw.ndim == 1:
            opacities_raw = opacities_raw[:, np.newaxis]
        # Clamp to avoid log(0) or log(inf)
        opacities_clamped = np.clip(opacities_raw, 1e-6, 1.0 - 1e-6)
        opacity_logits = _inverse_sigmoid(opacities_clamped)  # [N, 1]

        n_gaussians = xyz.shape[0]

        # Build PLY header (same format as save_ply, without normals)
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_gaussians}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

        buffer = sysio.BytesIO()
        buffer.write(header.encode("ascii"))

        # Write binary data for each Gaussian
        # Format: x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        for i in range(n_gaussians):
            # Position (x, y, z)
            buffer.write(struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            # Colors (f_dc_0, f_dc_1, f_dc_2) - spherical harmonics format
            buffer.write(struct.pack("<fff", colors_sh[i, 0], colors_sh[i, 1], colors_sh[i, 2]))
            # Opacity (logit)
            buffer.write(struct.pack("<f", opacity_logits[i, 0]))
            # Scales (scale_0, scale_1, scale_2) - log scale
            buffer.write(struct.pack("<fff", scale_logits[i, 0], scale_logits[i, 1], scale_logits[i, 2]))
            # Rotation quaternion (rot_0, rot_1, rot_2, rot_3)
            buffer.write(struct.pack("<ffff", rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))

        return buffer.getvalue()

    def _run_pytorch_inference(self, image: np.ndarray, f_px: float, width: int) -> Gaussians3D:
        """Run inference using PyTorch backend."""
        # Prepare tensor with appropriate dtype based on FP16 mode
        dtype = torch.float16 if self.use_fp16 else torch.float32
        image_pt = (
            torch.from_numpy(image.copy())
            .to(dtype)
            .to(self.device)
            .permute(2, 0, 1)
            / 255.0
        )
        disparity_factor = torch.tensor([f_px / width], dtype=dtype, device=self.device)

        # Resize for model
        image_resized = F.interpolate(
            image_pt[None],
            size=(self.internal_shape[1], self.internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        # Run inference
        LOGGER.info(f"Running PyTorch inference... (FP16={self.use_fp16})")
        gaussians_ndc = self.predictor(image_resized, disparity_factor)

        # Convert back to FP32 for post-processing if using FP16
        if self.use_fp16:
            gaussians_ndc = Gaussians3D(
                mean_vectors=gaussians_ndc.mean_vectors.float(),
                singular_values=gaussians_ndc.singular_values.float(),
                quaternions=gaussians_ndc.quaternions.float(),
                colors=gaussians_ndc.colors.float(),
                opacities=gaussians_ndc.opacities.float(),
            )

        return gaussians_ndc

    def _run_coreml_inference(self, image: np.ndarray, f_px: float, width: int) -> Gaussians3D:
        """Run inference using Core ML backend with PyTorch fallback.

        If Core ML fails (e.g., due to ANE incompatibility), automatically
        falls back to PyTorch inference.
        """
        # Prepare input
        image_pt = (
            torch.from_numpy(image.copy())
            .float()
            .permute(2, 0, 1)
            / 255.0
        )
        disparity_factor = np.array([f_px / width], dtype=np.float32)

        # Resize for model
        image_resized = F.interpolate(
            image_pt[None],
            size=(self.internal_shape[1], self.internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        # Run Core ML inference with fallback
        LOGGER.info("Running Core ML inference...")
        try:
            coreml_inputs = {
                "image": image_resized.numpy(),
                "disparity_factor": disparity_factor,
            }
            outputs = self.coreml_model.predict(coreml_inputs)

            # Convert outputs to Gaussians3D
            gaussians_ndc = Gaussians3D(
                mean_vectors=torch.from_numpy(outputs["mean_vectors"]).to(self.device),
                singular_values=torch.from_numpy(outputs["singular_values"]).to(self.device),
                quaternions=torch.from_numpy(outputs["quaternions"]).to(self.device),
                colors=torch.from_numpy(outputs["colors"]).to(self.device),
                opacities=torch.from_numpy(outputs["opacities"]).to(self.device),
            )

            return gaussians_ndc

        except RuntimeError as e:
            error_str = str(e)
            # Check for ANE-related errors
            if "ANE" in error_str or "Neural Engine" in error_str or "E5RT" in error_str:
                LOGGER.warning(f"Core ML ANE failed, falling back to PyTorch: {error_str[:200]}")
                return self._run_pytorch_inference(image, f_px, width)
            # Re-raise other RuntimeErrors
            raise

    def handle_client(self, client_socket: socket.socket):
        """Handle a single client connection."""
        try:
            # Receive request
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Check for complete JSON
                try:
                    request = json.loads(data.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    continue

            if not data:
                return

            LOGGER.info(f"Received request: {request.get('command', 'unknown')}")

            response: dict[str, Any]
            command = request.get("command", "")

            if command == "generate":
                input_path = request.get("input_path")
                output_path = request.get("output_path")

                if not input_path or not output_path:
                    response = {
                        "success": False,
                        "error": "Missing input_path or output_path",
                    }
                else:
                    response = self.process_image(input_path, output_path)

            elif command == "generate_direct":
                # Direct transfer mode - returns PLY data in response instead of file
                input_path = request.get("input_path")

                if not input_path:
                    response = {
                        "success": False,
                        "error": "Missing input_path",
                    }
                else:
                    response = self.process_image_direct(input_path)

            elif command == "ping":
                response = {"success": True, "message": "pong"}

            elif command == "status":
                response = {
                    "success": True,
                    "device": self.device,
                    "model_loaded": self.predictor is not None or self.coreml_model is not None,
                    "backend": "CoreML" if self.use_coreml else "PyTorch",
                }

            elif command == "shutdown":
                response = {"success": True, "message": "Shutting down"}
                client_socket.sendall(json.dumps(response).encode("utf-8"))
                client_socket.close()
                raise SystemExit("Shutdown requested")

            else:
                response = {"success": False, "error": f"Unknown command: {command}"}

            # Send response
            client_socket.sendall(json.dumps(response).encode("utf-8"))

        except Exception as e:
            LOGGER.error(f"Error handling client: {e}")
            try:
                error_response = {"success": False, "error": str(e)}
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
            except:
                pass
        finally:
            client_socket.close()

    def run(self):
        """Run the server main loop."""
        # Ensure directory exists
        socket_dir = os.path.dirname(self.socket_path)
        os.makedirs(socket_dir, exist_ok=True)

        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Create socket
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(self.socket_path)
        server_socket.listen(5)

        LOGGER.info(f"SHARP Server listening on {self.socket_path}")
        LOGGER.info(f"Device: {self.device}")

        try:
            while True:
                client_socket, _ = server_socket.accept()
                self.handle_client(client_socket)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("Server shutting down...")
        finally:
            server_socket.close()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)


def main():
    parser = argparse.ArgumentParser(
        description="SHARP Server for 3D Gaussian Splatting generation"
    )
    parser.add_argument(
        "--socket",
        type=str,
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix Domain Socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "mps", "cuda", "default"],
        help="Device to run inference on (default: mps)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (downloads default if not provided)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 (half precision) inference for faster processing (disabled by default due to compatibility issues)",
    )
    parser.add_argument(
        "--coreml",
        action="store_true",
        help="Use Core ML backend for inference (faster on Apple Silicon, requires converted model)",
    )
    parser.add_argument(
        "--coreml-path",
        type=str,
        default=None,
        help="Path to Core ML model (uses default models/sharp.mlpackage if not provided)",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=1.0,
        help="Fraction of Gaussians to keep (0.0-1.0). Default 1.0 (no downsampling). Example: 0.5 keeps 50%% of Gaussians.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="importance",
        choices=["importance", "random"],
        help="Sampling method: 'importance' prioritizes high-opacity Gaussians, 'random' samples uniformly. Default: importance.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # FP16 is disabled by default due to SHARP model compatibility issues
    use_fp16 = args.fp16

    # Check Core ML availability
    if args.coreml and not COREML_AVAILABLE:
        LOGGER.warning("Core ML requested but coremltools not installed. Falling back to PyTorch.")
        args.coreml = False

    server = SHARPServer(
        args.socket,
        args.device,
        use_fp16=use_fp16,
        use_coreml=args.coreml,
        coreml_path=args.coreml_path,
        sampling_ratio=args.sampling_ratio,
        sampling_method=args.sampling_method,
    )

    if args.sampling_ratio < 1.0:
        LOGGER.info(f"Gaussian sampling enabled: {args.sampling_ratio*100:.0f}% ({args.sampling_method})")

    server.load_model(args.checkpoint)
    server.run()


if __name__ == "__main__":
    main()
