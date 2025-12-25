#!/usr/bin/env python3
"""
Convert SHARP PyTorch model to Core ML format.

For licensing see accompanying LICENSE file.
Copyright (C) 2024

This script converts the SHARP model to Core ML format for faster inference
on Apple devices using the Apple Neural Engine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

# Add ml-sharp to path
ML_SHARP_PATH = Path(__file__).parent.parent / "ml-sharp" / "ml-sharp" / "src"
sys.path.insert(0, str(ML_SHARP_PATH))

from sharp.models import PredictorParams, create_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class SHARPInferenceWrapper(nn.Module):
    """Wrapper for SHARP model that simplifies the interface for Core ML conversion.

    Core ML conversion works better with simpler input/output signatures.
    This wrapper:
    1. Takes image and disparity_factor as inputs
    2. Returns Gaussian parameters as a tuple of tensors
    """

    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor
        # Exclude depth_alignment from tracing as it contains conditional logic
        self.predictor.depth_alignment = None

    def forward(self, image: torch.Tensor, disparity_factor: torch.Tensor) -> tuple:
        """Run inference and return Gaussian parameters.

        Args:
            image: Input image tensor [1, 3, H, W]
            disparity_factor: Disparity factor [1]

        Returns:
            Tuple of (mean_vectors, singular_values, quaternions, colors, opacities)
        """
        # Run the model components manually to avoid conditional logic issues
        monodepth_output = self.predictor.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor_expanded = disparity_factor[:, None, None, None]
        monodepth = disparity_factor_expanded / monodepth_disparity.clamp(min=1e-4, max=1e4)

        # Skip depth alignment (not needed for inference without ground truth)
        init_output = self.predictor.init_model(image, monodepth)
        image_features = self.predictor.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta_values = self.predictor.prediction_head(image_features)
        gaussians = self.predictor.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

        return (
            gaussians.mean_vectors,
            gaussians.singular_values,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacities,
        )


def load_pytorch_model(checkpoint_path: str | None = None) -> nn.Module:
    """Load the SHARP PyTorch model."""
    LOGGER.info("Loading SHARP PyTorch model...")

    if checkpoint_path is None:
        LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True
        )
    else:
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    LOGGER.info("PyTorch model loaded successfully")
    return predictor


def convert_to_coreml(
    predictor: nn.Module,
    output_path: str,
    compute_units: str = "ALL",
) -> None:
    """Convert SHARP model to Core ML format.

    Args:
        predictor: The SHARP PyTorch model
        output_path: Path to save the Core ML model
        compute_units: Target compute units ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")
    """
    LOGGER.info("Preparing model for Core ML conversion...")

    # Wrap the predictor
    wrapper = SHARPInferenceWrapper(predictor)
    wrapper.eval()

    # Create example inputs
    # SHARP expects 1536x1536 input
    example_image = torch.randn(1, 3, 1536, 1536)
    example_disparity = torch.tensor([1.0])

    LOGGER.info("Tracing model with TorchScript...")
    try:
        # Try scripting first (better for models with control flow)
        traced_model = torch.jit.trace(
            wrapper,
            (example_image, example_disparity),
            strict=False,
        )
    except Exception as e:
        LOGGER.warning(f"Tracing failed: {e}")
        LOGGER.info("Attempting scripting instead...")
        traced_model = torch.jit.script(wrapper)

    LOGGER.info("Converting to Core ML...")

    # Map compute units string to enum
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, 1536, 1536)),
            ct.TensorType(name="disparity_factor", shape=(1,)),
        ],
        outputs=[
            ct.TensorType(name="mean_vectors"),
            ct.TensorType(name="singular_values"),
            ct.TensorType(name="quaternions"),
            ct.TensorType(name="colors"),
            ct.TensorType(name="opacities"),
        ],
        compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.macOS14,
    )

    # Add metadata
    mlmodel.author = "MetalSplatter"
    mlmodel.short_description = "SHARP: Single-image 3D Gaussian Splatting predictor"
    mlmodel.version = "1.0.0"

    # Save the model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    LOGGER.info(f"Core ML model saved to {output_path}")

    # Print model info
    LOGGER.info(f"Model spec: {mlmodel.get_spec().description}")


def verify_conversion(
    pytorch_model: nn.Module,
    coreml_path: str,
) -> None:
    """Verify that Core ML model produces similar outputs to PyTorch."""
    LOGGER.info("Verifying Core ML conversion...")

    import coremltools as ct

    # Load Core ML model
    mlmodel = ct.models.MLModel(coreml_path)

    # Create test input
    test_image = torch.randn(1, 3, 1536, 1536)
    test_disparity = torch.tensor([1.0])

    # Run PyTorch inference
    wrapper = SHARPInferenceWrapper(pytorch_model)
    wrapper.eval()
    with torch.no_grad():
        pytorch_outputs = wrapper(test_image, test_disparity)

    # Run Core ML inference
    coreml_inputs = {
        "image": test_image.numpy(),
        "disparity_factor": test_disparity.numpy(),
    }
    coreml_outputs = mlmodel.predict(coreml_inputs)

    # Compare outputs
    output_names = ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]
    for i, name in enumerate(output_names):
        pytorch_out = pytorch_outputs[i].numpy()
        coreml_out = coreml_outputs[name]

        max_diff = np.abs(pytorch_out - coreml_out).max()
        mean_diff = np.abs(pytorch_out - coreml_out).mean()

        LOGGER.info(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if max_diff > 1e-3:
            LOGGER.warning(f"Large difference detected in {name}!")

    LOGGER.info("Verification complete")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SHARP PyTorch model to Core ML format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (downloads default if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/sharp.mlpackage",
        help="Output path for Core ML model",
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        help="Target compute units for Core ML",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by comparing outputs",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load PyTorch model
    predictor = load_pytorch_model(args.checkpoint)

    # Convert to Core ML
    convert_to_coreml(predictor, args.output, args.compute_units)

    # Optionally verify conversion
    if args.verify:
        verify_conversion(predictor, args.output)

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
