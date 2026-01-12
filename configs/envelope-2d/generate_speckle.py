#!/usr/bin/env python
"""
Generate a speckle pattern .npz file for the SpeckledDriver.

Example usage:
    python generate_speckle.py --ny 180 --num_colors 1 --output speckle.npz

This creates a square filter localized to the middle of the y-domain.
"""

import argparse
import numpy as np


def next_smooth_fft_size(n: int, max_prime: int = 5) -> int:
    """Find the next FFT-friendly size (only factors 2, 3, 5)."""
    def is_smooth(x):
        for p in [2, 3, 5]:
            while x % p == 0:
                x //= p
        return x == 1

    while not is_smooth(n):
        n += 1
    return n


def compute_ny(ymin_um: float, ymax_um: float, dx_nm: float) -> int:
    """Compute ny from grid parameters (matching helpers.py logic)."""
    dx_um = dx_nm / 1000.0
    ny = int((ymax_um - ymin_um) / dx_um)
    return next_smooth_fft_size(ny, max_prime=5)


def generate_square_filter_speckle(
    ny: int,
    num_colors: int,
    center_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate speckle pattern with square filter in the middle of the domain.

    Parameters
    ----------
    ny : int
        Number of y grid points
    num_colors : int
        Number of frequency components
    center_fraction : float
        Fraction of domain covered by the filter (0.5 = middle 50%)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    intensities : ndarray, shape (num_colors, ny)
    phases : ndarray, shape (num_colors, ny)
    """
    rng = np.random.default_rng(seed)

    # Create y coordinate (normalized to [-1, 1])
    y_norm = np.linspace(-1, 1, ny)

    # Square filter: 1 in the middle, 0 at edges
    half_width = center_fraction / 2
    filter_mask = np.where(np.abs(y_norm) <= half_width, 1.0, 0.0)

    # Apply filter to each color
    intensities = np.zeros((num_colors, ny))
    phases = np.zeros((num_colors, ny))

    for i in range(num_colors):
        # Base intensity with filter applied
        intensities[i, :] = filter_mask

        # Random phases in [-1, 1] (will be scaled to [-pi, pi] by driver)
        phases[i, :] = rng.uniform(-1, 1, ny)

    return intensities, phases


def main():
    parser = argparse.ArgumentParser(description="Generate speckle pattern for SpeckledDriver")
    parser.add_argument("--ny", type=int, default=None, help="Number of y grid points")
    parser.add_argument("--ymin", type=float, default=-6.0, help="ymin in um (default: -6)")
    parser.add_argument("--ymax", type=float, default=6.0, help="ymax in um (default: 6)")
    parser.add_argument("--dx", type=float, default=66.3, help="dx in nm (default: 66.3)")
    parser.add_argument("--num_colors", type=int, default=1, help="Number of frequency components")
    parser.add_argument("--center_fraction", type=float, default=0.5, help="Fraction of domain for filter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="speckle.npz", help="Output file path")

    args = parser.parse_args()

    # Compute ny if not provided
    if args.ny is None:
        ny = compute_ny(args.ymin, args.ymax, args.dx)
        print(f"Computed ny = {ny} from grid parameters")
    else:
        ny = args.ny

    # Generate speckle pattern
    intensities, phases = generate_square_filter_speckle(
        ny=ny,
        num_colors=args.num_colors,
        center_fraction=args.center_fraction,
        seed=args.seed,
    )

    # Save to file
    np.savez(
        args.output,
        intensities=intensities,
        phases=phases,
    )

    print(f"Saved speckle pattern to {args.output}")
    print(f"  intensities shape: {intensities.shape}")
    print(f"  phases shape: {phases.shape}")
    print(f"  center_fraction: {args.center_fraction}")


if __name__ == "__main__":
    main()
