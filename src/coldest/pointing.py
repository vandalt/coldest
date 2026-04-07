import numpy as np
from scipy.ndimage import uniform_filter, convolve


def find_regions(
    mask,
    window_size,
    kernel="uniform",
    n_top=10,
    forbidden_size=None,
    min_edge_distance=None,
    return_weighted=False,
):
    # Default min_edge_distance to window_size
    if min_edge_distance is None:
        min_edge_distance = window_size
    # Average of dq_mask in 70x70 region centered on each pixel
    if isinstance(kernel, np.ndarray):
        dq_filtered = convolve(mask.astype(float), kernel, mode="constant")
    elif kernel == "uniform":
        dq_filtered = uniform_filter(
            mask.astype(float), size=window_size, mode="constant"
        )
    else:
        raise ValueError(f"Invalid kernel {kernel}")

    # Convert average to count per region
    dq_count = dq_filtered * window_size**2

    # Apply forbidden region filter if specified
    if forbidden_size is not None:
        if forbidden_size >= window_size:
            raise ValueError(
                f"forbidden_size ({forbidden_size}) must be smaller than window_size ({window_size})"
            )

        # Create a small kernel to detect any True values in the forbidden central region
        forbidden_kernel = np.ones((forbidden_size, forbidden_size))

        # Convolve: result > 0 means at least one True in the forbidden region
        forbidden_check = convolve(
            mask.astype(float), forbidden_kernel, mode="constant"
        )

        # Mark regions with any forbidden pixels as invalid (set to inf)
        dq_count = np.where(forbidden_check > 0, np.inf, dq_count)

    flat_dq_count = dq_count.flatten()

    overlap_ok = False
    if overlap_ok:
        n_top = 10
        sorted_idx = np.argsort(flat_dq_count)[:n_top]
        best_x, best_y = np.unravel_index(sorted_idx, dq_count.shape)
    else:
        # Select non-overlapping positions
        selected = []
        sorted_idx = np.argsort(flat_dq_count)
        all_rows, all_cols = np.unravel_index(sorted_idx, dq_count.shape)
        mask_height, mask_width = dq_count.shape

        for i in range(len(sorted_idx)):
            row, col = int(all_rows[i]), int(all_cols[i])
            weighted_sum = float(flat_dq_count[sorted_idx[i]])

            # Skip if marked as invalid (forbidden region has mask=True)
            if np.isinf(weighted_sum):
                continue

            # Skip if center is too close to the edge
            if (
                row < min_edge_distance
                or row >= mask_height - min_edge_distance
                or col < min_edge_distance
                or col >= mask_width - min_edge_distance
            ):
                continue

            # Check if this position overlaps with any already selected
            overlaps = False
            for sel_row, sel_col, _ in selected:
                # Two windows overlap if they're within window_size of each other
                if (
                    abs(row - sel_row) < window_size
                    and abs(col - sel_col) < window_size
                ):
                    overlaps = True
                    break

            # If no overlap, add to results
            if not overlaps:
                selected.append((row, col, weighted_sum))

                # Stop once we have enough
                if len(selected) >= n_top:
                    break

        best_y = [s[0] for s in selected]
        best_x = [s[1] for s in selected]

    if return_weighted:
        return best_x, best_y, dq_count
    return best_x, best_y
