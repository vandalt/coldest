import numpy as np
from scipy.ndimage import uniform_filter, convolve


def _shift_with_fill(
    arr: np.ndarray, dy: int, dx: int, fill_value: float = np.inf
) -> np.ndarray:
    """Return `arr[row + dy, col + dx]` sampled on the original grid."""
    height, width = arr.shape
    shifted = np.full((height, width), fill_value, dtype=float)

    out_row_start = max(0, -dy)
    out_row_end = min(height, height - dy)
    out_col_start = max(0, -dx)
    out_col_end = min(width, width - dx)

    if out_row_start >= out_row_end or out_col_start >= out_col_end:
        return shifted

    src_row_start = out_row_start + dy
    src_row_end = out_row_end + dy
    src_col_start = out_col_start + dx
    src_col_end = out_col_end + dx

    shifted[out_row_start:out_row_end, out_col_start:out_col_end] = arr[
        src_row_start:src_row_end, src_col_start:src_col_end
    ]
    return shifted


def find_regions(
    mask: np.ndarray,
    window_size: int,
    kernel: str | np.ndarray = "uniform",
    n_top: int = 10,
    forbidden_size: int | None = None,
    separation: tuple[int, int] | None = None,
    joint_offsets: list[tuple[int, int]] | np.ndarray | None = None,
    min_edge_distance: int | None = None,
    return_weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    forbidden_invalid: np.ndarray | None = None

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

        forbidden_invalid = forbidden_check > 0

        # Mark regions with any forbidden pixels as invalid (set to inf)
        dq_count = np.where(forbidden_invalid, np.inf, dq_count)

    if separation is not None and joint_offsets is not None:
        raise ValueError("Use either separation or joint_offsets, not both")

    if separation is not None:
        joint_offsets = [tuple(int(v) for v in separation)]

    if joint_offsets is not None:
        if isinstance(joint_offsets, np.ndarray):
            if joint_offsets.ndim != 2 or joint_offsets.shape[1] != 2:
                raise ValueError(
                    "joint_offsets numpy array must have shape (n_offsets, 2)"
                )
            normalized_offsets = [tuple(map(int, offset)) for offset in joint_offsets]
        else:
            normalized_offsets = []
            for offset in joint_offsets:
                if len(offset) != 2:
                    raise ValueError("Each joint offset must be a (dy, dx) tuple")
                normalized_offsets.append(tuple(map(int, offset)))

        all_offsets = [*normalized_offsets]
        shifted = [_shift_with_fill(dq_count, dy, dx) for dy, dx in all_offsets]
        dq_count = np.sum(np.stack(shifted, axis=0), axis=0)

        if forbidden_invalid is not None:
            forbidden_float = forbidden_invalid.astype(float)
            shifted_forbidden = [
                _shift_with_fill(forbidden_float, dy, dx, fill_value=0.0)
                for dy, dx in all_offsets
            ]
            joint_forbidden = np.any(np.stack(shifted_forbidden, axis=0) > 0, axis=0)
            dq_count = np.where(joint_forbidden, np.inf, dq_count)

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

        best_y = np.array([s[0] for s in selected])
        best_x = np.array([s[1] for s in selected])

    if return_weighted:
        return best_x, best_y, dq_count
    return best_x, best_y
