from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from scipy.ndimage import convolve, uniform_filter, median_filter
from stdatamodels.jwst.datamodels import JwstDataModel


PSCALE_DICT = {
    "NRCBLONG": 0.063,
}
V2V3_REF_DICT = {
    "NRCBS_FULL": (-83.63, -495.98),
}


def filter_crop(img_crop):
    img_crop_clean = img_crop.copy()
    nan_crop_mask = np.isnan(img_crop)
    img_crop_clean[nan_crop_mask] = 0.0
    img_crop_clean[nan_crop_mask] = median_filter(img_crop_clean, size=5)[nan_crop_mask]
    return img_crop_clean


def _ensure_model(file):
    if not isinstance(file, JwstDataModel):
        return datamodels.open(file)
    else:
        return file


def apply_pointing(
    xoffset: float,
    yoffset: float,
    file: str | Path | JwstDataModel,
    coords: str = "detector",
):
    modle = _ensure_model(file)

    detector = model.meta.instrument.detector
    pscale = PSCALE_DICT[detector]

    aperture = model.meta.aperture.pps_name
    v2_ref, v3_ref = V2V3_REF_DICT[aperture]

    if coords == "detector":
        xref_pix, yref_pix = model.meta.wcs.transform(
            "v2v3", "detector", v2_ref, v3_ref
        )

        xoff_pix = xoffset / pscale
        yoff_pix = yoffset / pscale

        return xref_pix + xoff_pix, yref_pix + yoff_pix

    elif coords == "v2v3":
        v2_point = v2_ref - xoffset
        v3_point = v3_ref + yoffset

        return model.meta.wcs.transform("v2v3", "detector", v2_point, v3_point)
    else:
        raise ValueError(f"coords should be 'detector' or 'v2v3'. Got {coords}")


def calculate_offset(
    xpoint: float,
    ypoint: float,
    file: str | Path | JwstDataModel,
    coords: str = "detector",
):
    model = _ensure_model(file)

    detector = model.meta.instrument.detector
    pscale = PSCALE_DICT[detector]

    aperture = model.meta.aperture.pps_name
    v2_ref, v3_ref = V2V3_REF_DICT[aperture]

    if coords == "detector":
        xref_pix, yref_pix = model.meta.wcs.transform(
            "v2v3", "detector", v2_ref, v3_ref
        )

        xoff_pix = xpoint - xref_pix
        yoff_pix = ypoint - yref_pix

        xoff_arcsec = xoff_pix * pscale
        yoff_arcsec = yoff_pix * pscale
    elif coords == "v2v3":
        v2_point, v3_point = model.meta.wcs.transform(
            "detector", "v2v3", xpoint, ypoint
        )

        v2_off = v2_point - v2_ref
        v3_off = v3_point - v3_ref

        xoff_arcsec = -v2_off
        yoff_arcsec = v3_off
    else:
        raise ValueError(f"coords should be 'detector' or 'v2v3'. Got {coords}")

    return xoff_arcsec, yoff_arcsec


def _shift_with_fill(
    arr: np.ndarray, dx: int, dy: int, fill_value: float = np.inf
) -> np.ndarray:
    """Return `arr[row + dy, col + dx]` sampled on the original grid.

    :param arr: Array to resample
    :param dx: dx offset (along columns)
    :param dy: dy offset (along rows)
    :param fill_value: Value for out of bound pixels (defaults to np.inf)
    :return: Resampled array
    """
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
    joint_offsets: list[tuple[int, int]] | np.ndarray | None = None,
    min_edge_distance: int | None = None,
    return_weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find regions for optimal pointing

    Searches the ``n_top`` windows that minimize the number of bad pixels.
    Returns the optimal pointing position for each window.

    :param mask: Bad pixel mask
    :param window_size: Window size to optimize
    :param kernel: Kernel used to weigh bad pixels in the window.
                   Can be "uniform" or a custom array. Defaults to "uniform".
    :param n_top: Number of optimal pointings to return. 10 by default.
    :param forbidden_size: Size of the central region where no bad pixels are allowed.
                           Defaults to ``None``.
    :param joint_offsets: Offests to apply after the pointing to generate the window in pixels.
                           Useful to replicate dithers. Should be a list of (dx, dy) offsets
                           where y is along rows and x along columns.
                           Defaults to ``None``.
    :param min_edge_distance: Minimal distance to keep from the edge in pixels.
                              Defaults to ``None``.
    :param return_weighted: Whether the mask weighted by the kernel for all
                            explored windows should be returned.
                            Defaults to ``False``.
    :return: Arrays of x and y optimal pointing coordinate,
              along with the weighted mask if ``return_weighted`` is true.
    """
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
                    raise ValueError("Each joint offset must be a (dx, dy) tuple")
                normalized_offsets.append(tuple(map(int, offset)))

        all_offsets = [*normalized_offsets]
        shifted = [_shift_with_fill(dq_count, dx, dy) for dx, dy in all_offsets]
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


def do_region_search(
    dq_mask: np.ndarray,
    img: np.ndarray,
    region_size: int,
    psf: np.ndarray,
    n_top: int = 5,
    kernel: str | np.ndarray = "uniform",
    forbidden_size: int | None = None,
    joint_offsets: list[tuple[int, int]] | None = None,
    min_edge_distance: int | None = None,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform optimal region search and show some plots

    Thin wrapper around ``find_regions()``.

    :param mask: Bad pixel mask
    :param img: Full frame image used as reference
    :param region_size: Region size to optimize
    :param psf: PSF used for weighting and siplay
    :param n_top: Number of regions
    :param kernel: Kernel to weigh the bad pixels in the window. One of:
                       - "uniform": uniform on the region
                       - "weighted": weighted by a PSF centered on the region
                       - A custom array
    :param forbidden_size: Size of the central region where no bad pixels are allowed.
                           Defaults to ``None``.
    :param joint_offsets: Offests to apply after the pointing to generate the window in pixels.
                           Useful to replicate dithers. Should be a list of (dx, dy) offsets
                           where y is along rows and x along columns.
                           Defaults to ``None``.
    :param min_edge_distance: Minimal distance to keep from the edge in pixels.
                              Defaults to ``None``.
    :param show: Show the plots if True
    :return: The X and Y offsets
    """
    region_hs = region_size // 2

    if kernel == "weighted":
        kernel = psf + np.ones_like(psf)

    # Find the n_top best regions
    best_x, best_y, weighted_mask = find_regions(
        dq_mask,
        region_size,
        kernel=kernel,
        forbidden_size=forbidden_size,
        n_top=n_top,
        joint_offsets=joint_offsets,
        min_edge_distance=min_edge_distance,
        return_weighted=True,
    )

    # Plot the full frame DQ, weighted DQ and SCI frames with the best regions
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    axs[0].imshow(dq_mask)
    axs[1].imshow(weighted_mask, norm="symlog")
    axs[2].imshow(img, norm="symlog")
    axs[0].set_title("Bad pixel mask")
    axs[1].set_title("Weighted bad pixel mask")
    axs[2].set_title("Science image")
    fig.suptitle("Regions shown on the full detector frame")
    for i in range(n_top):
        axs[0].scatter(best_x[i], best_y[i], marker=f"${i + 1}$", color="r")
        axs[1].scatter(best_x[i], best_y[i], marker=f"${i + 1}$", color="r")
        axs[2].scatter(best_x[i], best_y[i], marker=f"${i + 1}$", color="r")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, n_top, figsize=(20, 5), sharex=True, sharey=True)
    for i in range(n_top):
        region_y, region_x = best_y[i], best_x[i]
        region = img[
            region_y - region_hs : region_y + region_hs,
            region_x - region_hs : region_x + region_hs,
        ]
        nan_count = np.sum(np.isnan(region))
        axs[0, i].imshow(region, norm="symlog")
        axs[0, i].set_title(f"Region {i + 1}: {nan_count} DQ")

        img_with_bad = psf.copy()
        region_mask = np.isnan(region)
        img_with_bad[region_mask] = np.nan
        axs[1, i].imshow(img_with_bad, norm="symlog")
    fig.suptitle("Regions shown on the science image and bad pixels overlaid on a PSF")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return best_x, best_y


def zoom_plot(
    img: np.ndarray,
    x: int,
    y: int,
    size: int,
    axs: np.ndarray[Axes] | None = None,
    psf: np.ndarray | None = None,
) -> np.ndarray:
    axs = axs if axs is not None else plt.gcf().axes
    hs = size // 2
    region = img[y - hs : y + hs, x - hs : x + hs]
    axs[0].imshow(region, norm="symlog")

    region_mask = np.isnan(region)
    if psf is not None:
        img_with_bad = psf.copy()
        img_with_bad[region_mask] = np.nan
        axs[1].imshow(img_with_bad, norm="symlog")
    else:
        axs[1].imshow(region_mask)
    return region_mask


def plot_dithers(img, xopt_all, yopt_all, size, psf=None):
    xopt_all = list(map(int, xopt_all))
    yopt_all = list(map(int, yopt_all))
    ndithers = len(xopt_all)
    fig_full = plt.figure()
    plt.imshow(img, norm="symlog")
    for i in range(ndithers):
        plt.scatter(xopt_all[i], yopt_all[i], marker=f"${i + 1}$", color="r")

    fig_zoom, axs = plt.subplots(2, ndithers, figsize=(10, 5))
    for i in range(ndithers):
        region_mask = zoom_plot(
            img, xopt_all[i], yopt_all[i], size, axs=axs[:, i], psf=psf
        )
        nbad = np.sum(region_mask)
        axs[0, i].set_title(f"Dither {i + 1}: {nbad} DQ")
    return fig_full, fig_zoom


def long_to_short(x, y, file_lw, file_sw):
    model_lw = _ensure_model(file_lw)
    model_sw = _ensure_model(file_sw)

    v2, v3 = model_lw.meta.wcs.transform("detector", "v2v3", x, y)
    x_sw, y_sw = model_sw.meta.wcs.transform("v2v3", "detector", v2, v3)

    return x_sw, y_sw
