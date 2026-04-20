from pathlib import Path

import pandas as pd
from astropy.io import fits
from astroquery.mast import Observations

from coldest import pointing


def get_2473_filename(pos_df, target):
    target_df = pos_df.query("target==@target")
    target_df_lw = target_df[target_df.file.str.endswith("long")].sort_values("file").reset_index()
    file_row = target_df_lw.iloc[0]
    file_id = file_row.file
    filename = f"{file_id}_cal.fits"
    return filename


def download_file(filename, data_dir):
    filepath = Path(filename)
    uri_lw = f"mast:JWST/product/{filepath.name}"
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    path = data_dir / filepath.name
    Observations.download_file(uri_lw, local_path=path)
    return path


def download_2473(filename, data_dir, pos_df):
    filepath = Path(filename)
    file_id = "_".join(filepath.stem.split("_")[:-1])
    file_row = pos_df.query("file==@file_id")
    x, y = file_row.x, file_row.y
    path_lw = download_file(filepath.name, data_dir)
    return path_lw, x.item(), y.item()


def download_other(filename, data_dir):
    path = download_file(filename, data_dir)
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        xoffset = hdr["XOFFSET"]
        yoffset = hdr["YOFFSET"]
    x_target, y_target = pointing.apply_pointing(xoffset, yoffset, path)
    return path, x_target, y_target
