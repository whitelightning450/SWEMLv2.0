import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pyproj import Transformer
import requests

HOME = os.getcwd()

STURM_NAMES = {
    1: 'tundra',
    2: 'boreal',
    3: 'maritime',
    4: 'ephemeral',
    5: 'prairie',
    6: 'montane',
    7: 'ice',
}

EXCLUDE_CLASSES = {4, 8, 9}  # 4 = ephemeral, 8 = ocean, 9 = fill
STURM_RES_M = 300  # approximate native resolution of Sturm raster (10 arcsec)

DEFAULT_BBOX = (-125.0, 30.0, -104.0, 49.5)  # (west, south, east, north) WGS84


def _get_sturm_file():
    url = (
        "https://daacdata.apps.nsidc.org/pub/DATASETS/"
        "nsidc0768_global_seasonal_snow_classification_v01/"
        "SnowClass_NA_300m_10.0arcsec_2021_v01.0.tif"
    )
    out_dir = os.path.join(HOME, "data", "SnowClassification")
    fname = "SnowClass_NA_300m_10.0arcsec_2021_v01.0.tif"
    fpath = os.path.join(out_dir, fname)
    if not os.path.exists(fpath):
        os.makedirs(out_dir, exist_ok=True)
        print("Downloading Sturm snow classification raster...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Sturm raster already present.")
    return fpath


def create_WUS_grid(output_res, bbox_wgs84=DEFAULT_BBOX):
    """
    Generate a persistent-snow WUS grid at output_res meters in EPSG:5070.

    Grid cells are aligned to the EPSG:5070 coordinate origin. Cells with
    Sturm classes 4 (ephemeral) 8 (ocean) and 9 (fill) are excluded.
    Future refinement may also exclude 5 (prairie) subject to an elevation or slope filter 

    Resampling from the ~300m Sturm raster:
        output_res <= 300m: nearest-neighbor
        output_res >  300m: mode

    Parameters
    ----------
    output_res : int
        Cell size in meters (e.g. 250, 500, 750).
    bbox_wgs84 : tuple
        (west, south, east, north) in WGS84 degrees.
        Default: CONUS west of 104W.

    Returns
    -------
    str
        Path to the saved parquet file.
    """
    sturm_path = _get_sturm_file()

    # Project bbox corners to EPSG:5070 and snap extent to grid
    t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    w, s, e, n = bbox_wgs84
    xs, ys = t_fwd.transform([w, e, e, w], [s, s, n, n])
    x_min = np.floor(min(xs) / output_res) * output_res
    x_max = np.ceil(max(xs) / output_res) * output_res
    y_min = np.floor(min(ys) / output_res) * output_res
    y_max = np.ceil(max(ys) / output_res) * output_res

    n_cols = round((x_max - x_min) / output_res)
    n_rows = round((y_max - y_min) / output_res)
    print(f"WUS grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols:,} cells at {output_res}m")

    dst_transform = from_origin(x_min, y_max, output_res, output_res)
    dst_crs = CRS.from_epsg(5070)

    resamp = Resampling.nearest if output_res <= STURM_RES_M else Resampling.mode
    print(f"Sturm resampling: {'nearest' if resamp == Resampling.nearest else 'mode'}")

    # Reproject Sturm raster onto WUS grid
    sturm_grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    with rasterio.open(sturm_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=sturm_grid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resamp,
            src_nodata=8,
            dst_nodata=8,
        )

    # Identify valid (persistent snow) cells
    valid_mask = ~np.isin(sturm_grid, sorted(EXCLUDE_CLASSES))
    valid_row, valid_col = np.where(valid_mask)
    n_valid = len(valid_row)
    pct = 100 * n_valid / (n_rows * n_cols)
    print(f"Valid snow cells: {n_valid:,} of {n_rows * n_cols:,} ({pct:.1f}%)")

    # Compute Albers centroids for valid cells only
    x_centers = x_min + (valid_col + 0.5) * output_res
    y_centers = y_max - (valid_row + 0.5) * output_res

    # Convert centroids to WGS84
    t_inv = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = t_inv.transform(x_centers, y_centers)
    lat = lat.round(6)
    lon = lon.round(6)

    sturm_vals = sturm_grid[valid_mask]

    df = pd.DataFrame({
        "cell_id": [f"{output_res}M_{la}_{lo}" for la, lo in zip(lat, lon)],
        "x_albers": x_centers,
        "y_albers": y_centers,
        "cen_lat": lat,
        "cen_lon": lon,
    })
    for code, name in STURM_NAMES.items():
        if code not in EXCLUDE_CLASSES:
            df[f"sturm_{name}"] = (sturm_vals == code).astype(np.int8)

    out_dir = os.path.join(HOME, "data", "PreProcessed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"WUS_grid_{output_res}M.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path, compression="BROTLI")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create WUS snow grid")
    parser.add_argument("output_res", type=int, help="Grid resolution in meters")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("W", "S", "E", "N"),
        default=list(DEFAULT_BBOX),
        help="Bounding box in WGS84 (default: CONUS west of 104W)",
    )
    args = parser.parse_args()
    create_WUS_grid(args.output_res, tuple(args.bbox))
