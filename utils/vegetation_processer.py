import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import rasterio
import rasterio.windows
from rasterio.warp import transform as warp_transform
from rasterio.transform import rowcol
import requests
import zipfile

#set home directory
HOME = os.getcwd()

def get_data(url, output_path, file):

    if not os.path.exists(f"{output_path}/{file}"):
        os.makedirs(output_path, exist_ok=True)
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(f"{output_path}/{file}", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("File downloaded successfully!")
        except requests.exceptions.RequestException as e:
            print("Error downloading the file:", e)
    else:
        print('Landcover data already downloaded')

def unzip_LC_data(output_path, file):
    with zipfile.ZipFile(f"{output_path}/{file}") as z:
        z.extractall(output_path)
    print(f"All files successfully extracted to {output_path}")


def _resolve_crs(src, fallback_crs):
    """Return the raster CRS as a WKT string, using fallback_crs if needed."""
    raster_crs = src.crs
    if raster_crs is None:
        if fallback_crs is None:
            raise ValueError(
                "Vegetation raster has no embedded CRS. "
                "Pass fallback_crs (e.g. 'EPSG:5070' for NLCD) to proceed."
            )
        print(f"Warning: raster has no embedded CRS, using fallback: {fallback_crs}")
        return fallback_crs
    # Convert to WKT to avoid EPSG lookup failures for non-standard
    # projection codes (e.g. ESRI:102001 used by NALCMS rasters)
    return raster_crs.to_wkt()


def _sample_window(src, rows, cols):
    """Read the minimal raster window covering the given pixel indices and
    return sampled values (NaN for out-of-bounds points)."""
    valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

    if not valid.any():
        return pd.array([pd.NA] * len(rows), dtype='Int16')

    row_min = int(rows[valid].min())
    row_max = int(rows[valid].max())
    col_min = int(cols[valid].min())
    col_max = int(cols[valid].max())

    window = rasterio.windows.Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min + 1,
        height=row_max - row_min + 1,
    )
    window_data = src.read(1, window=window)

    # Indices relative to the window origin
    r_rel = np.clip(rows - row_min, 0, window_data.shape[0] - 1)
    c_rel = np.clip(cols - col_min, 0, window_data.shape[1] - 1)

    values = pd.array(window_data[r_rel, c_rel], dtype='Int16')
    values[~valid] = pd.NA
    return values


# Function to sample Vegetation data at given coordinates — vectorized
def sample_vegetation_data(vegetation_file, coords, fallback_crs=None):
    with rasterio.open(vegetation_file) as src:
        raster_crs = _resolve_crs(src, fallback_crs)

        if not coords:
            return []

        lons, lats = zip(*coords)
        xs, ys = warp_transform('EPSG:4326', raster_crs, list(lons), list(lats))
        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)

        values = _sample_window(src, rows, cols)

    return values.tolist()


# Process Vegetation data for each Parquet file
def process_vegetation_data_for_files(input_directory, vegetation_file, output_directory, fallback_crs=None):
    parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]

    if not parquet_files:
        print(f"No Parquet files found in {input_directory}")
        return

    os.makedirs(output_directory, exist_ok=True)

    print('Adding Vegetation criteria to ASO grids')
    with rasterio.open(vegetation_file) as src:
        raster_crs = _resolve_crs(src, fallback_crs)

        for parquet_file in tqdm(parquet_files):
            input_file = os.path.join(input_directory, parquet_file)
            current_df = pd.read_parquet(input_file)

            if 'cen_lat' not in current_df.columns or 'cen_lon' not in current_df.columns:
                print(f"'cen_lat' or 'cen_lon' columns missing in {parquet_file}")
                continue

            # Batch coordinate transform — one call for all points
            lons = current_df['cen_lon'].tolist()
            lats = current_df['cen_lat'].tolist()
            xs, ys = warp_transform('EPSG:4326', raster_crs, lons, lats)

            rows, cols = rowcol(src.transform, xs, ys)
            rows = np.array(rows, dtype=np.int64)
            cols = np.array(cols, dtype=np.int64)

            # Read only the window of the raster covering these points
            current_df['vegetation_value'] = _sample_window(src, rows, cols)

            output_file = os.path.join(output_directory, f"Vegetation_{parquet_file}")
            current_df.to_parquet(output_file)

    print(f"Job complete, vegetation data saved to {output_directory}")
