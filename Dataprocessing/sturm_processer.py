import os
import pandas as pd
import numpy as np
import geopandas as gpd
import sturm_processer as stpro
from tqdm import tqdm
import rasterio

# Function to sample Sturm data at given coordinates with progress bar
def sample_sturm_data(sturm_file, coords):
    with rasterio.open(sturm_file) as src:
        transform = src.transform
        sturm_data = src.read(1)
        
        values = []
        for lon, lat in tqdm(coords, desc="Sampling Sturm Data"):
            # Transform coordinates to the raster's coordinate reference system
            row, col = ~transform * (lon, lat)
            row, col = int(row), int(col)
            # Check if the indices are within the bounds of the raster
            if 0 <= row < sturm_data.shape[0] and 0 <= col < sturm_data.shape[1]:
                value = sturm_data[row, col]
                values.append(value)
            else:
                values.append(np.nan)  # Append NaN if coordinates are out of bounds
    return values

# Process Sturm data for each Parquet file
def process_sturm_data_for_files(input_directory, sturm_file, output_directory):
    # Get list of all Parquet files in the directory
    parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"No Parquet files found in {input_directory}")
        return
    
    with rasterio.open(sturm_file) as src:
        sturm_bbox = src.bounds
        print(f"Sturm file bounds: {sturm_bbox}")
    
    # Progress bar for overall processing
    for parquet_file in tqdm(parquet_files, desc="Processing Parquet Files"):
        input_file = os.path.join(input_directory, parquet_file)
        
        # Load Current Data
        current_df = pd.read_parquet(input_file)
        
        # Assuming 'cen_lat' and 'cen_lon' columns represent the center of each grid cell
        if 'cen_lat' not in current_df.columns or 'cen_lon' not in current_df.columns:
            print(f"'cen_lat' or 'cen_lon' columns missing in {parquet_file}")
            continue

        # Create GeoDataFrame with centroids
        current_gdf = gpd.GeoDataFrame(current_df, geometry=gpd.points_from_xy(current_df.cen_lon, current_df.cen_lat), crs="EPSG:4326")

        # Define centroid coordinates
        centroid_coords = [(point.x, point.y) for point in current_gdf.geometry]

        # Sample Sturm data at the centroid coordinates
        sturm_values = sample_sturm_data(sturm_file, centroid_coords)

        # Add the sampled values to the Current GeoDataFrame
        current_gdf['sturm_value'] = sturm_values

        # Define the output file path
        output_file = os.path.join(output_directory, f"Sturm_{parquet_file}")
        
        # Save the updated GeoDataFrame back to Parquet
        current_gdf.drop(columns='geometry').to_parquet(output_file)

        # Display a random selection of rows with the new sturm_value column
        print(current_gdf.sample(n=5))