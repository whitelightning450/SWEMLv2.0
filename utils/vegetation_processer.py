import os
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.warp import transform as warp_transform
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
                        f.write(chunk)
            print("File downloaded successfully!")
        except requests.exceptions.RequestException as e:
            print("Error downloading the file:", e)
    else:
        print('Landcover data already downloaded')

def unzip_LC_data(output_path, file):

    z = zipfile.ZipFile(f"{output_path}/{file}")
    z.extractall(output_path)
    print(f"All files successfully extracted to {output_path}")
        

# Function to sample Vegetation data at given coordinates with progress bar
def sample_vegetation_data(vegetation_file, coords):
    with rasterio.open(vegetation_file) as src:
        vegetation_data = src.read(1)
        
        # Get CRS of the raster
        raster_crs = src.crs

        values = []
        for lon, lat in tqdm(coords, desc="Sampling Vegetation Data"):
            try:
                # Transform coordinates to the raster's coordinate reference system
                transformed_lon, transformed_lat = warp_transform('EPSG:4326', raster_crs, [lon], [lat])
                row, col = src.index(transformed_lon[0], transformed_lat[0])
                # Print some debug information
                #print(f"Coordinates (lon, lat): ({lon}, {lat}) -> Transformed (lon, lat): ({transformed_lon[0]}, {transformed_lat[0]}) -> (row, col): ({row}, {col})")
                # Check if the indices are within the bounds of the raster
                if 0 <= row < vegetation_data.shape[0] and 0 <= col < vegetation_data.shape[1]:
                    value = vegetation_data[row, col]
                    #value = vegetation_data[col, row] #all veg values were the same, looks like row, col was incorrect, should be col, row

                    #print(f"Sampled value: {value} at (row, col): ({row}, {col})")
                    values.append(value)
                else:
                    print(f"Coordinates out of bounds for raster: (row, col): ({row}, {col})")
                    values.append(np.nan)  # Append NaN if coordinates are out of bounds
            except Exception as e:
                print(f"Error sampling coordinate (lon, lat): ({lon}, {lat}) -> with error: {e}")
                values.append(np.nan)
    return values

# Process Vegetation data for each Parquet file
def process_vegetation_data_for_files(input_directory, vegetation_file, output_directory):
    # Get list of all Parquet files in the directory
    parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"No Parquet files found in {input_directory}")
        return
    
    with rasterio.open(vegetation_file) as src:
        vegetation_bbox = src.bounds
        print(f"Vegetation file bounds: {vegetation_bbox}")
        print(f"Vegetation CRS: {src.crs}")
    
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

        # Sample Vegetation data at the centroid coordinates
        vegetation_values = sample_vegetation_data(vegetation_file, centroid_coords)

        # Add the sampled values to the Current GeoDataFrame
        current_gdf['vegetation_value'] = vegetation_values

        # Define the output file path
        output_file = os.path.join(output_directory, f"Vegetation_{parquet_file}")
        
        # Save the updated GeoDataFrame back to Parquet
        current_gdf.drop(columns='geometry').to_parquet(output_file)

        # Display the first few rows with the new vegetation_value column
        #print(current_gdf.head())
