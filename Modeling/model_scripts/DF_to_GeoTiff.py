import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import numpy as np
import pandas as pd # Added pandas import as it's used for pd.to_numeric
from pathlib import Path
from shapely.geometry import Polygon, Point, LineString # For example data
import os
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import numpy as np
import pandas as pd # Added pandas import as it's used for pd.to_numeric
from pathlib import Path
from shapely.geometry import Polygon, Point, LineString # For example data
from tqdm.notebook import tqdm

def dataframe_to_geotiff(directory_path, Pred_col, value_column,
                             resolution_meters=None, fill_value=0):
    """
    Converts a GeoPandas GeoDataFrame to a GeoTIFF raster by rasterizing its geometries.

    Returns:
        Path: The path to the created GeoTIFF file, or None if an error occurred.
    """
    all_entries = os.listdir(directory_path)
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory_path, entry))]
    #print(files)
    for file in tqdm(files):
        df = pd.read_parquet(f"{directory_path}/{file}")
        cols = ['cen_lat', 'cen_lon', Pred_col]
        pred =  df[cols]
        pred.head()


        # Define output path
        output_dir = Path(f"{directory_path}/GeoTiffs")
        output_dir.mkdir(exist_ok=True)
        output_file = f"{output_dir}/{file.split('.')[0]}.tif"


        meters_per_degree_at_equator = 111320.0

        #convert to GeoDataFrame
        geometry = gpd.points_from_xy(pred['cen_lon'], pred['cen_lat'])
        gdf = gpd.GeoDataFrame(pred, geometry=geometry, crs="EPSG:4326") # Example CRS for lat/lon

        output_filepath = Path(output_file)

        if gdf.empty:
            print("Error: Input GeoDataFrame is empty. No data to rasterize.")
            return None
        if value_column not in gdf.columns:
            print(f"Error: Value column '{value_column}' not found in GeoDataFrame.")
            return None
        if gdf.crs is None:
            print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326 for conversion.")
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        # Ensure value column is numeric
        gdf[value_column] = pd.to_numeric(gdf[value_column], errors='coerce').fillna(fill_value)

        # Determine bounds of the GeoDataFrame
        min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

        # Determine the effective resolution in degrees
        resolution_to_use = None
        if resolution_meters is not None:
            # Calculate mean latitude for a more accurate conversion from meters to degrees
            # Handle case where gdf.geometry.centroid.y might be empty or single point
            if not gdf.geometry.empty:
                mean_lat = gdf.geometry.centroid.y.mean()
            else:
                mean_lat = (min_lat + max_lat) / 2 # Fallback to center of bounds

            mean_lat_radians = np.radians(mean_lat)

            # Approximate meters per degree latitude (at equator)
            meters_per_degree_at_equator = 111320.0

            # Convert meters to degrees. Using the latitude conversion as a general factor
            # for resolution, as it's less variable globally than longitude conversion.
            # Use a small epsilon to avoid division by zero if cos(mean_lat_radians) is near zero (poles)
            # However, for typical geographic data, this is not an issue.
            if np.cos(mean_lat_radians) == 0: # At poles, longitude resolution becomes infinite
                 resolution_to_use = resolution_meters / meters_per_degree_at_equator # Use latitude conversion
            else:
                 # This is a common approximation. For true square pixels in meters,
                 # a projected CRS is usually preferred.
                 resolution_to_use = resolution_meters / (meters_per_degree_at_equator * np.cos(mean_lat_radians))

           # print(f"Converted {resolution_meters}m resolution to approximately {resolution_to_use:.5f} degrees at mean latitude.")
        else:
            # Infer resolution based on bounds and a default number of pixels (original simplified logic)
            default_pixels_dim = 256 # Target dimension for the shorter side of the raster

            range_lon = max_lon - min_lon
            range_lat = max_lat - min_lat

            if range_lon == 0 and range_lat == 0: # Handle case of a single point
                resolution_to_use = 0.01 # A small default resolution for single points
            elif range_lon == 0: # Handle case of a vertical line
                resolution_to_use = range_lat / default_pixels_dim
            elif range_lat == 0: # Handle case of a horizontal line
                resolution_to_use = range_lon / default_pixels_dim
            else:
                resolution_to_use = max(range_lon / default_pixels_dim, range_lat / default_pixels_dim)

           # print(f"Inferred resolution (in degrees): {resolution_to_use}")

        # Calculate dimensions of the raster grid
        width = int(np.ceil((max_lon - min_lon) / resolution_to_use))
        height = int(np.ceil((max_lat - min_lat) / resolution_to_use))

        # Ensure minimum 1x1 pixel if bounds are single point
        if width == 0: width = 1
        if height == 0: height = 1

        # Define the georeferencing transform
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Create a generator of (geometry, value) pairs for rasterization
        # Ensure geometries are valid for rasterize
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[value_column]) if geom is not None and geom.is_valid)

        # Rasterize the geometries
        try:
            raster_array = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=fill_value, # Value for pixels not covered by any geometry
                dtype=np.float32,
                all_touched=False # Default to burning only pixel centers
            )
        except Exception as e:
            print(f"Error during rasterization: {e}")
            return None

        # Define metadata for the GeoTIFF
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,  # Number of bands
            'dtype': raster_array.dtype,
            'crs': gdf.crs, # Use the CRS from the GeoDataFrame
            'transform': transform,
            'nodata': fill_value # Set nodata value to the fill_value
        }

        # Write the array to a GeoTIFF file
        try:
            with rasterio.open(output_filepath, 'w', **profile) as dst:
                dst.write(raster_array, 1) # Write to band 1
            #print(f"Successfully created GeoTIFF at: {output_filepath}")
            #return output_filepath
        except Exception as e:
            print(f"Error writing GeoTIFF file: {e}")
            #return None