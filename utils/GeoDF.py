# Import packages
# Dataframe Packages
import numpy as np
from numpy import gradient, rad2deg, arctan2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Vector Packages
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import Transformer

# Raster Packages
import rioxarray as rxr

# Data Access Packages
import pystac_client
import planetary_computer

# General Packages
import os
import math
from tqdm.auto import tqdm
import concurrent.futures as cf

#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
import pickle as pkl
'''
To create .netrc file:
import earthaccess
earthaccess.login(persist=True)
'''

#load access key
HOME = os.getcwd()
KEYPATH = "utils/AWSaccessKeys.csv"


if os.path.isfile(f"{HOME}/{KEYPATH}") == True:
    ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

    #start session
    SESSION = boto3.Session(
        aws_access_key_id=ACCESS['Access key ID'][0],
        aws_secret_access_key=ACCESS['Secret access key'][0],
    )
    S3 = SESSION.resource('s3')
    #AWS BUCKET information
    BUCKET_NAME = 'national-snow-model'
    #S3 = boto3.resource('S3', config=Config(signature_version=UNSIGNED))
    BUCKET = S3.Bucket(BUCKET_NAME)
else:
    print("no AWS credentials present, skipping")
    
#set multiprocessing limits
CPUS = len(os.sched_getaffinity(0))
CPUS = max(1, int((CPUS/2)-2))
    

def row_snotel(row, distance_cache, nearest_snotel, snotel_gdf, n):
    cell_id = row.name
        # Check if distances for this cell_id are already calculated and cached
    if cell_id in distance_cache:
        nearest_snotel[cell_id] = distance_cache[cell_id]
    else:
        # Calculate Haversine distances between the grid cell and all SNOTEL locations
        distances = haversine_vectorized(
            row.geometry.y, row.geometry.x,
            snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)

        # Store the nearest stations in the cache
        nearest_snotel[cell_id] = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
        distance_cache[cell_id] = nearest_snotel[cell_id]



# Calculating nearest SNOTEL sites, n = the number of snotel sites
def calculate_nearest_snotel(region, aso_gdf, snotel_gdf,output_res, n=6, distance_cache=None):

    #nearest_snotel_dict_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution"
    nearest_snotel_dict_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"

    if distance_cache is None:
        distance_cache = {}

    nearest_snotel = {}
    print(f"Calculating haversine distance for {len(aso_gdf)} locations to in situ OBS, and saving cell-obs relationships in dictionary")
    #tqdm_notebook.pandas()
    tqdm.pandas()
    aso_gdf.progress_apply(lambda row: row_snotel(row, distance_cache, nearest_snotel, snotel_gdf,n), axis = 1) #try function to see if its working

    #saving nearest snotel file
    print(f"Saving nearest SNOTEL in {region} for each cell id in a pkl file")        
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'wb') as handle:
        pkl.dump(nearest_snotel, handle, protocol=pkl.HIGHEST_PROTOCOL)


def haversine_vectorized(lat1, lon1, lat2, lon2):
    
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371.0
    # Distance calculation
    distances = r * c

    return distances

def calculate_distances_for_cell(aso_row, snotel_gdf, n=6):
   
    distances = haversine_vectorized(
        aso_row.geometry.y, aso_row.geometry.x,
        snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)
    
    ns_s = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
    
    return ns_s

def create_polygon(row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])

def fetch_snotel_sites_for_cellids(region, output_res):  
    #relative file paths
    # aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/ASO/{region}/{output_res}M_SWE_parquet"
    # snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    
    aso_swe_files_folder_path = f"{HOME}/data/ASO/{region}/{output_res}M_SWE_parquet"
    snotel_path = f"{HOME}/data/SNOTEL_Data/"
    
    #load snotel geospatial metadata
    snotel_file = gpd.read_file('https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson').set_index('code')
    snotel_file = snotel_file[snotel_file['csvData']==True]
    snotel_file.reset_index(inplace = True, drop = False)
    snotel_file.rename(columns={'code':'station_id'}, inplace = True)
    
    #select only snotel sites with data throughout the modeling period (e.g., 2013-2025)
    GroundMeasures = pd.read_parquet(f"{HOME}/data/SNOTEL_Data/ground_measures_dp.parquet")
    mask = snotel_file['station_id'].isin(GroundMeasures.columns.tolist())
    snotel_file = snotel_file[mask]

    #add new prediction location here at this step -
    #will need to make grid for RegionVal.pkl.
    #build in method for adding to existing dictionary rather than rerunning for entire region...
    print('Loading all Geospatial prediction/observation files and concatenating into one dataframe')
    frames = []
    for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):
        try:
            frames.append(pd.read_parquet(os.path.join(aso_swe_files_folder_path, aso_swe_file)))
        except Exception as e:
            print(f"OSError: Corrupt brotli compressed data for {aso_swe_file}, skipping ({e})")
    ASO_meta_loc_DF = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    
    print('Identifying unique sites to create geophysical information dataframe') 
    ASO_meta_loc_DF.drop_duplicates(subset=['cell_id'], inplace=True)
    ASO_meta_loc_DF.set_index('cell_id', inplace=True)
    #ASO_meta_loc_DF.to_csv(f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/ASO_meta.parquet")
    #Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(ASO_meta_loc_DF)
    # Parquet with Brotli compression
   # metapath =  f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution"
    metapath =  f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"

    if not os.path.exists(metapath):
        os.makedirs(metapath, exist_ok=True)
    pq.write_table(table,f"{metapath}/ASO_meta.parquet", compression='BROTLI')


    print('converting to geodataframe')
    aso_geometry = [Point(xy) for xy in zip(ASO_meta_loc_DF['cen_lon'], ASO_meta_loc_DF['cen_lat'])]
    aso_gdf = gpd.GeoDataFrame(ASO_meta_loc_DF, geometry=aso_geometry, crs='EPSG:4326')

    snotel_geometry = [Point(xy) for xy in zip(snotel_file['longitude'], snotel_file['latitude'])]
    snotel_gdf = gpd.GeoDataFrame(snotel_file, geometry=snotel_geometry, crs='EPSG:4326')

    # Calculating nearest SNOTEL sites
    calculate_nearest_snotel(region,aso_gdf, snotel_gdf,output_res, n=6)


def GeoSpatial(region, output_res):
    print(f"Loading geospatial data for {region}")
    # ASO_meta_loc_DF = pd.read_parquet(f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/ASO_meta.parquet")
    ASO_meta_loc_DF = pd.read_parquet(f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/ASO_meta.parquet")

    cols = ['cen_lat', 'cen_lon']
    ASO_meta_loc_DF = ASO_meta_loc_DF[cols]

    print(f"Converting to geodataframe")
    aso_geometry = [Point(xy) for xy in zip(ASO_meta_loc_DF['cen_lon'], ASO_meta_loc_DF['cen_lat'])]
    aso_gdf = gpd.GeoDataFrame(ASO_meta_loc_DF, geometry=aso_geometry, crs='EPSG:4326')

    return aso_gdf


#Processing using gdal
def process_single_location(args):
    cell_id, lat, lon, DEMs, tiles = args
    try:
        tile_id = f"Copernicus_DSM_COG_30_N{math.floor(lat)}_00_W{math.ceil(abs(lon))}_00_DEM"
        index_id = DEMs.loc[tile_id]['sliceID']
        signed_asset = planetary_computer.sign(tiles[int(index_id)].assets["data"])
        elevation = rxr.open_rasterio(signed_asset.href)

        tilearray = np.around(elevation.values[0]).astype(float)
        grad_y, grad_x = gradient(tilearray)
        slope_arr  = np.sqrt(grad_x**2 + grad_y**2)
        aspect_arr = (-rad2deg(arctan2(-grad_y, grad_x)) + 270) % 360

        transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
        xx, yy = transformer.transform(lon, lat)

        x_idx = np.argmin(np.abs(elevation.x.values - xx))
        y_idx = np.argmin(np.abs(elevation.y.values - yy))

        elev  = round(float(tilearray[y_idx, x_idx]))
        slop  = round(float(slope_arr[y_idx, x_idx]))
        asp   = round(float(aspect_arr[y_idx, x_idx]))
        asp_w = round(float(-np.sin(aspect_arr[y_idx, x_idx] * np.pi / 180)), 3)
        asp_n = round(float( np.cos(aspect_arr[y_idx, x_idx] * np.pi / 180)), 3)
    except Exception as e:
        elev, slop, asp, asp_w, asp_n = np.nan, np.nan, np.nan, np.nan, np.nan
        print(f"{cell_id} does not have copernicus DEM data, manual input ({e})")

    return cell_id, elev, slop, asp, asp_w, asp_n


def _process_tile(args):
    """Fetch one DEM tile and extract terrain values for all cells within it."""
    tile_id, group_df, DEMs, tiles = args
    results = []
    try:
        index_id = DEMs.loc[tile_id]['sliceID']
        signed_asset = planetary_computer.sign(tiles[int(index_id)].assets["data"])
        elevation = rxr.open_rasterio(signed_asset.href)
        try:
            # Compute slope/aspect once for the whole tile
            tilearray  = np.around(elevation.values[0]).astype(float)
            grad_y, grad_x = gradient(tilearray)
            slope_arr  = np.sqrt(grad_x**2 + grad_y**2)
            aspect_arr = (-rad2deg(arctan2(-grad_y, grad_x)) + 270) % 360

            transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)

            for _, row in group_df.iterrows():
                xx, yy = transformer.transform(row['cen_lon'], row['cen_lat'])
                x_idx = np.argmin(np.abs(elevation.x.values - xx))
                y_idx = np.argmin(np.abs(elevation.y.values - yy))

                elev  = round(float(tilearray[y_idx, x_idx]))
                slop  = round(float(slope_arr[y_idx, x_idx]))
                asp   = round(float(aspect_arr[y_idx, x_idx]))
                asp_w = round(float(-np.sin(aspect_arr[y_idx, x_idx] * np.pi / 180)), 3)
                asp_n = round(float( np.cos(aspect_arr[y_idx, x_idx] * np.pi / 180)), 3)
                results.append((row['cell_id'], elev, slop, asp, asp_w, asp_n))
        finally:
            elevation.close()

    except Exception as e:
        print(f"Error for tile {tile_id}: {e}")
        for _, row in group_df.iterrows():
            results.append((row['cell_id'], np.nan, np.nan, np.nan, np.nan, np.nan))

    return results


def extract_terrain_data_threaded(metadata_df, region, output_res):
    metadata_df = metadata_df.reset_index()
    bounding_box = metadata_df.geometry.total_bounds
    min_x, min_y = math.floor(bounding_box[0])-0.5, math.floor(bounding_box[1])-0.5
    max_x, max_y = math.ceil(bounding_box[2])+0.5,  math.ceil(bounding_box[3])+0.5

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        ignore_conformance=True,
    )
    search = client.search(
        collections=["cop-dem-glo-90"],
        intersects={"type": "Polygon", "coordinates": [[
            [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]
        ]]},
    )
    tiles = list(search.items())
    DEMs = pd.DataFrame([[i, tiles[i].id] for i in range(len(tiles))],
                        columns=['sliceID', 'tileID']).set_index('tileID')
    print(f"There are {len(DEMs)} DEM tiles in the region")

    # Assign each cell to its DEM tile
    metadata_df['tile_id'] = metadata_df.apply(
        lambda row: f"Copernicus_DSM_COG_30_N{math.floor(row['cen_lat'])}_00_W{math.ceil(abs(row['cen_lon']))}_00_DEM",
        axis=1,
    )
    tile_groups = [
        (tid, grp.copy(), DEMs, tiles)
        for tid, grp in metadata_df.groupby('tile_id')
    ]
    print(f"Processing {len(tile_groups)} unique DEM tiles for {len(metadata_df)} grid cells")

    all_results = []
    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_process_tile, tg): tg[0] for tg in tile_groups}
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="DEM tiles"):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"Worker error ({futures[future]}): {e}")

    meta = pd.DataFrame(all_results, columns=['cell_id', 'Elevation_m', 'Slope_Deg', 'Aspect_Deg', 'Aspect_W', 'Aspect_N'])
    meta.set_index('cell_id', inplace=True)
    metadata_df.set_index('cell_id', inplace=True)
    metadata_df = pd.concat([metadata_df.drop(columns='tile_id'), meta], axis=1)

    dfpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"
    os.makedirs(dfpath, exist_ok=True)
    metadata_df_out = metadata_df.drop(columns='geometry')
    pq.write_table(pa.Table.from_pandas(metadata_df_out),
                   f"{dfpath}/{region}_metadata.parquet", compression='BROTLI')

    return metadata_df_out, DEMs, tiles


def add_geospatial_threaded(region, output_res):
    # Processed ASO observations folder with snotel measurements
    # TrainingDFpath = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution"
    TrainingDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"
    
    
    GeoObsdfs = f"{TrainingDFpath}/GeoObsDFs"

    #Make directory
    if not os.path.exists(GeoObsdfs):
        os.makedirs(GeoObsdfs, exist_ok=True)

    #Get Geospatial meta data
    print(f"Loading geospatial metadata for grids in {region}")
    aso_gdf = pd.read_parquet(f"{TrainingDFpath}/{region}_metadata.parquet").reset_index()

    #create dataframe
    print(f"Loading all available processed ASO observations for {region} at {output_res}M resolution")
    aso_swe_files = [filename for filename in os.listdir(f"{TrainingDFpath}/Obsdf")]
    
    print(f"Concatenating {len(aso_swe_files)} with geospatial data...")
    with cf.ProcessPoolExecutor(max_workers=CPUS) as executor:
        futures = {
            executor.submit(add_geospatial_single, (f"{TrainingDFpath}/Obsdf", f, aso_gdf, GeoObsdfs)): f
            for f in aso_swe_files
        }
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error ({futures[future]}): {e}")
    print(f"Job complete for connecting obs with geospatial data, the files can be found in {GeoObsdfs}")


def add_geospatial_single(args):

    aso_swe_path, aso_swe_file, aso_gdf, GeoObsdfs = args

    ObsDF = pd.read_parquet(f"{aso_swe_path}/{aso_swe_file}")

    #combine df with geospatial meta data
    final_df = pd.merge(ObsDF, aso_gdf, on = 'cell_id', how = 'left')
    cols = [
        'cell_id', 'Date',  'cen_lat', 'cen_lon', 'Elevation_m', 'Slope_Deg',
        'Aspect_Deg', 'Aspect_W', 'Aspect_N', 'swe_m', 'ns_1', 'ns_2', 'ns_3', 'ns_4',
        'ns_5', 'ns_6'
        ]
    final_df = final_df[cols]

    #Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(final_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{GeoObsdfs}/Geo{aso_swe_file.split('.')[0]}.parquet", compression='BROTLI')