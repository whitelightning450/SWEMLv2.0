import io
import os
import time as time_module
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import earthaccess
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

HOME = os.getcwd()

CPUS = len(os.sched_getaffinity(0))
CPUS = int((CPUS / 2) - 2)

THREDDS_BASE = (
    'http://thredds.northwestknowledge.net:8080'
    '/thredds/dodsC/MET/{var}/{var}_{year}.nc'
)

NLDAS_BASE = 'https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0'
NLDAS_VARS = ['Rainf', 'Tair', 'Qair', 'PSurf', 'SWdown']


# shared utilities

def _collect_centroids(df_dir):
    """Return DataFrame of unique (cell_id, cen_lat, cen_lon) across all parquets in df_dir."""
    files = [f for f in os.listdir(df_dir) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No training DF parquets in {df_dir}")
    parts = [
        pd.read_parquet(os.path.join(df_dir, f), columns=['cell_id', 'cen_lat', 'cen_lon'])
        for f in files
    ]
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset='cell_id').reset_index(drop=True)


def _bbox_from_centroids(centroids, pad=0.1):
    """Return (left, right, bottom, top) bounding box with padding."""
    return (
        centroids['cen_lon'].min() - pad,
        centroids['cen_lon'].max() + pad,
        centroids['cen_lat'].min() - pad,
        centroids['cen_lat'].max() + pad,
    )


def _precip_phase(precip, T_C, RH):
    """
    Jennings et al. (2018) binary phase partition.
    Returns (snow, rain) as float32 arrays in same units as precip.
    """
    P_snow = 1.0 / (1.0 + np.exp(-10.04 + 1.41 * T_C + 0.09 * RH))
    snow = np.where(P_snow >= 0.5, precip, 0.0).astype(np.float32)
    rain = np.where(P_snow < 0.5,  precip, 0.0).astype(np.float32)
    return snow, rain


def _rh_from_q_p(q, P_hPa, T_C):
    """Relative humidity (%) from specific humidity, pressure (hPa), and temperature (C)."""
    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    e  = q * P_hPa / (0.622 + 0.378 * q)
    return np.clip(e / es * 100.0, 0.0, 100.0)


def _rh_from_dew(T_C, Td_C):
    """Relative humidity (%) from temperature and dew-point temperature (Magnus formula)."""
    es = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    ed = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(ed / es * 100.0, 0.0, 100.0)


# gridMET

_GRIDMET_FETCH_VARS = ['pr', 'tmmx', 'tmmn', 'rmax', 'rmin', 'srad']


def _open_gridmet_var(var, year, lat_slice, lon_slice):
    """Open one gridMET variable for one calendar year from THREDDS OPeNDAP."""
    url = THREDDS_BASE.format(var=var, year=year)
    ds_v = xr.open_dataset(url, engine='netcdf4')
    lat_dim = 'latitude' if 'latitude' in ds_v.dims else 'lat'
    spatial = [v for v in ds_v.data_vars if lat_dim in ds_v[v].dims]
    if not spatial:
        raise KeyError(f"No spatial variable found in {var} CY{year}")
    da = ds_v[spatial[0]]
    rename = {}
    if 'latitude'  in da.dims:
        rename['latitude']  = 'lat'
    if 'longitude' in da.dims:
        rename['longitude'] = 'lon'
    if 'day'       in da.dims:
        rename['day']       = 'time'
    if rename:
        da = da.rename(rename)
    lat_vals = da.lat.values
    lat_s = lat_slice if lat_vals[0] < lat_vals[-1] else slice(lat_slice.stop, lat_slice.start)
    return da.sel(lat=lat_s, lon=lon_slice)


def add_gridmet_df(WY, output_res, threshold):
    """
    Stream gridMET daily forcings from THREDDS OPeNDAP, extract at ASO centroid points,
    apply Jennings (2018) binary phase partition, and add season-to-date met features
    to each training DF. No intermediate zarr is written.

    Reads from: Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh
    Writes to:  gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh

    New columns:
        gridMET_precip  - season-to-date total precipitation (mm)
        gridMET_snow    - season-to-date snow via Jennings (2018) (mm)
        gridMET_rain    - season-to-date rain via Jennings (2018) (mm)
        gridMET_APDD    - accumulated positive degree-days since Oct 1 (deg C days)
        gridMET_T7d     - 7-day mean air temperature ending on obs date (deg C)
        gridMET_T14d    - 14-day mean air temperature ending on obs date (deg C)
        gridMET_SW7d    - 7-day mean shortwave radiation ending on obs date (W/m^2)
        gridMET_SW14d   - 14-day mean shortwave radiation ending on obs date (W/m^2)
    """
    training_df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    os.makedirs(df_path, exist_ok=True)

    WY_start = f"{WY-1}-10-01"
    WY_end   = f"{WY}-09-30"

    centroids = _collect_centroids(training_df_path)
    left, right, bottom, top = _bbox_from_centroids(centroids)
    print(f"gridMET WY{WY}: {len(centroids)} unique centroids, "
          f"bbox lon [{left:.2f}, {right:.2f}] lat [{bottom:.2f}, {top:.2f}]")

    lat_slice = slice(bottom, top)
    lon_slice = slice(left, right)

    parts = []
    for cy in (WY - 1, WY):
        print(f"  Loading gridMET CY{cy}...")
        var_das = {var: _open_gridmet_var(var, cy, lat_slice, lon_slice)
                   for var in _GRIDMET_FETCH_VARS}
        parts.append(xr.Dataset(var_das))

    ds_bbox = xr.concat(parts, dim='time').sel(time=slice(WY_start, WY_end)).load()
    print("  Extracting at centroid points...")

    # vectorized nearest-neighbor extraction at all centroid points
    lats_da = xr.DataArray(centroids['cen_lat'].values, dims='cell_id')
    lons_da = xr.DataArray(centroids['cen_lon'].values, dims='cell_id')

    tmean_k = ((ds_bbox['tmmx'] + ds_bbox['tmmn']) / 2.0).sel(
        lat=lats_da, lon=lons_da, method='nearest').values.astype(np.float32)
    rmean   = np.clip(((ds_bbox['rmax'] + ds_bbox['rmin']) / 2.0).sel(
        lat=lats_da, lon=lons_da, method='nearest').values, 0.0, 100.0).astype(np.float32)
    pr      = ds_bbox['pr'].sel(
        lat=lats_da, lon=lons_da, method='nearest').values.astype(np.float32)
    srad    = ds_bbox['srad'].sel(
        lat=lats_da, lon=lons_da, method='nearest').values.astype(np.float32)
    time_idx = ds_bbox.time.values
    del ds_bbox

    T_C = (tmean_k - 273.15).astype(np.float32)
    snow, rain = _precip_phase(pr, T_C, rmean)

    # in-memory (time, cell_id) dataset for per-file lookup
    ds_all = xr.Dataset(
        {
            'pr':    xr.DataArray(pr,      dims=('time', 'cell_id')),
            'tmean': xr.DataArray(tmean_k, dims=('time', 'cell_id')),
            'srad':  xr.DataArray(srad,    dims=('time', 'cell_id')),
            'SNOW':  xr.DataArray(snow,    dims=('time', 'cell_id')),
            'RAIN':  xr.DataArray(rain,    dims=('time', 'cell_id')),
        },
        coords={'time': time_idx, 'cell_id': centroids['cell_id'].values},
    )

    training_dfs = [f for f in os.listdir(training_df_path) if f.endswith('.parquet')]

    for geofile in tqdm(training_dfs, desc=f"Adding gridMET features WY{WY}"):
        date    = geofile.split('_')[-1].split('.parquet')[0]
        region  = geofile.split('_')[-2]
        strdate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        print(f"  gridMET -> {region} {strdate}")

        GDF      = pd.read_parquet(os.path.join(training_df_path, geofile))
        df_cells = GDF['cell_id'].values
        GDF.set_index('cell_id', inplace=True)

        ds_sl  = ds_all.sel(time=slice(WY_start, strdate), cell_id=df_cells)
        T_C_sl = ds_sl['tmean'].values - 273.15
        SW_sl  = ds_sl['srad'].values

        GDF['gridMET_precip'] = np.round(ds_sl['pr'].values.sum(axis=0),    2)
        GDF['gridMET_snow']   = np.round(ds_sl['SNOW'].values.sum(axis=0),  2)
        GDF['gridMET_rain']   = np.round(ds_sl['RAIN'].values.sum(axis=0),  2)
        GDF['gridMET_APDD']   = np.round(np.maximum(T_C_sl, 0).sum(axis=0), 2)
        GDF['gridMET_T7d']    = np.round(T_C_sl[-7:].mean(axis=0),          2)
        GDF['gridMET_T14d']   = np.round(T_C_sl[-14:].mean(axis=0),         2)
        GDF['gridMET_SW7d']   = np.round(SW_sl[-7:].mean(axis=0),           2)
        GDF['gridMET_SW14d']  = np.round(SW_sl[-14:].mean(axis=0),          2)

        GDF.reset_index(inplace=True)
        table = pa.Table.from_pandas(GDF)
        pq.write_table(table, f"{df_path}/gridMET_{geofile}", compression='BROTLI')


# NLDAS

def _nldas_url(dt):
    return (
        f"{NLDAS_BASE}/{dt.year}/{dt.day_of_year:03d}"
        f"/NLDAS_FORA0125_H.A{dt.strftime('%Y%m%d')}.{dt.hour:02d}00.020.nc"
    )


def _fetch_nldas_file(args):
    session, url = args
    for attempt in range(5):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            return io.BytesIO(r.content)
        except Exception:
            if attempt < 4:
                time_module.sleep(2 ** attempt)
    return None


def _load_nldas_month(session, hours, n_workers=8):
    """
    Download hourly NLDAS files for one month in parallel.
    Missing hours are filled with NaN rather than skipping the month.
    Returns an xr.Dataset with NLDAS_VARS, loaded into memory.
    """
    urls = [_nldas_url(dt) for dt in hours]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        buffers = list(ex.map(_fetch_nldas_file, [(session, u) for u in urls]))

    n_failed = sum(b is None for b in buffers)
    if n_failed == len(buffers):
        raise RuntimeError("All hourly NLDAS files failed after retries")
    if n_failed > 0:
        print(f"    {n_failed}/{len(buffers)} files failed; filling with NaN")

    first_good = next(i for i, b in enumerate(buffers) if b is not None)
    template   = xr.open_dataset(buffers[first_good], engine='h5netcdf')[NLDAS_VARS].load()

    datasets = []
    for i, (buf, dt) in enumerate(zip(buffers, hours)):
        if i == first_good:
            datasets.append(template)
        elif buf is not None:
            datasets.append(xr.open_dataset(buf, engine='h5netcdf')[NLDAS_VARS].load())
        else:
            nan_ds = xr.Dataset({
                v: xr.DataArray(
                    np.full(template[v].squeeze().shape, np.nan, dtype=np.float32),
                    dims=[d for d in template[v].dims if d != 'time'],
                    coords={d: template[d] for d in template[v].dims if d != 'time'},
                ).expand_dims({'time': np.array([np.datetime64(dt.to_datetime64())], dtype='datetime64[ns]')})
                for v in NLDAS_VARS
            })
            datasets.append(nan_ds)

    ds = xr.concat(datasets, dim='time').load()
    for d in datasets:
        d.close()
    del buffers
    return ds


def get_nldas_wy_daily(WY, output_res, thresh, save_path=None, n_workers=8):
    """
    Download hourly NLDAS-2 FORA data for a water year via authenticated HTTPS (GES DISC),
    apply Jennings (2018) binary phase partition at the hourly timestep, aggregate to daily,
    and save as a (time, lat, lon) zarr on the native NLDAS grid.

    Requires Earthdata credentials in ~/.netrc (machine urs.earthdata.nasa.gov).

    Saved variables:
        Rainf   - daily precipitation sum (mm/day)
        Tair    - daily mean air temperature (K)
        SWdown  - daily mean shortwave radiation (W/m^2)
        SNOW    - Jennings (2018) binary snow daily sum (mm/day)
        RAIN    - Jennings (2018) binary rain daily sum (mm/day)
    """
    if save_path is None:
        save_path = os.path.join(
            HOME, "data", "Precipitation", str(WY), "NLDAS", f"NLDAS_daily_WY{WY}.zarr"
        )

    if os.path.exists(save_path):
        print(f"NLDAS daily WY{WY} already exists at {save_path}")
        return xr.open_zarr(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    viirs_dir = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"VIIRSGeoObsDFs/{thresh}_fSCA_Thresh"
    )
    centroids = _collect_centroids(viirs_dir)
    left, right, bottom, top = _bbox_from_centroids(centroids)
    print(f"NLDAS WY{WY}: bbox lon [{left:.2f}, {right:.2f}] lat [{bottom:.2f}, {top:.2f}]")

    earthaccess.login(strategy='netrc')
    session = earthaccess.get_requests_https_session()

    wy_start_ts = pd.Timestamp(WY - 1, 10, 1)
    wy_end_ts   = pd.Timestamp(WY, 9, 30, 23, 59, 59)

    times    = []
    accum    = {v: [] for v in ['Rainf', 'Tair', 'SWdown', 'SNOW', 'RAIN']}
    grid_lat = None
    grid_lon = None

    for m_start in pd.date_range(wy_start_ts, wy_end_ts, freq='MS'):
        m_end = min(m_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), wy_end_ts)
        hours = pd.date_range(m_start, m_end, freq='h')
        print(f"  NLDAS {m_start.strftime('%Y-%m')}: downloading {len(hours)} hourly files...")

        try:
            ds_hr = _load_nldas_month(session, hours, n_workers=n_workers)
        except RuntimeError as e:
            print(f"  Skipping {m_start.strftime('%Y-%m')}: {e}")
            continue

        rename = {}
        if 'latitude'  in ds_hr.dims:
            rename['latitude']  = 'lat'
        if 'longitude' in ds_hr.dims:
            rename['longitude'] = 'lon'
        if rename:
            ds_hr = ds_hr.rename(rename)

        # clip to domain bbox
        lat_ord = ds_hr.lat.values
        lat_s   = slice(bottom, top) if lat_ord[0] < lat_ord[-1] else slice(top, bottom)
        ds_hr   = ds_hr.sel(lat=lat_s, lon=slice(left, right))

        # hourly phase partition on native grid
        rainf  = ds_hr['Rainf'].values.astype(np.float32)
        tair   = ds_hr['Tair'].values.astype(np.float32)
        qair   = ds_hr['Qair'].values.astype(np.float32)
        psurf  = ds_hr['PSurf'].values.astype(np.float32)
        T_C_hr = tair - 273.15
        RH_hr  = _rh_from_q_p(qair, psurf / 100.0, T_C_hr)
        snow_hr, rain_hr = _precip_phase(rainf, T_C_hr, RH_hr)

        dims_hr   = ds_hr['Rainf'].dims
        coords_hr = ds_hr['Rainf'].coords
        ds_ext = ds_hr.assign({
            'SNOW': xr.DataArray(snow_hr, dims=dims_hr, coords=coords_hr),
            'RAIN': xr.DataArray(rain_hr, dims=dims_hr, coords=coords_hr),
        })

        daily = xr.merge([
            ds_ext[['Rainf', 'SNOW', 'RAIN']].resample(time='1D').sum(min_count=1),
            ds_hr[['Tair', 'SWdown']].resample(time='1D').mean(),
        ]).load()
        ds_ext.close()
        ds_hr.close()

        if grid_lat is None:
            grid_lat = daily.lat.values
            grid_lon = daily.lon.values
        times.append(daily.time.values)
        for v in accum:
            accum[v].append(daily[v].values.astype(np.float32))
        daily.close()

    if not times:
        raise RuntimeError(f"No NLDAS data downloaded for WY{WY}")

    time_arr = np.concatenate(times)
    ds_out = xr.Dataset(
        {v: xr.DataArray(np.concatenate(accum[v], axis=0), dims=('time', 'lat', 'lon'))
         for v in accum},
        coords={'time': time_arr, 'lat': grid_lat, 'lon': grid_lon},
    )

    ds_out = ds_out.chunk({'time': 30, 'lat': len(grid_lat), 'lon': len(grid_lon)})
    print(f"Saving NLDAS daily zarr for WY{WY} to {save_path} ...")
    ds_out.to_zarr(save_path, mode='w', consolidated=True)
    print("Done.")
    return xr.open_zarr(save_path)


def add_nldas_df(WY, output_res, threshold):
    """
    Add season-to-date NLDAS met features to each training DF from the (time, lat, lon) zarr.

    Reads from: gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh
    Writes to:  NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh

    New columns:
        NLDAS_precip  - season-to-date total precipitation (mm)
        NLDAS_snow    - season-to-date snow via Jennings (2018) (mm)
        NLDAS_rain    - season-to-date rain via Jennings (2018) (mm)
        NLDAS_APDD    - accumulated positive degree-days since Oct 1 (deg C days)
        NLDAS_T7d     - 7-day mean air temperature ending on obs date (deg C)
        NLDAS_T14d    - 14-day mean air temperature ending on obs date (deg C)
        NLDAS_SW7d    - 7-day mean shortwave radiation ending on obs date (W/m^2)
        NLDAS_SW14d   - 14-day mean shortwave radiation ending on obs date (W/m^2)
    """
    training_df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    os.makedirs(df_path, exist_ok=True)

    zarr_path = os.path.join(
        HOME, "data", "Precipitation", str(WY), "NLDAS", f"NLDAS_daily_WY{WY}.zarr"
    )
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"NLDAS zarr not found at {zarr_path}. Run get_nldas_wy_daily() first."
        )

    ds_nl    = xr.open_zarr(zarr_path)
    WY_start = f"{WY-1}-10-01"
    training_dfs = [f for f in os.listdir(training_df_path) if f.endswith('.parquet')]

    for geofile in tqdm(training_dfs, desc=f"Adding NLDAS features WY{WY}"):
        date    = geofile.split('_')[-1].split('.parquet')[0]
        region  = geofile.split('_')[-2]
        strdate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        print(f"  NLDAS -> {region} {strdate}")

        GDF  = pd.read_parquet(os.path.join(training_df_path, geofile))
        lats = xr.DataArray(GDF['cen_lat'].values, dims='points')
        lons = xr.DataArray(GDF['cen_lon'].values, dims='points')
        GDF.set_index('cell_id', inplace=True)

        ds_sl = (ds_nl.sel(time=slice(WY_start, strdate))
                      .sel(lat=lats, lon=lons, method='nearest')
                      .load())
        T_C   = ds_sl['Tair'].values - 273.15
        SW    = ds_sl['SWdown'].values

        GDF['NLDAS_precip'] = np.round(ds_sl['Rainf'].values.sum(axis=0), 2)
        GDF['NLDAS_snow']   = np.round(ds_sl['SNOW'].values.sum(axis=0),  2)
        GDF['NLDAS_rain']   = np.round(ds_sl['RAIN'].values.sum(axis=0),  2)
        GDF['NLDAS_APDD']   = np.round(np.maximum(T_C, 0).sum(axis=0),    2)
        GDF['NLDAS_T7d']    = np.round(T_C[-7:].mean(axis=0),             2)
        GDF['NLDAS_T14d']   = np.round(T_C[-14:].mean(axis=0),            2)
        GDF['NLDAS_SW7d']   = np.round(SW[-7:].mean(axis=0),              2)
        GDF['NLDAS_SW14d']  = np.round(SW[-14:].mean(axis=0),             2)

        GDF.reset_index(inplace=True)
        table = pa.Table.from_pandas(GDF)
        pq.write_table(table, f"{df_path}/NLDAS_{geofile}", compression='BROTLI')

    ds_nl.close()


# PRISM

def _progress_hook(block_num, block_size, total_size, t):
    downloaded = block_num * block_size
    if total_size > 0:
        t.update(min(block_size, total_size - t.n))
    else:
        t.update(downloaded - t.n)


def _prism_download_days(start, stop, path, var):
    """Download daily PRISM zip files for one variable between start and stop (inclusive)."""
    base_url = "https://services.nacse.org/prism/data/get/us/800m/"
    while start <= stop:
        day = start.strftime("%Y%m%d")
        url = f"{base_url}/{var}/{day}?format=nc"
        out_file = os.path.join(path, day)
        if not os.path.exists(out_file):
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=f'Downloading {var} {day}') as t:
                urllib.request.urlretrieve(
                    url, out_file,
                    reporthook=lambda bn, bs, ts: _progress_hook(bn, bs, ts, t)
                )
        start += timedelta(days=1)


def _prism_unzip(start, stop, zipped_path, unzipped_path):
    while start <= stop:
        day = start.strftime("%Y%m%d")
        zip_file = os.path.join(zipped_path, day)
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(unzipped_path)
            os.remove(zip_file)
        start += timedelta(days=1)


def get_prism(WY, vars=None):
    """
    Download PRISM 800m daily data for the given variables and water year.
    Saves one NC file per variable: data/Precipitation/{WY}/prism_data/{WY}_{var}.nc
    """
    if vars is None:
        vars = ['ppt', 'tmean', 'tdmean']
    if isinstance(vars, str):
        vars = [vars]
    prism_path = f'{HOME}/data/Precipitation/{WY}/prism_800m'
    start = datetime.strptime(f"{WY-1}-10-01", "%Y-%m-%d")
    stop  = datetime.strptime(f"{WY}-09-30",   "%Y-%m-%d")
    date_idx = pd.Index(pd.date_range(start=start, end=stop), name='time')

    for var in vars:
        out_nc = f'{prism_path}/{WY}_{var}.nc'
        if os.path.exists(out_nc):
            print(f"PRISM {var} WY{WY} already exists")
            continue

        zipped_path   = f'{prism_path}/zipped_{var}'
        unzipped_path = f'{prism_path}/unzipped_{var}'
        os.makedirs(zipped_path,   exist_ok=True)
        os.makedirs(unzipped_path, exist_ok=True)

        _prism_download_days(start, stop, zipped_path, var)
        _prism_unzip(start, stop, zipped_path, unzipped_path)

        ds = xr.open_mfdataset(
            f'{unzipped_path}/*.nc',
            combine='nested',
            concat_dim=[date_idx],
        )
        ds.to_netcdf(out_nc)
        ds.close()
        print(f"Saved PRISM {var} WY{WY} -> {out_nc}")


def add_prism_df(WY, output_res, threshold):
    """
    Add season-to-date PRISM precipitation, phase-partitioned snow/rain,
    and ablation features to each training DF.

    Reads from: AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh
    Writes to:  PRISM_AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh

    Requires PRISM NC files from get_prism(WY, vars=['ppt', 'tmean', 'tdmean']).

    New columns:
        PRISM_precip  - season-to-date total precipitation (mm)
        PRISM_snow    - season-to-date snow via Jennings (2018) (mm)
        PRISM_rain    - season-to-date rain via Jennings (2018) (mm)
        PRISM_APDD    - accumulated positive degree-days since Oct 1 (deg C days)
        PRISM_T7d     - 7-day mean air temperature ending on obs date (deg C)
        PRISM_T14d    - 14-day mean air temperature ending on obs date (deg C)
    """
    training_df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    df_path = (
        f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/"
        f"PRISM_AORC_NLDAS_gridMET_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    )
    os.makedirs(df_path, exist_ok=True)

    prism_path = f'{HOME}/data/Precipitation/{WY}/prism_800m'
    for var in ('ppt', 'tmean', 'tdmean'):
        nc_path = f'{prism_path}/{WY}_{var}.nc'
        if not os.path.exists(nc_path):
            raise FileNotFoundError(
                f"PRISM {var} NC not found at {nc_path}. "
                f"Run get_prism({WY}, vars=['ppt','tmean','tdmean']) first."
            )

    # load full WY datasets and reproject to WGS84
    ppt_ds    = xr.open_dataset(f'{prism_path}/{WY}_ppt.nc').rio.write_crs('EPSG:4269').rio.reproject('EPSG:4326')
    tmean_ds  = xr.open_dataset(f'{prism_path}/{WY}_tmean.nc').rio.write_crs('EPSG:4269').rio.reproject('EPSG:4326')
    tdmean_ds = xr.open_dataset(f'{prism_path}/{WY}_tdmean.nc').rio.write_crs('EPSG:4269').rio.reproject('EPSG:4326')

    y_desc = ppt_ds.y.values[0] > ppt_ds.y.values[-1]
    x_asc  = ppt_ds.x.values[0] < ppt_ds.x.values[-1]

    WY_start = f'{WY-1}-10-01'
    training_dfs = [f for f in os.listdir(training_df_path) if f.endswith('.parquet')]

    for geofile in tqdm(training_dfs, desc=f"Adding PRISM features WY{WY}"):
        date    = geofile.split('_')[-1].split('.parquet')[0]
        region  = geofile.split('_')[-2]
        strdate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        print(f"  PRISM -> {region} {strdate}")

        GDF  = pd.read_parquet(os.path.join(training_df_path, geofile))
        meta = GDF[['cell_id', 'cen_lat', 'cen_lon']]
        GDF.set_index('cell_id', inplace=True)

        left   = meta['cen_lon'].min() - 0.1
        right  = meta['cen_lon'].max() + 0.1
        bottom = meta['cen_lat'].min() - 0.1
        top    = meta['cen_lat'].max() + 0.1
        y_s    = slice(top, bottom) if y_desc else slice(bottom, top)
        x_s    = slice(left, right) if x_asc  else slice(right, left)

        lats = xr.DataArray(meta['cen_lat'].values, dims='points')
        lons = xr.DataArray(meta['cen_lon'].values, dims='points')

        def _slice_pts(ds):
            sliced = ds.loc[dict(time=slice(WY_start, strdate))].sel(y=y_s, x=x_s)
            return sliced['Band1'].sel(x=lons, y=lats, method='nearest').values

        ppt_vals    = _slice_pts(ppt_ds)    # mm
        tmean_vals  = _slice_pts(tmean_ds)  # deg C
        tdmean_vals = _slice_pts(tdmean_ds) # deg C

        RH = _rh_from_dew(tmean_vals, tdmean_vals)
        snow_arr, rain_arr = _precip_phase(ppt_vals, tmean_vals, RH)

        GDF['PRISM_precip'] = np.round(ppt_vals.sum(axis=0),               2)
        GDF['PRISM_snow']   = np.round(snow_arr.sum(axis=0),               2)
        GDF['PRISM_rain']   = np.round(rain_arr.sum(axis=0),               2)
        GDF['PRISM_APDD']   = np.round(np.maximum(tmean_vals, 0).sum(axis=0), 2)
        GDF['PRISM_T7d']    = np.round(tmean_vals[-7:].mean(axis=0),       2)
        GDF['PRISM_T14d']   = np.round(tmean_vals[-14:].mean(axis=0),      2)

        GDF.reset_index(inplace=True)
        table = pa.Table.from_pandas(GDF)
        pq.write_table(table, f"{df_path}/PRISM_{geofile}", compression='BROTLI')

    ppt_ds.close()
    tmean_ds.close()
    tdmean_ds.close()
