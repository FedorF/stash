import math
from typing import List, Optional, Tuple

import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point



EARTH_RAD = 6367000


def coos_to_gdf(
    df: pd.DataFrame, 
    lng_colname: str = 'longitude', 
    lat_colname: str = 'latitude',
) -> gpd.GeoDataFrame:
    
    df[lng_colname] = df[lng_colname].astype(float)
    df[lat_colname] = df[lat_colname].astype(float)

    df['geometry'] = df.loc[:, [lng_colname, lat_colname]].apply(Point, axis=1)
    df = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry='geometry')
    
    return df


def load_wkt(x, ignore_errors: bool = False):
    if ignore_errors:
        try:
            output = wkt.loads(x)
            return output
        except:
            return None
    else:
        return wkt.loads(x)

    
def wkt_to_gdf(
    df: pd.DataFrame, 
    wkt_colname: str = 'geometry', 
    crs: str = 'EPSG:4326', 
) -> gpd.GeoDataFrame:
    df['geometry'] = df[wkt_colname].map(load_wkt)
    return gpd.GeoDataFrame(df, crs=crs, geometry='geometry')


def calc_epsg(lng):
    return 32601 + round((lng + 180) / 6) % 60


def calc_area(series, epsg=6933):
    return gpd.GeoSeries(series, crs='epsg:4326').to_crs(epsg=epsg).area


def calc_buffer(
    gdf: gpd.GeoDataFrame,
    radius: float,
    long_colname: str = 'longitude',
    init_epsg: float = 4326,
    buffer_colname = 'buffer',
):
    """Returns gdf with buffer of radius in meters"""
    
    gdf['epsg'] = gdf[long_colname].map(calc_epsg)
    groups = []
    for epsg, group in gdf.groupby('epsg'):
        group[buffer_colname] = group['geometry'].to_crs(epsg).buffer(radius).to_crs(init_epsg)
        groups.append(group)

    gdf = gpd.GeoDataFrame(pd.concat(groups), crs=f"EPSG:{init_epsg}", geometry=buffer_colname)
    gdf = gdf.drop(columns=['epsg'])
    return gdf
    
    
def calc_area_by_group(gdf, geometry_col='geometry', epsg_col='epsg'):
    if epsg_col not in gdf.columns:
        raise ValueError('There is no EPSG column in DataFrame.')
    else:
        groups = []
        for epsg, group in gdf.groupby(epsg_col):
            group['area'] = calc_area(group[geometry_col], epsg)
            groups.append(group)
        
        return pd.concat(groups)


def cut_rectangle(df, lat_min, lat_max, long_min, long_max, lng_colname='lng', lat_colname='lat'):
    mask = (
            (df[lng_colname] >= long_min) & (df[lng_colname] <= long_max)
            & (df[lat_colname] >= lat_min) & (df[lat_colname] <= lat_max)
    )
    return df.loc[mask].copy()


def calc_haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return c * EARTH_RAD


def calc_polygons_intersection(a, b):
    if a.is_valid and b.is_valid:
        return a.intersection(b)

    elif a.is_valid and not b.is_valid:
        return a.intersection(b.buffer(0))

    elif not a.is_valid and b.is_valid:
        return a.buffer(0).intersection(b)

    return a.buffer(0).intersection(b.buffer(0))


def plot_geospatial_figure(
    dfs: List[pd.DataFrame],
    geom_colname: str = 'geometry',
    colors: Optional[List[str]] = None,
    start_point: Tuple[float, float] = (60, 70),
    max_zoom: int = 20,
    zoom_start: int = 3,
) -> folium.Map:
    """Plot geographical map."""
    mmap = folium.Map(start_point, max_zoom=max_zoom, zoom_start=zoom_start)
    for i, df in enumerate(dfs):
        for geom in df[geom_colname].values:
            color = lambda x: {'color': colors[i] if colors and len(colors) > i else 'red'}  
            geom = wkt.loads(geom) if isinstance(geom, str) else geom
            folium.GeoJson(geom, style_function=color).add_to(mmap)    
    return mmap
