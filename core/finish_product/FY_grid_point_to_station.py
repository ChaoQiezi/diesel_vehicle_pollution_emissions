# @Author   : ChaoQiezi
# @Time     : 2023-08-28  9:54
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to match the nearest valid grid points of the GSMap dataset by the latitude and longitude of the
weather stations.
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
from osgeo import gdal


def read_tiff(path, geo_info=True):
    df = gdal.Open(path)
    ds = df.GetRasterBand(1).ReadAsArray()
    if geo_info:
        geo_info = df.GetGeoTransform()
        del df
        return ds, geo_info
    else:
        del df
        return ds

def match_by_row_col(station, raster, geo_info, col_name='precipitation_by_row_col'):
    # calculate the row and col on the raster for each station
    station['row'] = np.floor((station['lat'] - geo_info[3]) / geo_info[5]).astype(int)  # because the first row is 0, so floor
    station['col'] = np.floor((station['lon'] - geo_info[0]) / geo_info[1]).astype(int)

    # eliminate the stations that are out of the raster
    station = station[(station['row'] >= 0) & (station['row'] < raster.shape[0]) & (station['col'] >= 0) & (
            station['col'] < raster.shape[1])].reset_index(drop=True)

    station[col_name] = raster[station['row'], station['col']]

    return station

def set_nan_by_hour(station_year, year, hour_path=None):
    if not hour_path:
        hour_path = r'E:\风云4A QPE数据 REGC 数据缺失情况-小时.xlsx'
    hour_df = pd.read_excel(hour_path, header=None).iloc[:, [1]]
    hour_df.columns = ['日期']
    hour_df['日期'] = pd.to_datetime(hour_df['日期'], format="%Y%m%d%H")
    hour_df = hour_df[hour_df['日期'].dt.year == year].reset_index(drop=True)
    hour_df['年月日'] = hour_df['日期'].dt.strftime('%Y%m%d')

    _nodata = hour_df.groupby('年月日').count()
    nodata_row = pd.Series(name='无效值', dtype=object)
    station_year.append(nodata_row)

    for index, missing_hour in zip(_nodata.index, _nodata['日期']):
        station_year.loc['无效值', index] = int(-1000 + missing_hour)

    return station_year


# preparation
in_dir = r'E:\中国大陆风云4a天'
station_path = r'E:\站点结果\CN_PRCPst2825_Clist2421_station2016st839_match.xls'
out_dir = r'E:\站点结果'
start_date = datetime.datetime(2018, 3, 12)
end_date = datetime.datetime(2018, 12, 31)
days = (end_date - start_date).days + 1

# preprocessing
station = pd.read_excel(station_path, sheet_name='PRCPst2422')
station = station[['id_2825', 'lon', 'lat', 'sheng', 'name']]
station.columns = ['station_id', 'lon', 'lat', 'province', 'station_name']
station_year = station.copy()
folder_name = os.path.basename(in_dir)

# match-method 1
print('start_date: {}'.format(start_date))
print('end_date: {}'.format(end_date))
print('handle: {}'.format(folder_name))
for day in range(days):
    date = start_date + datetime.timedelta(days=day)
    date_str = date.strftime('%Y%m%d')
    path = os.path.join(in_dir, '*{}.tiff'.format(date_str))
    path = glob.glob(path)

    if len(path) != 1:
        print('There is a problem with the file: {}, len: {}'.format(date_str, len(path)))
        continue

    # read raster matrix of the *.tif file
    raster, geo_info = read_tiff(path[0])

    # match
    station_year = match_by_row_col(station_year, raster, geo_info, date_str)

    # save
    if ((date.month == 12) and (date.day == 31)) or (day == (days - 1)):
        # nodata_value ==> -1000
        station_year = station_year.replace(np.nan, -1000)
        # nodata_value ==> -999, -998··· according to the missing hour
        station_year = set_nan_by_hour(station_year, date.year, r'E:\missing_hour.xls')
        # out
        out_path = os.path.join(out_dir, '{}_{}.csv'.format(folder_name, date.year))
        station_year.to_csv(out_path, index=False, encoding='utf-8-sig')  # utf-8-sig: 解决中文乱码问题
        station_year = station.copy()
        print('Finish: {} - {}'.format(folder_name, date.year))
# else:
#     out_path = os.path.join(out_dir, '{}_{}.csv'.format(folder_name, date.year))
#     station_year.to_csv(out_path, index=False, encoding='utf-8-sig')
#     print('Finish: {} - {}'.format(folder_name, date.year))
print('Finish: {}'.format(folder_name))

# # match-method 2
# for index, row in station.iterrows():
#     lon = row['lon']
#     lat = row['lat']
#     station.loc[index, 'precipitation_by_window'] = match_by_window(lon, lat, raster, geo_info)


