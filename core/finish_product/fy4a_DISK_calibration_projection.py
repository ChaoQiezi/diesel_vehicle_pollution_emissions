# @炒茄子  2023-07-04

"""
当前脚本用于处理FY4A数据，包括辐射定标、基于地理查找表(.raw文件)的校正、GeoTIFF文件的创建等, 最后输出为GeoTIFF文件(WGS84)
--批量处理
"""

# 函数
# @炒茄子  2023-07-09

import os
import glob
import h5py
import numpy as np
from numpy import sqrt, sin, cos, power, arctan, pi
from osgeo import gdal
from scipy.interpolate import griddata


# 读取HDF5文件数据集
def h5_data_get(hdf_path, dataset_name):
    """
    读取HDF5文件中的数据集
    :param hdf_path: HDF5文件的路径
    :param dataset_name: HDF5文件中数据集的名称
    :return: 数据集
    """

    with h5py.File(hdf_path, 'r') as f:
        dataset = f[dataset_name][:]

    return dataset


# 读取HDF5文件数据集属性
def h5_attr_get(hdf_path, dataset_name, attr_name):
    """
    获取HDF5文件数据集的属性
    :param hdf_path: HDF5文件的路径
    :param dataset_name: HDF5文件中数据集的名称
    :param attr_name: HDF5文件中数据集属性的名称
    :return: 数据集属性
    """

    with h5py.File(hdf_path, 'r') as f:
        attr = f[dataset_name].attrs[attr_name]

    return attr


# 对FY4A数据集进行辐射定标
def fy4a_calibration(hdf_path, nom_channel_name, cal_channel_name):
    """
    获取FY4A数据集并对其进行辐射定标
    :param hdf_path: HDF5文件的路径
    :param nom_channel_name: 待定标的数据集名称
    :param cal_channel_name: 用于定标的数据集名称
    :return: 辐射定标后的数据集
    """

    # 读取数据集
    nom_channel = h5_data_get(hdf_path, nom_channel_name)
    cal_channel = h5_data_get(hdf_path, cal_channel_name)

    # 读取数据集属性
    nom_min, nom_max = h5_attr_get(hdf_path, nom_channel_name, 'valid_range')
    cal_min, cal_max = h5_attr_get(hdf_path, cal_channel_name, 'valid_range')
    nom_fill_value = h5_attr_get(hdf_path, nom_channel_name, 'FillValue')[0]
    cal_fill_value = h5_attr_get(hdf_path, cal_channel_name, 'FillValue')[0]

    # 数据集掩码和填充值预准备
    nom_mask = (nom_channel > nom_min) & (nom_channel < nom_max) & (nom_channel != nom_fill_value)
    cal_mask = (cal_channel > cal_min) | (cal_channel < cal_max)
    cal_channel[cal_channel == cal_fill_value] = int(cal_min - 10)

    # 辐射定标
    target_channel = np.zeros_like(nom_channel, dtype=np.float32)
    target_channel[nom_mask] = cal_channel[cal_mask][nom_channel[nom_mask]]

    # 无效值处理(包括不在范围及填充值)
    target_channel[~nom_mask] = np.nan
    target_channel[target_channel == int(cal_min - 10)] = np.nan

    return target_channel


def get_lon_lat(dataset):
    """
    FY4A数据集经纬度计算(基于FY4A数据格式说明书的公式)
    :param dataset: FY4A数据集
    :return: 经纬度
    """
    # 行列号 ==> 经纬度, 公式由FY4A数据格式说明书给出

    # 获取数据集的行列号
    rows, cols = dataset.shape
    # 生成行列号矩阵
    col_mesh, row_mesh = np.meshgrid(np.arange(cols), np.arange(rows))

    # 基本参数(均只用于FY4A 4000m分辨率, 其它分辨率需要依据官方说明修改参数)
    ea = 6378.137  # 地球长半轴, 单位km
    eb = 6356.7523  # 地球短半轴, 单位km
    h = 42164  # 卫星高度, 单位km, 即地心到卫星质心的距离
    lon0 = 104.7  # 投影中心经度, 单位度, 也即卫星星下点所在的经度
    coff = 1373.5  # 列偏移
    cfac = 10233137  # 列比例因子
    loff = 1373.5  # 行偏移
    lfac = 10233137  # 行比例因子

    # step1. 求x, y
    x = (pi * (col_mesh - coff)) / (180 * (2 ** (-16)) * cfac)
    y = (pi * (row_mesh - loff)) / (180 * (2 ** (-16)) * lfac)

    # step2. 求sd, sn, s1, s2, s4, sxy
    sd = sqrt(
        power(h * cos(x) * cos(y), 2) - (power(cos(y), 2) + power(ea / eb, 2) * power(sin(y), 2)) * (h ** 2 - ea ** 2)
    )
    sn = (h * cos(x) * cos(y) - sd) / (power(cos(y), 2) + power(ea / eb, 2) * power(sin(y), 2))
    s1 = h - sn * cos(x) * cos(y)
    s2 = sn * sin(x) * cos(y)
    s3 = -sn * sin(y)
    sxy = sqrt(power(s1, 2) + power(s2, 2))

    # step3. 求lon, lat
    lon = lon0 + arctan(s2 / s1) * 180 / pi
    lat = arctan(power(ea / eb, 2) * s3 / sxy) * 180 / pi

    return lon, lat


def read_geo(geo_path, row_name, col_name):
    """
    基于FY4A的GEO文件(定位配准产品中的对地图像定位产品)获取像元级的行列号矩阵, 并通过公式计算获取对应的经纬度数据集
    :param geo_path: FY4A的GEO文件路径
    :param row_name: 行号数据集名称
    :param col_name: 列号数据集名称
    :return: 经纬度数据集, 形式(lon, lat)
    """
    # 读取数据集及其属性
    original_row_mesh = h5_data_get(geo_path, row_name)
    original_col_mesh = h5_data_get(geo_path, col_name)
    row_fill_value = h5_attr_get(geo_path, col_name, 'FillValue')[0]
    col_fill_value = h5_attr_get(geo_path, row_name, 'FillValue')[0]
    # 有效值掩码
    mask = (original_row_mesh != row_fill_value) & (original_col_mesh != col_fill_value)
    # 有效值
    row_mesh = original_row_mesh[mask]
    col_mesh = original_col_mesh[mask]

    # 基本参数(均只用于FY4A 4000m分辨率, 其它分辨率需要依据官方说明修改参数)
    ea = 6378.137  # 地球长半轴, 单位km
    eb = 6356.7523  # 地球短半轴, 单位km
    h = 42164  # 卫星高度, 单位km, 即地心到卫星质心的距离
    lon0 = 104.7  # 投影中心经度, 单位度, 也即卫星星下点所在的经度
    coff = 1373.5  # 列偏移
    cfac = 10233137  # 列比例因子
    loff = 1373.5  # 行偏移
    lfac = 10233137  # 行比例因子

    # step1. 求x, y
    x = (pi * (col_mesh - coff)) / (180 * (2 ** (-16)) * cfac)
    y = (pi * (row_mesh - loff)) / (180 * (2 ** (-16)) * lfac)

    # step2. 求sd, sn, s1, s2, s4, sxy
    sd = sqrt(
        power(h * cos(x) * cos(y), 2) - (power(cos(y), 2) + power(ea / eb, 2) * power(sin(y), 2)) * (h ** 2 - ea ** 2)
    )
    sn = (h * cos(x) * cos(y) - sd) / (power(cos(y), 2) + power(ea / eb, 2) * power(sin(y), 2))
    s1 = h - sn * cos(x) * cos(y)
    s2 = sn * sin(x) * cos(y)
    s3 = -sn * sin(y)
    sxy = sqrt(power(s1, 2) + power(s2, 2))

    # step3. 求lon, lat
    lon = lon0 + arctan(s2 / s1) * 180 / pi
    lat = arctan(power(ea / eb, 2) * s3 / sxy) * 180 / pi

    # 输出
    out_lon = np.zeros_like(original_row_mesh, dtype=np.float32)
    out_lat = np.zeros_like(original_row_mesh, dtype=np.float32)
    out_lon[mask] = lon
    out_lat[mask] = lat
    out_lon[~mask] = np.nan
    out_lat[~mask] = np.nan

    return out_lon, out_lat


# 读取地理对照表(.raw, FY4A)
def read_glt(raw_path, shape):
    """
    该函数用于读取地理对照表
    :param raw_path: 地理查找表的路径
    :param shape: 地理查找表的形状
    :return: 二维数组(lon, lat)
    """
    # 读取二进制数据，将其转换为两个2D数组（经度和纬度）
    with open(raw_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.float64)  # 读取二进制数据, '<f8'表示小端模式，8字节浮点数
    raw_lat = raw_data[::2].reshape(shape)
    raw_lon = raw_data[1::2].reshape(shape)
    raw_lat[(raw_lat < -90.0) | (raw_lat > 90.0)] = np.nan
    raw_lon[(raw_lon < -180.0) | (raw_lon > 180.0)] = np.nan

    """
    官方文档中的说明：文件从北向南按行（从西到东）填写，每个网格对应 16 字节，前8字节为经度值，后 8 字节为纬度值，double 类型，高字节在前
    -----------------------------------------------------------------------------------------------------------------
    |  通过查看输出的lon, lat发现, lon的范围为[-90, 90], lat的范围为[-180, 180];
    |  通过读取二进制流时，如果以高字节在前(Big-Endian/小端模式)的方式(dtype=‘<f8’, , <表示小, 若填入np.float64等价于'<f8'
    |  即默认是小端模式)读取数据时，会出现错误的结果;
    -----------------------------------------------------------------------------------------------------------------
    综上所述, 官方文档的说明存在错误: 应该为前8字节为纬度值，后 8 字节为经度值，double 类型,低字节在前;
    参考: https://blog.csdn.net/modabao/article/details/83834580
    """

    return raw_lon, raw_lat


def clip(original_band, original_lon, original_lat, lon_lat_range=None):
    """
    裁剪, 非常规意义的矩形裁剪, 而是基于经纬度范围的裁剪并进行重组
    :param original_band: 原始数据集
    :param original_lon: 对应的经度矩阵
    :param original_lat: 对应的纬度矩阵
    :param lon_lat_range: 经纬度范围, 默认为中国区域(形式:[lon_min, lon_max, lat_min, lat_max])
    :return: 裁剪并重组好的数据集, 包括(band, lon, lat)
    """
    if lon_lat_range is None:
        lon_lat_range = [-170, 170, -80, 80]  # 中国区域(为兼容REGC, 仅南至海南省主岛屿)
    lon_min, lon_max, lat_min, lat_max = lon_lat_range

    # 获取经纬度范围内的掩码
    mask = (original_lon >= lon_min) & (original_lon <= lon_max) & (original_lat >= lat_min) & (original_lat <= lat_max)
    valid_lon = original_lon[mask]
    valid_lat = original_lat[mask]
    valid_band = original_band[mask]
    valid_num = np.sum(mask)  # True为1, False为0
    valid_num_sqrt = int(sqrt(valid_num))
    # 重组数据
    reform_lon = valid_lon[:valid_num_sqrt ** 2].reshape(valid_num_sqrt, valid_num_sqrt)
    reform_lat = valid_lat[:valid_num_sqrt ** 2].reshape(valid_num_sqrt, valid_num_sqrt)
    reform_band = valid_band[:valid_num_sqrt ** 2].reshape(valid_num_sqrt, valid_num_sqrt)

    return reform_band, reform_lon, reform_lat


def glt_warp(original_dataset, original_lon, original_lat, out_res, method='nearest'):
    """
    基于地理查找表的校正
    :param original_dataset:  待校正的数据
    :param original_lon:  原始数据的经度
    :param original_lat:  原始数据的纬度
    :param out_res: 输出数据的分辨率
    :param method: 插值方法, 默认为最近邻插值('linear', 'nearest', 'cubic')
    :return: 校正后的数据
    """

    # 校正数据的最大最小经纬度
    lon_min, lon_max = np.nanmin(original_lon), np.nanmax(original_lon)
    lat_min, lat_max = np.nanmin(original_lat), np.nanmax(original_lat)

    # 生成目标数据的经纬度矩阵
    grid_lon, grid_lat = np.meshgrid(
        np.arange(lon_min, lon_max, out_res),
        np.arange(lat_max, lat_min, -out_res),
    )

    # 插值
    interp_dataset = griddata(
        (original_lon.ravel(), original_lat.ravel()),  # 原始数据的经纬度, ravel()将多维数组转换为一维数组
        original_dataset.ravel(),  # 原始数据
        (grid_lon, grid_lat),  # 目标数据的经纬度
        method=method,  # 插值方法
        fill_value=np.nan,  # 用于填充超出边界的点
    )

    return interp_dataset


# 输出TIF
def write_tiff(out_path, dataset, transform, nodata=np.nan):
    """
    输出TIFF文件
    :param out_path: 输出文件的路径
    :param dataset: 待输出的数据
    :param transform: 坐标转换信息(形式:[左上角经度, 经度分辨率, 旋转角度, 左上角纬度, 旋转角度, 纬度分辨率])
    :param nodata: 无效值
    :return: None
    """

    # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, dataset[0].shape[1], dataset[0].shape[0], len(dataset), gdal.GDT_Float32)

    # 设置基本信息
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection('WGS84')

    # 写入数据
    for i in range(len(dataset)):
        out_ds.GetRasterBand(i + 1).WriteArray(dataset[i])  # GetRasterBand()传入的索引从1开始, 而非0
        out_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    out_ds.FlushCache()


# 准备工作
in_path = r'F:\diesel_vehicle_pollution_emissions\data\DISK'  # FY4A L1(DISK全圆盘)产品所在文件夹
raw_path = r'F:\diesel_vehicle_pollution_emissions\data\DISK\FullMask_Grid_4000.raw'  # 地理对照表(地理查找表)文件路径
out_path = r'F:\diesel_vehicle_pollution_emissions\data\output'  # 输出文件夹
out_res = 0.036  # 0.036为输出的分辨率(°/度), 约为4km

# 检索HDF5文件
paths = glob.glob(in_path + r'\*DISK*FDI*.HDF')
# 读取地理查找表
lon, lat = read_glt(raw_path, (2748, 2748))  # 地理查找表的shape与band的shape相同,或可从官方文档中获取((2748, 2748))
# 循环处理每一个HDF5文件
for path in paths[:1]:
    # HDF5文件的文件名
    hdf_name = os.path.basename(path).split('.')[0]
    # 读取HDF5文件
    hdf = h5py.File(path, 'r')
    nom_channels_name = [key for key in hdf.keys() if 'NOMChannel' in key]  # 读取各个波段的通道名
    cal_channels_name = [key for key in hdf.keys() if 'CALChannel' in key]  # 读取对应各个波段的辐射定标通道名
    channel_names = zip(nom_channels_name, cal_channels_name)
    # 循环处理每一个通道
    bands = []
    transform = None
    a = 0
    for nom_channel_name, cal_channel_name in channel_names:
        if a != 0:
            break
        a = 1
        # 辐射定标
        band = fy4a_calibration(path, nom_channel_name, cal_channel_name)
        # 裁剪
        clip_band, clip_lon, clip_lat = clip(band, lon, lat)
        # GLT校正
        processed_band = glt_warp(clip_band, clip_lon, clip_lat, out_res, method='linear')
        # 仿射变换参数
        transform = (
            np.nanmin(clip_lon),  # 左上角经度
            out_res,  # x方向分辨率
            0,  # 旋转角度
            np.nanmax(clip_lat),  # 左上角纬度
            0,  # 旋转角度
            -out_res,  # y方向分辨率, 由于自左上角开始(纬度往下逐渐减小), 因此为负;
        )
        bands.append(processed_band)
    # 将处理后的数据和地理查找表的信息写入GeoTIFF文件
    hdf_out_path = out_path + '\\' + hdf_name + '.tif'
    write_tiff(hdf_out_path, bands, transform)
