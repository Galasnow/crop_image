import os
import time
from typing import Sequence, Callable

import numpy as np
import cv2
from osgeo import gdal
import yaml


def time_it(func: Callable):
    """ decorator used to calculate time, use @time_it in front of any function definition """
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = func(*args, **kwargs)
        time_end = time.time()
        print('consumed time of "', getattr(func, "__name__"), '" is : ', str(time_end - time_start) + ' s')
        return ret

    return wrapper


def read_config(config_path: str):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
        input_path = config['input_path']
        output_path = config['output_path']
        output_format = config['output_format']
        mode = config['mode']
        crop_size = config['crop_size']
        center = config['center']['center']
        gap_size = config['grid']['gap_size']
        params = [input_path, output_path, output_format, mode, center, crop_size, gap_size]
    return params


def create_folder(path: str):
    existence = os.path.exists(path)
    if not existence:
        os.makedirs(path)


def generate_new_geo_transform(window, original_geo_transform, offset: int):
    new_geo_transform = list(original_geo_transform)
    new_geo_transform[0] = new_geo_transform[0] + new_geo_transform[1] * (window['x_start'] + offset)
    new_geo_transform[3] = new_geo_transform[3] + new_geo_transform[5] * (window['y_start'] + offset)
    return new_geo_transform


def read_image(input_path: str):
    stem, suffix = os.path.splitext(input_path)
    file_name_stem = stem.split('/')[-1]
    geo_transform = None
    projection = None
    match suffix:
        case '.jpg' | '.png':
            image = cv2.imread(input_path)
        case '.jp2':  # Sentinel-2
            image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        case '.tif' | '.tiff':
            with gdal.Open(input_path) as tiff_file:
                geo_transform = tiff_file.GetGeoTransform()
                projection = tiff_file.GetProjection()
                image = np.transpose(tiff_file.ReadAsArray(), [1, 2, 0])

        case _:
            raise RuntimeError('Unknown file type!')
    return image, file_name_stem, geo_transform, projection


def expand_image(image, gap_size: int, fill_value=None):
    half_gap_size = int(gap_size / 2)
    if fill_value is None:
        fill_value = (0, 0, 0)
    out_image = cv2.copyMakeBorder(image, half_gap_size, half_gap_size, half_gap_size, half_gap_size,
                                   cv2.BORDER_CONSTANT, value=fill_value)
    return out_image


def copy_image(image, crop_size: Sequence[int], window, fill_value=None):
    crop_width = window['x_stop'] - window['x_start']
    crop_height = window['y_stop'] - window['y_start']
    x_fill_width = int((crop_size[0] - crop_width) / 2)
    y_fill_height = int((crop_size[1] - crop_height) / 2)

    if image.ndim == 2:
        if fill_value is None:
            out_image = image[window['y_start']:window['y_stop'], window['x_start']:window['x_stop']]
        else:
            out_image = cv2.copyMakeBorder(
                image[window['y_start']:window['y_stop'], window['x_start']:window['x_stop']],
                y_fill_height, y_fill_height, x_fill_width, x_fill_width,
                cv2.BORDER_CONSTANT, value=fill_value)
    elif image.ndim == 3:
        if fill_value is None:
            out_image = image[window['y_start']:window['y_stop'], window['x_start']:window['x_stop'], :]
        else:
            out_image = cv2.copyMakeBorder(
                image[window['y_start']:window['y_stop'], window['x_start']:window['x_stop'], :],
                y_fill_height, y_fill_height, x_fill_width, x_fill_width,
                cv2.BORDER_CONSTANT, value=fill_value)
    else:
        raise RuntimeError("image's dimension is not 2 or 3 !")

    return out_image


def export_image(out_image, output_path: str, file_name_stem: str, window, **kwargs):
    out_image_stem = f'{file_name_stem}_{window['x_start']}_{window['x_stop']}_{window['y_start']}_{window['y_stop']}'
    output_format = 'tif'
    jpg_quality = 95
    jp2_compression = 950
    tif_data_type = 'int16'
    crop_size = None
    gap_size = None
    geo_transform = None
    projection = None
    for key, value in kwargs.items():
        if key == 'output_format':
            if value in ['jpg', 'png', 'jp2', 'tif']:
                output_format = value
            else:
                raise RuntimeError('"output_format" keyword argument must be "jpg", "png", "jp2" or "tif"')
        elif key == 'jpg_quality':
            if 0 < value <= 100:
                jpg_quality = value
            else:
                raise RuntimeError('"jpg_quality" keyword argument range from 1 to 100')
        elif key == 'jp2_compression':
            if 0 <= value <= 1000:
                jp2_compression = value
            else:
                raise RuntimeError('"jp2_jp2_compression" keyword argument range from 0 to 1000')
        elif key == 'tif_data_type':
            if value is str:
                tif_data_type = value
            else:
                raise RuntimeError('"tif_data_type" keyword argument should be string')
        elif key == 'crop_size':
                crop_size = value
        elif key == 'crop_size':
                gap_size = value
        elif key == 'geo_transform':
                geo_transform = value
        elif key == 'projection':
                projection = value

    match output_format:
        case 'jpg':
            if out_image.ndim == 3 and out_image.shape[0] > 3:
                out_image = out_image[:, :, 0:3]
            cv2.imwrite(f'{output_path}/{out_image_stem}.jpg', out_image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        case 'png':
            # TODO
            cv2.imwrite(f'{output_path}/{out_image_stem}.png', out_image)
        case 'jp2':
            # TODO
            cv2.imwrite(f'{output_path}/{out_image_stem}.jp2', out_image,
                        [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), jp2_compression])
        case 'tif':
            driver = gdal.GetDriverByName('GTiff')
            if out_image.ndim == 3:
                band_count = out_image.shape[-1]
            else:
                band_count = 1
                # Create a new GeoTIFF file to store the result
            with driver.Create(f'{output_path}/{out_image_stem}.tif', crop_size[0], crop_size[1], band_count,
                               gdal.GDT_UInt16) as out_tiff:
                # Set the geotransform and projection information for the out TIFF based on the input tif
                output_geo_transform = generate_new_geo_transform(window, geo_transform, int(-gap_size / 2))
                out_tiff.SetGeoTransform(output_geo_transform)
                out_tiff.SetProjection(projection)

                # Write the out array to the first band of the new TIFF
                if out_image.ndim == 3:
                    for i in range(out_image.shape[-1]):
                        out_tiff.GetRasterBand(i + 1).WriteArray(out_image[:, :, i])
                else:
                    out_tiff.GetRasterBand(1).WriteArray(out_image)

                # Write the data to disk
                out_tiff.FlushCache()
        case _:
            raise RuntimeError('Unknown file type!')