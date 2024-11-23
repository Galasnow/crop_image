import argparse
import logging
from tqdm import tqdm
from utils import *

logging.basicConfig(format='%(levelname)s: %(funcName)s: %(message)s', level=logging.INFO)
gdal.UseExceptions()

center_by_click = [-1, -1]
exact_coordinates = [-1, -1]
support_file_list = ['jpg', 'png', 'jp2', 'tif', 'tiff']


def get_coordinate_callback(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        global center_by_click
        center_by_click = [x, y]


def generate_grid(image_shape: Sequence[int], crop_size: Sequence[int], gap_size: int):
    w = image_shape[1]
    h = image_shape[0]

    x_split = range(0, w, (crop_size[0] - gap_size))
    y_split = range(0, h, (crop_size[1] - gap_size))
    logging.debug(f'x_split = {x_split}')
    logging.debug(f'y_split = {y_split}')

    windows = []
    for x_start in x_split:
        x_stop = min(x_start + crop_size[0], w)
        for y_start in y_split:
            y_stop = min(y_start + crop_size[1], h)

            window = {'x_start': x_start,
                      'x_stop': x_stop,
                      'y_start': y_start,
                      'y_stop': y_stop
                      }
            windows.append(window)

    logging.debug(f'windows = {windows}')
    return windows


def crop_image_by_grid(input_path: str, output_path: str, crop_size: Sequence[int], gap_size: int, show=False, **kwargs):
    swap_rb_channel = False
    output_format = 'tif'
    fill_value = None
    for key, value in kwargs.items():
        if key == 'swap_rb_channel':
            if isinstance(value, bool):
                swap_rb_channel = value
            else:
                raise RuntimeError('"swap_rb_channel" keyword argument must be True or False')
        elif key == 'output_format':
            if value in support_file_list:
                output_format = value
            else:
                raise RuntimeError(f'"output_format" keyword argument must in {support_file_list}')
        elif key == 'fill_value':
            fill_value = value

    (image, file_name_stem, geo_transform, projection) = read_image(input_path)
    if swap_rb_channel:
        image[[0, 2], :, :] = image[[2, 0], :, :]

    image_preview = image

    if image_preview.size > 5e8:
        resize_scale = np.sqrt(5e8 / image_preview.size)
        image_preview = cv2.resize(image_preview,
                                   [int(image_preview.shape[0] * resize_scale),
                                    int(image_preview.shape[1] * resize_scale)])

    if show:
        cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Input Image', get_coordinate_callback)
        cv2.imshow('Input Image', image_preview)
        cv2.waitKey(0)

    if output_format == 'jpg':
        image = image / np.max(image)
        image *= 255

    expanded_image = expand_image(image, gap_size, fill_value)
    windows = generate_grid(expanded_image.shape, crop_size, gap_size)

    for window in windows:
        out_image = copy_image(expanded_image, crop_size, window, fill_value)

        if show:
            cv2.namedWindow('Out Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Out Image', out_image)
            cv2.waitKey(0)

        export_image(out_image, output_path, file_name_stem, window,
                     crop_size=crop_size, gap_size=gap_size, geo_transform=geo_transform, projection=projection, **kwargs)


def batch_crop_image_by_grid(input_path: str, *args, **kwargs):
    image_name_list = [image_name for image_name in os.listdir(input_path)
                       if os.path.splitext(image_name)[-1].replace('.', '') in support_file_list]
    for image_name in tqdm(image_name_list):
        crop_image_by_grid(f'{input_path}/{image_name}', *args, **kwargs)


def generate_center_based_window(image_shape: Sequence[int], center: Sequence[int], crop_size: Sequence[int]) -> dict:
    x_start = max(0, center[0] - int(crop_size[0] / 2))
    x_stop = min(image_shape[1], center[0] + int(crop_size[0] / 2))
    y_start = max(0, center[1] - int(crop_size[1] / 2))
    y_stop = min(image_shape[0], center[1] + int(crop_size[1] / 2))
    window = {'x_start': x_start,
              'x_stop': x_stop,
              'y_start': y_start,
              'y_stop': y_stop
              }
    logging.debug(window)
    return window


def crop_image_by_center(input_path: str, output_path: str, center: Sequence[int], crop_size: Sequence[int],
                         show=True, **kwargs):
    swap_rb_channel = False
    output_format = 'tif'
    fill_value = None
    for key, value in kwargs.items():
        if key == 'output_format':
            if value in support_file_list:
                output_format = value
            else:
                raise RuntimeError(f'"output_format" keyword argument must in {support_file_list}')
        elif key == 'fill_value':
            fill_value = value

    (image, file_name_stem, geo_transform, projection) = read_image(input_path)
    if swap_rb_channel:
        image[[0, 2], :, :] = image[[2, 0], :, :]

    image_preview = image

    if image_preview.size > 5e8:
        resize_scale = np.sqrt(5e8 / image_preview.size)
        image_preview = cv2.resize(image_preview,
                                   [int(image_preview.shape[0] * resize_scale),
                                    int(image_preview.shape[1] * resize_scale)])
    else:
        resize_scale = 1

    if show:
        cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Input Image', get_coordinate_callback)
        cv2.imshow('Input Image', image_preview)
        cv2.waitKey(0)

    if center_by_click != [-1, -1]:
        center = [int(center_by_click[0] / resize_scale), int(center_by_click[1] / resize_scale)]

    print(f'center = {center}')

    geo_calibration = True
    if geo_calibration:
        global exact_coordinates
        if exact_coordinates == [-1, -1]:
            geo_transform_l = list(geo_transform)
            exact_coordinates[0] = geo_transform_l[0] + geo_transform[1] * (center[0] - int(gap_size / 2))
            exact_coordinates[1] = geo_transform_l[3] + geo_transform[5] * (center[1] - int(gap_size / 2))
        else:
            center[0] = int((exact_coordinates[0] - geo_transform[0]) / geo_transform[1] + int(gap_size / 2))
            center[1] = int((exact_coordinates[1] - geo_transform[3]) / geo_transform[5] + int(gap_size / 2))
            print(f'new_center = {center}')

    if output_format == 'jpg':
        image = image / np.max(image)
        image *= 255
    window = generate_center_based_window(image.shape, center, crop_size)

    out_image = copy_image(image, crop_size, window, fill_value)

    print(f'image.shape = {image.shape}')
    print(f'out_image.shape = {out_image.shape}')

    if show:
        cv2.namedWindow('Out Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Out Image', out_image)
        cv2.waitKey(0)

    export_image(out_image, output_path, file_name_stem, window, crop_size=crop_size, gap_size=gap_size,
                 geo_transform=geo_transform, projection=projection, **kwargs)


def batch_crop_image_by_center(input_path: str, *args, **kwargs):
    image_name_list = [image_name for image_name in os.listdir(input_path)
                       if os.path.splitext(image_name)[-1].replace('.', '') in support_file_list]
    for image_name in tqdm(image_name_list):
        crop_image_by_center(f'{input_path}/{image_name}', *args, **kwargs)


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('config', help='config.yml')
    _args = _parser.parse_args()
    return _args


if __name__=="__main__":
    args = parse_args()
    config_file = args.config
    input_path, output_path, output_format, mode, center, crop_size, gap_size = read_config(config_file)

    if not os.path.isabs(input_path):
        current_path = os.path.abspath(__file__)
        input_path = os.path.join(current_path, input_path)
    if output_path is None:
        output_path = f'{input_path}/output'
    create_folder(output_path)

    if mode == 'center':
        if os.path.isdir(input_path):
            # path
            batch_crop_image_by_center(input_path, output_path, center, crop_size,
                                       show=True, output_format=output_format, jpg_quality=100)
        elif os.path.isfile(input_path):
            crop_image_by_center(input_path, output_path, center, crop_size,
                                 show=False, output_format=output_format, jpg_quality=100)
    elif mode == 'grid':
        if os.path.isdir(input_path):
            # path
            batch_crop_image_by_grid(input_path, output_path, crop_size, gap_size,
                                     show=False, output_format=output_format, jpg_quality=100, fill_value=(0, 0, 0))
        elif os.path.isfile(input_path):
            crop_image_by_grid(input_path, output_path, crop_size, gap_size,
                               show=True, output_format=output_format, jpg_quality=100, fill_value=(0, 0, 0))
    