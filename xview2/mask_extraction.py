import warnings
warnings.filterwarnings("ignore")

from shapely import wkt

import os
os.environ["PROJ_LIB"]="/home/wwymak/anaconda3/envs/fastai-solaris/share/proj"

# import solaris as sol


import geopandas as gpd
import numpy as np
import cv2
from functools import partial

from fastai.imports import *
from fastai.vision import *
from fastai.metrics import dice
from fastai.callbacks import *
import ujson as json
from joblib import Parallel, delayed
import torch.nn.functional as F
import torch
import functools, traceback

os.environ['FASTAI_TB_CLEAR_FRAMES']="1"


def parse_json(label_json):
    df_xy = pd.DataFrame([{'id': x['properties']['uid'], 'feature': x['properties']['feature_type'],
                           'damage': x['properties'].get('subtype', 'no-damage'),
                           'geometry_pixel': x['wkt']} for x in label_json['features']['xy']])
    df_lnglat = pd.DataFrame(
        [{'id': x['properties']['uid'], 'geometry_lnglat': x['wkt'], } for x in label_json['features']['lng_lat']])
    if len(df_xy) == 0:
        return
    df_xy.set_index('id', inplace=True)

    df_xy['damage_cls'] = df_xy['damage'].map({
        'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4, 'un-classified': 5
    })
    df_xy.geometry_pixel = df_xy.geometry_pixel.apply(wkt.loads)
    df_lnglat.set_index('id', inplace=True)
    label_df = df_xy.merge(df_lnglat['geometry_lnglat'], left_index=True, right_index=True)
    label_gdf = gpd.GeoDataFrame(label_df, geometry='geometry_pixel')
    label_gdf['centroid'] = label_gdf.geometry_pixel.centroid
    return label_gdf

def create_mask(json_file, data_dir, mask_dir, tile_size=1024):
    with open(data_dir/"train"/"labels"/json_file, 'r') as f:
        label = json.load(f)
    label_gdf = parse_json(label)
    if label_gdf is None or len(label_gdf) == 0:
        return
    fb_mask = sol.vector.mask.df_to_px_mask(df=label_gdf, geom_col="geometry_pixel",
                                         channels=['footprint', 'boundary','boundary'],
                                         shape=(tile_size,tile_size),
                                         boundary_width=3, boundary_type='inner')
    PIL.Image.fromarray(fb_mask).save(mask_dir/(json_file.replace('.json', '.png')))

def create_small_tiles_masks(mask_filepath, save_dir):
    img_id = mask_filepath.name.replace('.png', '')
    if not mask_filepath.exists():
        return
    mask_im = PIL.Image.open(mask_filepath)

    # no point in making smaller tiles with a masks that has no building int it
    if mask_im.getextrema()[1] == 0:
        return

    #     box = left, upper, right, and lower pixe

    boxes = [
        (0, 0, 512, 512),
        (0, 256, 512, 768),
        (0, 512, 512, 1024),
        (256, 0, 768, 512),
        (256, 256, 768, 768),
        (256, 512, 768, 1024),
        (512, 0, 1024, 512),
        (512, 256, 1024, 768),
        (512, 512, 1024, 1024)
    ]

    mask_crops = [mask_im.crop(box) for box in boxes]

    for i, mask_crop in enumerate(mask_crops):
        mask_max = np.array([x for x in mask_crop.getextrema()]).max()

        # only save crops that have buildings to help with training
        if mask_max > 0:
            mask_crop.save(save_dir / f"mask_{img_id}_{i}.png")

def create_small_tiles(img_filepath, mask_filepath, im_id, save_dir_rgb, save_dir_mask):
    if not mask_filepath.exists():
        return
    img_rgb = PIL.Image.open(img_filepath)
    mask_im = PIL.Image.open(mask_filepath)

    # no point in making smaller tiles with a masks that has no building int it
    if mask_im.getextrema()[1] == 0:
        return

    #     box = left, upper, right, and lower pixe

    boxes = [
        (0, 0, 512, 512),
        (0, 256, 512, 768),
        (0, 512, 512, 1024),
        (256, 0, 768, 512),
        (256, 256, 768, 768),
        (256, 512, 768, 1024),
        (512, 0, 1024, 512),
        (512, 256, 1024, 768),
        (512, 512, 1024, 1024)
    ]

    mask_crops = [mask_im.crop(box) for box in boxes]
    im_crops = [img_rgb.crop(box) for box in boxes]

    for i, (mask_crop, im_crop) in enumerate(zip(mask_crops, im_crops)):
        mask_max = np.array([x[1] for x in mask_crop.getextrema()]).max()

        # only save crops that have buildings to help with training
        if mask_max > 0:
            mask_crop.save(save_dir_mask / f"mask_{im_id}_{i}.png")
            im_crop.save(save_dir_rgb / f"rgb_{im_id}_{i}.png")


if __name__ == "__main__":
    data_dir = Path("/media/wwymak/Storage/xView2")
    mask_dir = data_dir / "mask_full_size"
    mask_dir.mkdir(exist_ok=True)

    train_images_crops = data_dir / "train_crops"
    train_mask_crops = data_dir / "mask_crops"
    train_images_crops.mkdir(exist_ok=True)
    train_mask_crops.mkdir(exist_ok=True)
    undamaged_fnames = [fname.name for fname in (data_dir / "train" / "labels").ls() if 'pre_disaster' in fname.name]

    _ = Parallel(n_jobs=16)(delayed(create_mask)(f, data_dir, mask_dir) for f in undamaged_fnames)

    create_tile = partial(create_small_tiles, save_dir_rgb=train_images_crops, save_dir_mask=train_mask_crops)

    img_filepaths = (data_dir / "train" / "images").ls()
    mask_filepaths = [mask_dir / f.name for f in img_filepaths]
    img_ids = [f.name.replace('png', '') for f in img_filepaths]
    # [create_tile(img_filepath, mask_filepath, im_id) for (img_filepath, mask_filepath, im_id) in zip(img_filepaths, mask_filepaths, img_ids )]
    _ = Parallel(n_jobs=14)(delayed(create_tile)(img_filepath, mask_filepath, im_id) \
                            for (img_filepath, mask_filepath, im_id) in zip(img_filepaths, mask_filepaths, img_ids))
    # fro