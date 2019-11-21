import warnings
warnings.filterwarnings("ignore")

import glob

from osgeo import gdal
from shapely import wkt

import os
os.environ["PROJ_LIB"]="/home/wwymak/anaconda3/envs/solaris/share/proj"


import rasterio
from torch.utils.tensorboard import SummaryWriter
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import solaris as sol
import PIL
from tqdm import tqdm
import tifffile as sktif

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

def create_small_tiles(img_filepath, save_dir_rgb):
    img_rgb = PIL.Image.open(img_filepath)
    img_id = img_filepath.name.replace('.png', '')

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
    im_crops = [img_rgb.crop(box) for box in boxes]

    for i, im_crop in enumerate(im_crops):
        im_crop.save(save_dir_rgb / f"rgb_{img_id}_{i}.png")

if __name__ == "__main__":
    data_dir = Path("/media/wwymak/Storage/xView2")
    mask_dir = data_dir / "mask_full_size_single_channel"

    test_images_crops = data_dir / "test_crops"
    test_mask_crops = data_dir / "test_mask_crops_single_channel"
    test_images_crops.mkdir(exist_ok=True)
    test_mask_crops.mkdir(exist_ok=True)

    undamaged_fnames = [fname for fname in (data_dir / "test" / "images").ls() if '_pre_' in fname.name]
    mg_ids = [f.name.replace('.png', '') for f in undamaged_fnames]

    create_tile = partial(create_small_tiles, save_dir_rgb=test_images_crops)
    _ = Parallel(n_jobs=14)(delayed(create_tile)(img_filepath) for img_filepath in undamaged_fnames)
