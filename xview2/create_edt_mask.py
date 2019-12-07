import warnings
warnings.filterwarnings("ignore")

import skimage

from tqdm import tqdm
import numpy as np
import cv2
from functools import partial

from fastai.imports import *
from fastai.vision import *
from fastai.metrics import dice
from fastai.callbacks import *

from joblib import Parallel, delayed

from scipy import ndimage

data_dir = Path("/media/wwymak/Storage/xView2")
mask_dir = data_dir /"mask_full_size"
mask_dir_edt = data_dir /"mask_full_size_edt"

train_images_crops = data_dir/"train_crops"
# train_mask_crops = data_dir/"mask_crops"
train_mask_crops = data_dir/"mask_crops_single_channel"
label_dir = data_dir/"train"/"labels"

def create_mask_edt(mask_filepath):
    sample_mask = np.array(PIL.Image.open(mask_filepath))[:, :, 0] / 255
    inverted_sample_mask = 1 - sample_mask
    transformed = ndimage.distance_transform_edt(sample_mask)
    trasformed_inverted = ndimage.distance_transform_edt(inverted_sample_mask) * (-1)

    combined = transformed + trasformed_inverted

    bins = np.arange(-20, 20, 5)
    combined_binned = np.digitize(combined, bins, right=False)
    PIL.Image.fromarray(combined_binned.astype(np.uint8)).save(mask_dir_edt/mask_filepath.name)

if __name__ == "__main__":
    pre_image_masks = [x for x in mask_dir.ls() if 'pre_' in x.name]
    mask_dir_edt.mkdir(exist_ok=True)
    _ = Parallel(n_jobs=14)(delayed(create_mask_edt)(mask_filepath) for mask_filepath in tqdm(pre_image_masks))
