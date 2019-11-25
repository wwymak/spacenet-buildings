from pathlib import Path
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

def f1(y_pred:Tensor, y_true:Tensor):
    eps=1e-10
    def to_onehot(indices, num_classes):
        """Convert a tensor of indices of any shape `(N, ...)` to a
        tensor of one-hot indicators of shape `(N, num_classes, ...) and of type uint8. Output's device is equal to the
        input's device`.
        """
        onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:],
                             dtype=torch.uint8,
                             device=indices.device)
        return onehot.scatter_(1, indices.unsqueeze(1), 1)
    def recall(y_pred,y_true):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(y_true)
        recall = true_positives / (possible_positives + eps)
        return recall

    def precision(y_pred,y_true):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives +eps)
        return precision

    y_true = to_onehot(y_true.view(-1), num_classes=4)
    precision = precision(y_pred,y_true)
    recall = recall(y_pred,y_true)
    return 2*((precision*recall)/(precision+recall+eps))


from cv2 import fillPoly, imwrite


def create_image(img_id, output_path, test_results):
    mask_img = np.zeros((1024, 1024, 1), np.uint8)
    img_polys = test_results[test_results.img_id == img_id]
    if len(img_polys) > 0:
        for r in img_polys.iterrows():
            row = r[1]
            poly_np = np.array(row.geometry.coords, np.int32)
            fillPoly(mask_img, [poly_np], row['damage_cls'])

    img_id.replace('post', 'damage').replace('pre', 'localization').replace('.png', '_prediction.png')

    imwrite(str(output_path / img_id), mask_img)
    return mask_img


if __name__ == "__main__":
    data_dir = Path("/media/wwymak/Storage/xView2")
    mask_dir = data_dir / "mask_full_size"

    train_images_crops = data_dir / "train_crops"
    # train_mask_crops = data_dir/"mask_crops"
    train_mask_crops = data_dir / "mask_crops_single_channel"
    test_images_crops = data_dir / "test_crops"
    test_mask_crops = data_dir / "test_mask_crops_single_channel"

    classifcation_crop_dir = data_dir / "classification_crops"

    labels = pd.read_csv(data_dir / "test_polygons.csv")
    labels["crop_filename"] = labels.polygon_id.apply(lambda x: f"{x}.png")
    damage_crops_test_folder = data_dir / "classification_crops_test"
    classification_labels = pd.read_csv(data_dir / "train.csv")
    size = 64
    bs = 32
    src = (ImageList
           .from_df(classification_labels, path=classifcation_crop_dir, cols=['crop_filename'])
           .split_by_rand_pct(0.2)
           .label_from_df(cols='label'))

    data = (src
            .transform(get_transforms(do_flip=True,
                                      flip_vert=True,
                                      max_rotate=180,
                                      max_zoom=1.2,
                                      max_lighting=0.5,
                                      max_warp=0.2,
                                      p_affine=0.75,
                                      p_lighting=0.75), size=size, tfm_y=False)
            .add_test_folder(damage_crops_test_folder)
            .databunch(bs=bs)
            .normalize(imagenet_stats))

    predictions = learn.get_preds(DatasetType.Test)