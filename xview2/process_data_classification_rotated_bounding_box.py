"""
crops images to a rotated bounding box for damage classifcation
"""
from PIL import Image
import PIL
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import os
from pathlib import Path
from joblib import Parallel, delayed
import random
import argparse
import logging
import json
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
from sklearn.model_selection import train_test_split
from mask_extraction import parse_json

logging.basicConfig(level=logging.INFO)

# Configurations
NUM_WORKERS = 4
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_STEP = 150

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def process_img(img_array, polygon_pts, buffer_px=10):
    """Process Raw Data into
            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.
            Returns:
                numpy array: .
    """
    src = img_array.copy()

    center, size, theta = cv2.minAreaRect(polygon_pts[:, None, :].astype(np.int32))
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(lambda x: int(x + buffer_px), size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

def process_data(data_dir, image_crops_dir, output_csv_path,):
    """Process Raw Data into
        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.
        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form
    """
    data = []

    images = [data_dir / "train" / "images" / x for x in os.listdir(data_dir / "train" / "images")]
    labels = [data_dir / "train" / "labels" / (x.name.replace('png', 'json')) for x in images]

    print(len(images), len(labels))

    for label_file, img_file in tqdm(zip(labels, images)):
        with open(label_file, 'r') as f:
            label = json.load(f)
        label_gdf = parse_json(label)
        if label_gdf is None or len(label_gdf) == 0:
            continue
        img_array = np.array(Image.open(data_dir / "train" / "images"/img_file))
        for r in label_gdf.iterrows():
            row = r[1]
            poly_uuid = r[0]
            damage_type = damage_intensity_encoding[row['damage']]
            polygon_pts = np.array(list(row.geometry_pixel.exterior.coords))
            poly_img = PIL.Image.fromarray(process_img(img_array, polygon_pts))
            poly_img.save(image_crops_dir / f"{poly_uuid}.png")

            data.append({'uuid': poly_uuid, 'img_id': img_file.name, 'label': damage_type})

    output_train_csv_path = output_csv_path/ "train.csv"

    df = pd.DataFrame(data)
    df.to_csv(output_train_csv_path)

    return df

def parse_one_pre_post_set(label_file, img_file_pre, img_file_post, image_crops_dir_pre, image_crops_dir_post):
    data = []
    with open(label_file, 'r') as f:
        label = json.load(f)
    label_gdf = parse_json(label)
    if label_gdf is None or len(label_gdf) == 0:
        return
    img_array_pre = np.array(Image.open( img_file_pre))
    img_array_post = np.array(Image.open(img_file_post))
    for r in label_gdf.iterrows():
        row = r[1]
        poly_uuid = r[0]
        damage_type = damage_intensity_encoding[row['damage']]
        polygon_pts = np.array(list(row.geometry_pixel.exterior.coords))

        poly_img = PIL.Image.fromarray(process_img(img_array_pre, polygon_pts))
        poly_img.save(image_crops_dir_pre/ f"{poly_uuid}.png")

        poly_img_post = PIL.Image.fromarray(process_img(img_array_post, polygon_pts))
        poly_img_post.save(image_crops_dir_post / f"{poly_uuid}.png")

        data.append({'uuid': poly_uuid, 'img_id_pre': img_file_pre.name, 'img_id_post': img_file_post.name,
                     'label': damage_type})
    return data


def process_data_pre_post_crop_pairs(data_dir, image_crops_dir_pre, image_crops_dir_post, output_csv_path,):
    """Process Raw Data into
        a pair of images corresponding to the pre and post of a building
    """

    images_post = [data_dir / "train" / "images"/x for x in os.listdir(data_dir / "train" / "images") if 'post_disaster' in x]
    images_pre = [data_dir / "train" / "images"/x.name.replace('post', 'pre') for x in images_post]
    labels = [data_dir / "train" / "labels"/(x.name.replace('png', 'json')) for x in images_post]

    all_data = Parallel(n_jobs=14)(delayed(parse_one_pre_post_set)(label_file, img_file_pre,
        img_file_post, image_crops_dir_pre, image_crops_dir_post) for
        label_file, img_file_pre, img_file_post in  tqdm(zip(labels, images_pre, images_post)))

    data_list_flat = []
    for sublist in all_data:
        if sublist is not None:
            for item in sublist:
                data_list_flat.append(item)
    output_train_csv_path = output_csv_path/ "train_pre_post_rotated_bbox.csv"



    df = pd.DataFrame(data_list_flat)
    df["crop_filename"] = df["uuid"].apply(lambda x: f"{x}.png")
    df.to_csv(output_train_csv_path, index=False)

    return df


def main():

    data_dir = Path("/media/wwymak/Storage/xView2")
    image_crops_dir_post = data_dir / "classification_crops_post_rotated_bbox"
    image_crops_dir_pre = data_dir / "classification_crops_pre_rotated_bbox"
    image_crops_dir_post.mkdir(exist_ok=True)
    image_crops_dir_pre.mkdir(exist_ok=True)
    process_data_pre_post_crop_pairs(data_dir, image_crops_dir_pre, image_crops_dir_post, data_dir)


if __name__ == '__main__':
    main()