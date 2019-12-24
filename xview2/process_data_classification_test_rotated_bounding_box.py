from PIL import Image
import PIL
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import os
from pathlib import Path
import math
import random
from joblib import Parallel, delayed
import logging
import json
import cv2
import datetime

import shapely
from shapely import wkt
from shapely.geometry import Polygon
from collections import defaultdict
from sklearn.model_selection import train_test_split
from mask_extraction import parse_json

logging.basicConfig(level=logging.INFO)


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

def process_one_image(segmentation_results_df, data_dir, image_crops_dir,img_id):
    img_polygons_df = segmentation_results_df[segmentation_results_df['img_id'] == img_id]
    post_image_fpath = data_dir / "test" / "images" / f"test_post_{img_id.replace('test_localization_', '').replace('_prediction.png', '')}.png"
    if len(img_polygons_df) == 0:
        return
    img_array = np.array(PIL.Image.open(post_image_fpath))
    for r in img_polygons_df.iterrows():
        row = r[1]
        poly_uuid = row["polygon_id"]
        polygon_pts = np.array(list(row.geometry.exterior.coords))
        poly_img = PIL.Image.fromarray(process_img(img_array, polygon_pts))
        poly_img.save(image_crops_dir / f"{poly_uuid}.png")

def process_data(data_dir, image_crops_dir):
    """Process Raw Data into
        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.
        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form
    """

    buildings_segmentation_data = pd.read_csv(data_dir/"test_polygons.csv")
    buildings_segmentation_data['geometry'] = buildings_segmentation_data['geometry'].apply(lambda x: wkt.loads(x))
    img_ids = buildings_segmentation_data['img_id'].unique()

    _ = Parallel(n_jobs=14)(delayed(process_one_image)(buildings_segmentation_data, data_dir, image_crops_dir, img_id) \
                            for img_id in tqdm(img_ids))

def main():
    data_dir = Path("/media/wwymak/Storage/xView2")
    image_crops_dir = data_dir / "classification_crops_rotated_bbox_test"
    image_crops_dir.mkdir(exist_ok=True)
    process_data(data_dir,  image_crops_dir)
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()