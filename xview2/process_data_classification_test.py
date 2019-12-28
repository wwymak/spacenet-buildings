"""
xView2
Copyright 2019 Carnegie Mellon University. All Rights Reserved.
NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
This material has been approved for public release and unlimited distribution.
Please see Copyright notice for non-US Government use and distribution.
This Software includes and/or makes use of the following Third-Party Software subject to its own license:
1. Matterport/Mask_RCNN - MIT Copyright 2017 Matterport, Inc.
   https://github.com/matterport/Mask_RCNN/blob/master/LICENSE
"""
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


def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into
            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.
            Returns:
                numpy array: .
    """

    height, width, _ = img_array.shape

    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    # Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]

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
        poly_img = PIL.Image.fromarray(process_img(img_array, polygon_pts, 0.8))
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

    buildings_segmentation_data = pd.read_csv(data_dir/"test_polygons_edt.csv")
    buildings_segmentation_data['geometry'] = buildings_segmentation_data['geometry'].apply(lambda x: wkt.loads(x))
    img_ids = buildings_segmentation_data['img_id'].unique()

    img_id = img_ids[0]
    #process_one_image(buildings_segmentation_data, data_dir, image_crops_dir, img_id)
    _ = Parallel(n_jobs=14)(delayed(process_one_image)(buildings_segmentation_data, data_dir, image_crops_dir, img_id) \
                            for img_id in tqdm(img_ids))

def main():
    data_dir = Path("/media/wwymak/Storage/xView2")
    image_crops_dir = data_dir / "classification_crops_test"
    image_crops_dir.mkdir(exist_ok=True)
    process_data(data_dir,  image_crops_dir)
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()