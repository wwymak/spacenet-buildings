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

    images = [Path(x) for x in os.listdir(data_dir / "train" / "images") if 'post_disaster' in x]
    labels = [data_dir / "train" / "labels"/(x.name.replace('png', 'json')) for x in images]
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
            poly_img = PIL.Image.fromarray(process_img(img_array, polygon_pts, 0.8))
            poly_img.save(image_crops_dir / f"{poly_uuid}.png")

            data.append({'uuid': poly_uuid, 'img_id': img_file.name, 'label': damage_type})

    output_train_csv_path = output_csv_path/ "train.csv"

    df = pd.DataFrame(data)
    df.to_csv(output_train_csv_path)

    return df


def main():
    data_dir = Path("/media/wwymak/Storage/xView2")
    image_crops_dir = data_dir / "classification_crops"
    image_crops_dir.mkdir(exist_ok=True)
    process_data(data_dir,  image_crops_dir,  data_dir)
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()