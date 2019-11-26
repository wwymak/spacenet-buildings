from solaris.vector.mask import mask_to_poly_geojson
import numpy as np
from uuid import uuid4
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import geopandas as gpd
import PIL


def filter_files(files, include=[], exclude=[]):
    for incl in include:
        files = [f for f in files if incl in f.name]
    for excl in exclude:
        files = [f for f in files if excl not in f.name]
    return sorted(files)

def ls(x, recursive=False, include=[], exclude=[]):
    if not recursive:
        out = list(x.iterdir())
    else:
        out = [o for o in x.glob('**/*')]
    out = filter_files(out, include=include, exclude=exclude)
    return out

Path.ls = ls


def create_label_file(img_path):
    img_array = np.array(PIL.Image.open(img_path))
    img_id = img_path.name.replace('.png', '')
    gdf = mask_to_poly_geojson(img_array, simplify=True)
    idx = [str(uuid4()) for _ in range(len(gdf))]

    gdf["polygon_id"] = idx
    gdf["img_id"] = img_path.name

    return gdf

if __name__ == "__main__":
    data_dir = Path("/media/wwymak/Storage/xView2")
    mask_dir = data_dir / "mask_full_size"

    train_images_crops = data_dir / "train_crops"
    train_mask_crops = data_dir / "mask_crops_single_channel"
    test_images_crops = data_dir / "test_crops"
    test_mask_crops = data_dir / "test_mask_crops_single_channel"
    test_masks = data_dir / "test_masks"

    gdf_array = Parallel(n_jobs=14)(delayed(create_label_file)(img_id) for img_id in tqdm(test_masks.ls()))
    gdf_array = pd.concat(gdf_array)
    gdf_array.to_csv(data_dir/"test_polygons.csv", index=False)


