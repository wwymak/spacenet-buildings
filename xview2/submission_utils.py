import re
import numpy as np
import PIL
import os
from pathlib import Path


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

def load_and_validate_image( path):

    valid_fname_re1 = r"test_damage_([\d+]{5})_prediction.png"
    valid_fname_re2 = r"test_localization_([\d+]{5})_prediction.png"
    assert re.match(valid_fname_re1, path.name) or re.match(valid_fname_re2, path.name), f"{path.name} is of wrong filename format"
    assert path.is_file(), f"file '{path}' does not exist or is not a file"
    img = np.array(PIL.Image.open(path))
    assert img.dtype == np.uint8, f"{path.name} is of wrong format {img.dtype} - should be np.uint8"
    assert set(np.unique(img)) <= {0,1,2,3,4}, f"values must ints 0-4, found {np.unique(img)}, path: {path}"
    assert img.shape == (1024,1024), f"{path} must be a 1024x1024 image"
    return img


def validate_submission_folder(submission_folder_path):
    num_damaged = len([x for x in os.listdir(submission_folder_path) if 'damage' in x])
    num_localisation = len([x for x in os.listdir(submission_folder_path) if 'localisation' in x])
    assert num_damaged == 933
    assert num_localisation == 933

    _ = [load_and_validate_image(x) for x in submission_folder_path.ls()]
