from fastai.imports import *
from fastai.vision import *
from joblib import Parallel, delayed



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


def combine_images(img_id, test_mask_crops_dir, test_masks_dir):
    full_img = np.zeros((1024, 1024))

    for i in range(9):
        arr = PIL.Image.open(test_mask_crops_dir / f"{img_id}_{i}.png").resize((512,512))
        start_index_row, start_index_col = boxes[i][1], boxes[i][0]
        end_index_row, endindex_col = boxes[i][3], boxes[i][2]
        full_img[start_index_row:end_index_row, start_index_col: endindex_col] = arr

    img = PIL.Image.fromarray(full_img.astype(np.uint8))
    fname = f"test_localization_{img_id.replace('mask_test_pre_', '')}_prediction.png"
    img.save(test_masks_dir / fname, mode="L")


if __name__=="__main__":
    data_dir = Path("/media/wwymak/Storage/xView2")
    mask_dir = data_dir / "mask_full_size"

    train_images_crops = data_dir / "train_crops"
    train_mask_crops = data_dir / "mask_crops_single_channel"
    test_images_crops = data_dir / "test_crops"
    test_mask_crops = data_dir / "test_mask_crops_single_channel"
    test_mask_edt_crops = data_dir / "test_mask_crops_edt"
    test_masks = data_dir / "test_masks"
    test_masks_edt = data_dir / "test_masks_edt"
    test_masks_edt.mkdir(exist_ok=True)

    img_ids = list(set(['_'.join(x.name.split('_')[:-1]) for x in test_mask_edt_crops.ls()]))
    img_ids = [x for x in img_ids if 'pre_' in x]

    _ = Parallel(n_jobs=14)(delayed(combine_images)(img_id, test_mask_edt_crops, test_masks_edt) for img_id in img_ids)
