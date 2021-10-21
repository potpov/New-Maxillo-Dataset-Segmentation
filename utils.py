from torch.utils.data import DistributedSampler
from scipy.ndimage import binary_fill_holes
import pathlib
import torchio as tio
import logging
import os
import numpy as np
import yaml
import sys
import torch
from tqdm import tqdm
import SimpleITK as sitk
import json
from scipy.linalg import norm


def set_logger(log_path=None):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if not log_path:
            # Logging to console
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
        else:
            # Logging to a file
            file_handler = logging.FileHandler(os.path.join(log_path))
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)


def load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'), yaml.FullLoader)

def resample(ctvol, is_label, original_spacing=.3, out_spacing=.4):
    original_spacing = (original_spacing, original_spacing, original_spacing)
    out_spacing = (out_spacing, out_spacing, out_spacing)

    ctvol_itk = sitk.GetImageFromArray(ctvol)
    ctvol_itk.SetSpacing(original_spacing)
    original_size = ctvol_itk.GetSize()
    out_shape = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    # Perform resampling:
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_shape)
    resample.SetOutputDirection(ctvol_itk.GetDirection())
    resample.SetOutputOrigin(ctvol_itk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    resampled_ctvol = resample.Execute(ctvol_itk)
    return sitk.GetArrayFromImage(resampled_ctvol)


def create_dataset(splits_todo, is_competitor, saving_dir):
    split_filepath = "path/to/splits.json"
    with open(split_filepath) as f:
        folder_splits = json.load(f)

    # for split in ['train', 'syntetic', 'val']:
    if is_competitor:
        base = '/path/to/data/SPARSE'
    else:
        base = "/path/to/data/DENSE"

    for split in splits_todo:
        dirs = [os.path.join(base, p) for p in
                folder_splits[split]]

        dataset = {'data': [], 'gt': []}
        for i, dir in tqdm(enumerate(dirs), total=len(dirs), desc=f"processing {split}"):
            gt_dir = os.path.join(dir, 'syntetic.npy') if is_competitor else os.path.join(dir, 'gt_alpha.npy')
            data_dir = os.path.join(dir, 'data.npy')

            image = np.load(data_dir)
            gt_orig = np.load(gt_dir)

            # rescale
            image = resample(image, is_label=False)
            gt = resample(gt_orig, is_label=True)

            # DICOM_MAX = 3095 if is_competitor else 2100
            DICOM_MAX = 2100
            DICOM_MIN = 0
            image = np.clip(image, DICOM_MIN, DICOM_MAX)
            image = (image.astype(float) + DICOM_MIN) / (DICOM_MAX + DICOM_MIN)  # [0-1] with shifting

            if split not in ["test", "val"]:
                s = tio.Subject(
                    data=tio.ScalarImage(tensor=image[None]),
                    label=tio.LabelMap(tensor=gt[None]),
                )

                grid_sampler = tio.inference.GridSampler(
                    s,
                    patch_size=(32, 32, 32),
                    patch_overlap=(10, 10, 10),
                )

                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
                for a in patch_loader:
                    image = a['data'][tio.DATA].squeeze().numpy()
                    gt = a['label'][tio.DATA].squeeze().numpy()
                    if np.sum(gt) != 0:
                        dataset['data'].append(image)
                        dataset['gt'].append(gt)
            else:  # do not cut volumes for testing - we do this at runtime
                dataset['data'].append(image)
                dataset['gt'].append(gt_orig)

        log_dir = pathlib.Path(os.path.join(saving_dir, split))
        log_dir.mkdir(parents=True, exist_ok=True)
        for partition in ['data', 'gt']:
            a = np.empty(len(dataset[partition]), dtype=object)
            for i in range(len(dataset[partition])):
                a[i] = dataset[partition][i]
            np.save(os.path.join(saving_dir, split, f'{partition}.npy'), a)
        print(f"split {split} completed. created {len(dataset['data'])} subvolumes")


def create_syntetic():
    data_dir = "path/toyour/SPARSE/npy_files"

    for folder in os.listdir(data_dir):
        print(f"processing {folder}")
        gt = np.load(os.path.join(data_dir, folder, "gt_sparse.npy"))
        example = np.zeros_like(gt)
        points = np.argwhere(gt == 1)
        splits = [
            points[points[:, -1] < gt.shape[-1] // 2],
            points[points[:, -1] > gt.shape[-1] // 2]
        ]
        for jj in range(2):
            points = splits[jj]
            points = points[np.lexsort((points[:, 2], points[:, 0], points[:, 1]))]
            for i in range(points.shape[0] - 2):
                # axis and radius

                p0 = np.array(points[i])
                p1 = np.array(points[i + 1])
                R = 1.6
                # vector in direction of axis
                v = p1 - p0
                # find magnitude of vector
                mag = norm(v)
                # unit vector in direction of axis
                v = v / mag
                # make some vector not in the same direction as v
                not_v = np.array([1, 0, 0])
                if (v == not_v).all():
                    not_v = np.array([0, 1, 0])
                # make vector perpendicular to v
                n1 = np.cross(v, not_v)
                # normalize n1
                n1 /= norm(n1)
                # make unit vector perpendicular to v and n1
                n2 = np.cross(v, n1)
                # surface ranges over t from 0 to length of axis and 0 to 2*pi
                t = np.linspace(0, mag, 100)
                theta = np.linspace(0, 2 * np.pi, 100)
                # use meshgrid to make 2d arrays
                t, theta = np.meshgrid(t, theta)
                # generate coordinates for surface
                Z, Y, X = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

                example[(Z+4).astype(int), Y.astype(int), X.astype(int)] = 1

        example = binary_fill_holes(example).astype(int)
        np.save(os.path.join(data_dir, folder, 'syntetic.npy'), example)


if __name__ == '__main__':

    # generate cicle expanded dataset - set your paths!
    create_syntetic()
    print("syntetic dataset has been created!")
    # generate training and syntetic datasets (32x32x32) and the test set (resampling to 0.3 voxel space)
    create_dataset(['train', 'syntetic', 'val', 'test'], is_competitor=True, saving_dir="saving_dir/sparse")
    create_dataset(['train', 'val', 'test'], is_competitor=False, saving_dir="saving_dir/dense")
    print("subvolumes for training have been created!")
