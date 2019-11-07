import os
import sys
import pathlib
import pandas as pd
import json

import numpy as np
import cv2
import scipy
import scipy.io
from sklearn.model_selection import train_test_split

np.random.seed(seed=42)


def load_polygons(directory):
    """
    Load polygons from polygons.mat file.

    Args:
        (Path): pathlib Path object of directory to load samples from.

    Returns
    """
    # Load polygons file
    annotation_path = directory.joinpath('polygons.mat')
    mat = scipy.io.loadmat(annotation_path.resolve())
    # Load polygons data structure
    polygons = mat['polygons'][0]
    images_boxes = [
        get_boxes(polygons, frame_idx) for frame_idx in range(polygons.shape[0])
    ]
    return images_boxes


def get_boxes(polygons, frame_idx):
    """
    Get all bounding boxes belonging to a single image.

    Args:
        polygons (ndarray): Numpy array containing bounding boxes for each image in a directory
            extracted from .mat file struct. Image bounding boxes should follow image order.
        frame_idx (int): Index of image in folder (when sorted alphabetically).

    Returns:
        [(float, float, float, float)] List of bounding boxes belonging to a sigle image.
        Bounding box is represented as (ymin, xmin, ymax, ymax).
    """
    frame_polygons = polygons[frame_idx]
    boxes_list = []
    i = 0
    while True:
        try:
            poly = frame_polygons[i]
        except IndexError:
            break
        if poly.shape[1] == 2:
            xs, ys = zip(*[(int(poly[ci][0]), int(poly[ci][1])) for ci in range(poly.shape[0])])
            boxes_list.append((min(ys), min(xs), max(ys), max(xs)))
        i += 1
    return boxes_list

# Get all samples for each directory
def get_path_boxes(directory):
    """
    Get path and boxes represented as string.

    Args:
        directory (Path): pathlib Path object of directory to load samples from.

    Returns:
        [(str, str)]. List of tuple of (path, boxes as json)
    """
    images_boxes = load_polygons(directory)
    return [
        (path.absolute(), json.dumps(boxes_list)) for boxes_list, path
        in zip(images_boxes, sorted(directory.glob('*.jpg'))) if boxes_list]


def add_to_train_test(directory, df_train, df_test):
    """
    Add path and boxes to the training and testing dataframes.

    Args:
        directory (Path): pathlib Path object of directory to load samples from.
        df_files_boxes (DataFrame): Pandas dataframe to save path and boxes to.
    """
    data = get_path_boxes(directory)
    df_data = pd.DataFrame.from_records(data, columns=['path', 'boxes'])
    df_data_train, df_data_test = train_test_split(
        df_data, shuffle=False, test_size=0.1, random_state=42)
    df_train = pd.concat([df_train, df_data_train], ignore_index=True)
    df_test = pd.concat([df_test, df_data_test], ignore_index=True)
    return df_train, df_test

if __name__=='__main__':
    DOWNLOAD_DIR = '../data/egohands/'
    DATASET_PATH = os.path.join(DOWNLOAD_DIR, '_LABELLED_SAMPLES')
    TRAINING_FILE = '../data/egohands/train_files.csv'
    TESTING_FILE = '../data/egohands/test_files.csv'

    sample_directories = [f for f in pathlib.Path(DATASET_PATH).iterdir() if f.is_dir()]

    # Create empty dataframe
    df_train = pd.DataFrame({'path' : [], 'boxes':[]}, columns=['path', 'boxes'])
    df_test = pd.DataFrame({'path' : [], 'boxes':[]}, columns=['path', 'boxes'])
    # Add samples each directory to the dataframes
    for directory in sample_directories:
        df_train, df_test  = add_to_train_test(directory, df_train, df_test)
    # Save dataframe
    df_train.to_csv(TRAINING_FILE)
    df_test.to_csv(TESTING_FILE)
