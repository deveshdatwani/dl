import os
from glob import glob
import scipy.io
from skimage.io import imread
import numpy as np
from skimage.segmentation import find_boundaries
from matplotlib import pyplot as plt


class BSDS500Dataset:
    def __init__(self, root_dir, split='train'):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.gt_dir = os.path.join(root_dir, 'ground_truth', split)
        self.img_files = sorted(glob(os.path.join(self.img_dir, '*.jpg')))
        self.gt_files = sorted(glob(os.path.join(self.gt_dir, '*.mat')))
        assert len(self.img_files) == len(self.gt_files), "Image/Ground truth mismatch"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = imread(self.img_files[idx])
        mat = scipy.io.loadmat(self.gt_files[idx])
        gt_list = mat['groundTruth'][0]  
        gts = [gt_item[0,0]['Segmentation'] for gt_item in gt_list]
        gts = np.stack(gts, axis=0)       
        return img, gts