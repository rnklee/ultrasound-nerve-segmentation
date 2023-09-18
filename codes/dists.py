import sys, os
import re

import cv2

import pandas as pd
import numpy as np

import itertools
from itertools import combinations, permutations, product

from tqdm import tqdm
from functools import partial

from skimage.metrics import structural_similarity as ssim
from skimage.util import compare_images, view_as_blocks, view_as_windows

from PIL import Image, ImageOps


def ssim_dist(img1, img2):
    return 1-ssim(img1, img2, data_range=255)


def euclidean_dist(img1, img2):
    u, v = img1.flatten().astype('float'), img2.flatten().astype('float')
    return scipy.spatial.distance.euclidean(u,v)  


def block_dist(img1, img2, block_shape=(28,29), measure=euclidean_dist):
    '''
    Same idea as the dist used in https://www.kaggle.com/code/chefele/animated-images-with-outlined-nerve-area.
    Segment the images into blocks, compute the mean brightness for each block,
    and then compute their Euclidean distance.
    '''
    blocks1, blocks2= view_as_blocks(img1, block_shape), view_as_blocks(img2, block_shape)
    means1, means2 = blocks1.mean(axis=(2,3)), blocks2.mean(axis=(2,3))
    return measure(means1, means2)


def compute_distance_matrix(images, path, from_files=True, measure='block'):
    '''
    Given all image files from one subject, returns the distance matrix.
    distance_matrix[i,j] = distance (measure by 1-ssim) between (i+1)-th sample and (j+1)-th sample.
    '''
    measure_dict = {
        'ssim': lambda x,y: 1-ssim(x,y),
        'block': block_dist
    }
    if isinstance(measure, str):
        measure = measure_dict[measure]

    N = len(images)
    distance_matrix = np.zeros((N,N))
    if from_files:
        combs = itertools.combinations(range(N), 2)
        for (i1,i2) in tqdm(combs):
            o1, o2 = images[i1], images[i2]
            img1 = cv2.imread(path + o1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path + o2, cv2.IMREAD_GRAYSCALE)
            d = measure(img1, img2)

            distance_matrix[i1, i2] = d
            distance_matrix[i2, i1] = d

    else:
        combs = itertools.combinations(range(N), 2)
        for (i1,i2) in tqdm(combs):
            img1, img2 = images[i1], images[i2]
            d = measure(img1, img2)

            distance_matrix[i1, i2] = d
            distance_matrix[i2, i1] = d

    return distance_matrix