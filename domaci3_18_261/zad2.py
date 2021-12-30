# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:16:05 2021

@author: mm180261d
"""

from pylab import *
import skimage
from skimage import io
from skimage import filters
from skimage import color
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import math

def make_hist(img_in: np.array) -> np.array:
    num_hist = np.zeros(256, dtype=int)
    for i in range(0, 256):
        # number of elements that are equal to i
        num_hist[i] = np.sum(img_in == i)
    # normalizing the histogram
    return num_hist/img_in.size

def coin_mask(img_in: np.array) -> np.array:
    # img = filters.gaussian(img_in, sigma=0.5, truncate=6)
    
    img_hsv = color.rgb2hsv(img_in)
    plt.figure(figsize=(12, 9), dpi=80)
    plt.imshow(img_hsv[:, :, 0])
    
    plt.figure(figsize=(12, 9), dpi=80)
    plt.imshow(img_hsv[:, :, 1])
    
    plt.figure(figsize=(12, 9), dpi=80)
    plt.imshow(img_hsv[:, :, 2])
    
    histogram_h, bin_h = np.histogram(img_hsv[:, :, 0].flatten(), bins=256, range=(0,1))
    plt.figure(figsize=(8,4), dpi=80);
    plt.plot(bin_h[0:-1], histogram_h)
    plt.title('Histogram');
    
    histogram_s, bin_s = np.histogram(img_hsv[:, :, 1].flatten(), bins=256, range=(0,1))
    plt.figure(figsize=(8,4), dpi=80);
    plt.plot(bin_s[0:-1], histogram_s)
    
    # 2 classes are separated (which we can see from the histogram)
    otsu_h = filters.threshold_otsu(img_hsv[:, :, 0])

    mask_h = img_hsv[:, :, 0] < otsu_h
    
    # when it comes to saturation otsu is not the best option,
    # because we can not see the exact separation between 
    # classes on histogram
    mask_s = img_hsv[:, :, 1] < 0.1
    
    plt.figure(figsize=(8,4), dpi=80);
    plt.imshow(mask_h*1.0, cmap='gray')
    
    plt.figure(figsize=(8,4), dpi=80);
    plt.imshow(mask_s*1.0, cmap='gray')
    
    mask = logical_or(mask_h, mask_s)
    
    plt.figure(figsize=(8,4), dpi=80);
    plt.imshow(mask*1.0, cmap='gray')
    mask1 = np.copy(mask)
    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            if mask[i, j] == True:
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        mask1[i+k, j+l] = True
    plt.figure(figsize=(8,4), dpi=80);
    plt.imshow(mask1*1.0, cmap='gray')
    return mask1

def bw_label(img_bin: np.array) -> np.array:
    label = np.zeros(img_bin.shape, dtype=int)
    last_ind = 1
    for_equal = []
    
    for i in range(1, img_bin.shape[0]-1):
        for j in range(1, img_bin.shape[1]-1):
            if img_bin[i, j]==True:
                if label[i-1, j-1] != 0:
                    label[i, j] = label[i-1, j-1]
                    
                if label[i-1, j] != 0 and label[i-1, j] != label[i,j]:
                    if label[i, j] != 0:
                        label[label==label[i-1, j]] = label[i, j]
                    else:
                        label[i, j] = label[i-1, j]
                        
                if label[i-1, j+1] != 0 and label[i-1, j+1] != label[i,j]:
                    if label[i, j] != 0:
                        label[label==label[i-1, j+1]] = label[i, j]
                    else:
                        label[i, j] = label[i-1, j+1]
                
                if label[i, j-1] != 0 and label[i, j-1] != label[i,j]:
                    if label[i, j] != 0:
                        label[label==label[i, j-1]] = label[i, j]
                    else:
                        label[i, j] = label[i, j-1]
                    
                if label[i, j] == 0:
                    label[i, j] = last_ind
                    last_ind = last_ind+1
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(label/(last_ind-1), cmap='gray')
    return label

def coin_classification(img_in: np.array) -> []:
    mask = coin_mask(img_in)
    label = bw_label(mask)
    
    maks = np.max(label)
    hist = np.zeros(maks, dtype=int)
    for i in range(1, maks+1):
        mask1 = (label==i)
        hist[i-1] = np.sum(mask1)
    mean = np.mean(hist[hist>0])
    
    print(mean)
    print(np.sum(hist>mean), np.sum(logical_and(hist>0, hist<mean)))

if __name__ == "__main__":
    img_in = imread('../sekvence/coins/coins9.jpg')
    plt.figure(figsize=(12, 9), dpi=80)
    plt.imshow(img_in)
    coin_classification(img_in)
    