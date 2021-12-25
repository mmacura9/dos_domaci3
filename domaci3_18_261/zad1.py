# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:23:16 2021

@author: mm180261d
"""
from pylab import *
from skimage import io
from skimage import filters
from skimage import color
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import math

def canny_edge_detection(img_in: np.array, sigma: float, threshold_low: float, threshold_high: float) -> np.array:
    # filtering the input image
    img = filters.gaussian(img_in, sigma=sigma, truncate=3)
    plt.figure()
    io.imshow(img)
    
    # gradients
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8
    sobel_horizontal = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])/8
    
    gradient_vertical = ndimage.correlate(img, sobel_vertical)
    gradient_horizontal = ndimage.correlate(img, sobel_horizontal)
    # vertical gradient
    plt.figure()
    io.imshow(gradient_vertical, cmap='gray')
    
    # horizontal gradient
    plt.figure()
    io.imshow(gradient_horizontal, cmap='gray')
    
    magnitude = np.sqrt(gradient_vertical**2 + gradient_horizontal**2)
    angle = np.arctan2(gradient_vertical, gradient_horizontal)
    
    # magnitude
    plt.figure()
    io.imshow(magnitude, cmap='gray')
    
    # angle
    plt.figure()
    io.imshow(angle, cmap='gray')
    
    # quantization
    radian225 = math.radians(22.5)
    radian675 = math.radians(67.5)
    radian1125 = math.radians(112.5)
    radian1575 = math.radians(157.5)
    
    radian_225 = math.radians(-22.5)
    radian_675 = math.radians(-67.5)
    radian_1125 = math.radians(-112.5)
    radian_1575 = math.radians(-157.5)
    
    horizontal = ((angle > radian_225) * (angle < radian225)) + (angle < radian_1575) + (angle > radian1575)
    plt.figure()
    io.imshow(horizontal*1.)
    
    vertical = ((angle > radian675) * (angle < radian1125)) + ((angle < radian_675) * (angle > radian_1125))
    plt.figure()
    io.imshow(vertical*1.)
    
    diag45 = ((angle > radian1125) * (angle < radian1575)) + ((angle < radian_225) * (angle > radian_675))
    plt.figure()
    io.imshow(diag45*1.)
    
    diag_45 = ((angle > radian225) * (angle < radian675)) + ((angle < radian_1125) * (angle > radian_1575))
    plt.figure()
    io.imshow(diag_45*1.)
    
    img_horizontal = horizontal * magnitude
    edges_horizontal = img_horizontal
    for i in range(edges_horizontal.shape[0]):
        peaks, _ = find_peaks(edges_horizontal[i, :], height=threshold_low)
        arr = np.ones(edges_horizontal.shape[1], dtype=bool)
        arr[peaks] = False
        if len(peaks) != 0:
            edges_horizontal[i, arr] = 0
    plt.figure()
    io.imshow(edges_horizontal, cmap='gray')
    
    img_vertical = vertical * magnitude
    edges_vertical = img_vertical
    for i in range(edges_vertical.shape[1]):
        peaks, _ = find_peaks(edges_vertical[:, i], height=threshold_low)
        arr = np.ones(edges_vertical.shape[0], dtype=bool)
        arr[peaks] = False
        if len(peaks) != 0:
            edges_vertical[arr, i] = 0
    plt.figure()
    io.imshow(edges_vertical, cmap='gray')
    
    edges = edges_horizontal + edges_vertical
    plt.figure()
    io.imshow(edges, cmap='gray')
    
if __name__ == "__main__":
    img_in = imread('../sekvence/clocks/clock1.png')
    img_in = color.rgb2gray(color.rgba2rgb(img_in))
    canny_edge_detection(img_in, 4, 0, 5)
    # plt.figure()
    # io.imshow(img_in)