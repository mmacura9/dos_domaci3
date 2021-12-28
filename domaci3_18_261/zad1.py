# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:23:16 2021

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

def canny_edge_detection(img_in: np.array, sigma: float, threshold_low: float, threshold_high: float) -> np.array:
    # filtering the input image
    img = filters.gaussian(img_in, sigma=sigma, truncate=3)
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(img)
    
    # gradients
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8
    sobel_horizontal = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])/8
    
    gradient_vertical = ndimage.correlate(img, sobel_vertical)
    gradient_horizontal = ndimage.correlate(img, sobel_horizontal)
    # vertical gradient
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(gradient_vertical, cmap='gray')
    
    # horizontal gradient
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(gradient_horizontal, cmap='gray')
    
    magnitude = np.sqrt(np.square(gradient_vertical) + np.square(gradient_horizontal))
    print(np.max(magnitude))
    angle = np.rad2deg(np.arctan2(gradient_vertical, gradient_horizontal))
    
    # magnitude
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(magnitude, cmap='gray')
    
    # angle
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(angle, cmap='gray')
    
    # quantization
    
    horizontal = ((angle > -22.5) * (angle < 22.5) + (angle < -157.5) + (angle > 157.5))
    plt.figure()
    io.imshow(horizontal*1.)
    
    vertical = ((angle > 67.5) * (angle < 112.5)) + ((angle < -67.5) * (angle > -112.5))
    plt.figure()
    io.imshow(vertical*1.)
    
    diag45 = ((angle > 112.5) * (angle < 157.5)) + ((angle < -22.5) * (angle > -67.5))
    plt.figure()
    io.imshow(diag45*1.)
    
    diag_45 = ((angle > 22.5) * (angle < 67.5)) + ((angle < -112.5) * (angle > -157.5))
    plt.figure()
    io.imshow(diag_45*1.)
    
    img_horizontal = horizontal * magnitude
    edges_horizontal = np.copy(img_horizontal)
    for i in range(1, img_horizontal.shape[0]-1):
        for j in range(1, img_horizontal.shape[1]-1):
            if img_horizontal[i, j] < img_horizontal[i, j-1] or img_horizontal[i, j] < img_horizontal[i, j+1]:
                edges_horizontal[i, j] = 0
    
    img_vertical = vertical * magnitude
    edges_vertical = np.copy(img_vertical)
    
    for i in range(1, img_vertical.shape[0]-1):
        for j in range(1, img_vertical.shape[1]-1):
            if img_vertical[i, j] < img_vertical[i-1, j] or img_vertical[i, j] < img_vertical[i+1, j]:
                edges_vertical[i, j] = 0
                
    img_45 = diag_45 * magnitude
    edges_45 = np.copy(img_45)
    for i in range(1, img_45.shape[0]-1):
        for j in range(1, img_45.shape[1]-1):
            if img_45[i, j] < img_45[i-1, j-1] or img_45[i, j] < img_45[i+1, j+1]:
                edges_45[i, j] = 0
    
    img45 = diag45 * magnitude
    edges45 = np.copy(img45)
    for i in range(1, img45.shape[0]-1):
        for j in range(1, img45.shape[1]-1):
            if img45[i, j] < img45[i-1, j+1] or img45[i, j] < img45[i+1, j-1]:
                edges45[i, j] = 0
                
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(edges_horizontal, cmap='gray')
    
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(edges_vertical, cmap='gray')
    
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(edges45, cmap='gray')
    
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(edges_45, cmap='gray')
    
    edges = edges_horizontal + edges_vertical + edges45 + edges_45

    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(edges, cmap='gray')
    
    # the final
    output = (edges>threshold_low) * (edges<threshold_high)
    check = ~output*(edges>0)
    
    num = np.sum(output)
    num1 = num-1
    while num != num1:
        num1 = num
        for i in range(check.shape[0]):
            for j in range(check.shape[1]):
                if check[i, j] == True:
                    pom = False
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if output[i+k, j+l]:
                                output[i, j] = True
                            if pom == False:
                                break
                        if pom == False:
                            break
        num = np.sum(output)
        check = ~output*(edges>0)
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(output*magnitude, cmap='gray')
    
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(output*angle, cmap='gray')
    return output*magnitude#, output*angle

def get_line_segments(img_edges: np.array, line: np.array, min_size: int, max_gaps: int, tolerancy: float) -> np.array:
    theta = math.radians(line[0])
    rho = line[1]
    y = np.arange(img_edges.shape[1]*1.)
    x = np.arange(img_edges.shape[0]*1.)
    matx = np.zeros((img_edges.shape[0], img_edges.shape[1]), dtype=float)
    maty = np.zeros((img_edges.shape[0], img_edges.shape[1]), dtype=float)
    for i in range(img_edges.shape[0]):
        maty[i, :] = y
        matx[:, i] = x
    output = (matx*cos(theta)+maty*sin(theta)) >= rho - tolerancy
    output = logical_and(output,(matx*cos(theta)+maty*sin(theta)) <= rho + tolerancy)
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(output*1., cmap='gray')
    
    plt.figure(figsize=(12,9), dpi=80)
    io.imshow(output*img_edges, cmap='gray')
    
    
if __name__ == "__main__":
    img_in = imread('../sekvence/clocks/clock8.jpg')
    img_in = color.rgb2gray(img_in)
    img_in = skimage.img_as_float(img_in)
    canny = canny_edge_detection(img_in, 0.5, 0.2, 0.55)
    theta = 141
    rho = cos(math.radians(theta-45))*sqrt(canny.shape[0]*canny.shape[0] + canny.shape[1]*canny.shape[1])/2
    get_line_segments(canny, np.array([theta, rho]), 4, 2, 1)
    # plt.figure()
    # io.imshow(img_in)