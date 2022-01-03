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
    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(img)

    # gradients
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8
    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8

    gradient_vertical = ndimage.correlate(img, sobel_vertical)
    gradient_horizontal = ndimage.correlate(img, sobel_horizontal)
    # vertical gradient
    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(gradient_vertical, cmap='gray')

    # horizontal gradient
    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(gradient_horizontal, cmap='gray')

    magnitude = np.sqrt(np.square(gradient_vertical) +
                        np.square(gradient_horizontal))
    angle = np.rad2deg(np.arctan2(gradient_vertical, gradient_horizontal))

    # magnitude
    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(magnitude, cmap='gray')

    # angle
    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(angle, cmap='gray')

    # quantization
    horizontal = ((angle > -22.5) * (angle < 22.5) +
                  (angle < -157.5) + (angle > 157.5))
    # plt.figure()
    # io.imshow(horizontal*1.)

    vertical = ((angle > 67.5) * (angle < 112.5)) + \
        ((angle < -67.5) * (angle > -112.5))
    # plt.figure()
    # io.imshow(vertical*1.)

    diag45 = ((angle > 112.5) * (angle < 157.5)) + \
        ((angle < -22.5) * (angle > -67.5))
    # plt.figure()
    # io.imshow(diag45*1.)

    diag_45 = ((angle > 22.5) * (angle < 67.5)) + \
        ((angle < -112.5) * (angle > -157.5))
    # plt.figure()
    # io.imshow(diag_45*1.)

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

    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(edges_horizontal, cmap='gray')

    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(edges_vertical, cmap='gray')

    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(edges45, cmap='gray')

    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(edges_45, cmap='gray')

    edges = edges_horizontal + edges_vertical + edges45 + edges_45

    # plt.figure(figsize=(12, 9), dpi=80)
    # io.imshow(edges, cmap='gray')

    # the final
    output = (edges > threshold_low) * (edges < threshold_high)
    check = ~output*(edges > 0)

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
        check = ~output*(edges > 0)
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(output*magnitude, cmap='gray')

    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(output*angle, cmap='gray')
    return output*magnitude  # , output*angle


def get_line_segments(img_edges: np.array, line: np.array, min_size: int, max_gaps: int, tolerancy: int) -> tuple:
    theta = line[0]
    rho = line[1]
    originy = np.arange(tolerancy, img_edges.shape[0]-tolerancy)
    originx = np.arange(tolerancy, img_edges.shape[1]-tolerancy)
    
    x = np.round((rho - originx * np.cos(theta)) / np.sin(theta))
    x = x.astype(int)
    y = np.round((rho - originy * np.sin(theta)) / np.cos(theta))
    y = y.astype(int)
    line_start = [-1, -1]
    line_stop = [-1, -1]
    length = 0
    ok = False
    gap_len = 0
    lines = []
    if not (theta<deg2rad(45) and theta>deg2rad(-45)):
        for i in range(x.size):
            first = np.arange(x[i]-tolerancy, x[i]+tolerancy+1)
            second = np.arange(originx[i]-tolerancy, originx[i]+tolerancy+1)
            
            second = second[first>=0]
            first = first[first>=0]
            
            second = second[first<img_edges.shape[0]]
            first = first[first<img_edges.shape[0]]
            
            mat = img_edges[first, second]
            mat = mat>0
            if np.any(mat) and mat.size>0:
                if ok == False:
                    line_start = np.array([x[i], originx[i]])
                    ok = True
                line_stop = np.array([x[i], originx[i]])
                length = length + 1 + gap_len
                gap_len = 0
            else:
                if ok == True:
                    gap_len = gap_len + 1
            
            if gap_len > max_gaps:
                ok=False
                gap_len = 0
                if length > min_size:
                    lines = lines + [[line_start, line_stop]]
                length = 0
    else:
        for i in range(y.size):
            first = np.arange(originy[i]-tolerancy, originy[i]+tolerancy+1)
            second = np.arange(y[i]-tolerancy, y[i]+tolerancy+1)
            
            first = first[second>=0]
            second = second[second>=0]
            
            first = first[second<img_edges.shape[1]]
            second = second[second<img_edges.shape[1]]
            
            mat = img_edges[first, second]
            mat = mat>0
            if np.any(mat):
                if ok == False:
                    line_start = np.array([originy[i], y[i]])
                    ok = True
                line_stop = np.array([originy[i], y[i]])
                length = length + 1 + gap_len
                gap_len = 0
            else:
                if ok == True:
                    gap_len = gap_len + 1
            
            if gap_len > max_gaps:
                ok=False
                gap_len = 0
                if length > min_size:
                    lines = lines + [[line_start, line_stop]]
                length = 0
    return lines

def extract_time(img_in: np.array) -> tuple:
    canny = canny_edge_detection(img_in, 0.7, 0.2, 0.55)
    
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    ok=False
    left = 0
    s = 0
    for i in range(canny.shape[0]):
        if np.sum(canny[i, :]) != 0 and ok == False:
            s = i
            ok = True
        if np.sum(canny[i, :]) == 0 and ok == True:
            canny = canny[s:i, :]
            break
    ok=False
    left = 0
    s = 0
    for i in range(canny.shape[1]):
        if np.sum(canny[:, i]) != 0 and ok == False:
            ok = True
            s = i
        if np.sum(canny[:, i]) == 0 and ok == True:
            st = i
            canny = canny[:, s:i]
            break
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    canny[canny<np.max(canny)*3/4] = 0

    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    [out, angles, distances] = skimage.transform.hough_line(canny)
    [intensity, peak_angles, peak_distances] = skimage.transform.hough_line_peaks(out, angles=angles, dists=distances, min_distance=20, threshold=0.4*amax(out), num_peaks=5)
    
    line_lengths = []
    
    fix, axes = plt.subplots(1, 1, figsize=(20, 8))
    axes.imshow(canny, cmap=plt.cm.gray)
    axes.set_title('Input image with detected lines')
    origin = np.array((0, img_in.shape[1]))
    max1 = -1
    ang1 = -1
    right_half1 = False
    top_half1 = False
    max2 = -1
    ang2 = -1
    right_half2 = False
    top_half2 = False
    
    for _, angle, dist in zip(intensity, peak_angles, peak_distances):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        lines = get_line_segments(canny, np.array([angle, dist]), 60, 20, 1)
        for line in lines:
            if not line[0][0] in range(round(canny.shape[0]/2)-30, round(canny.shape[0]/2)+30) and not line[1][0] in range(round(canny.shape[0]/2)-30, round(canny.shape[0]/2)+30):
                if not line[0][1] in range(round(canny.shape[1]/2)-30, round(canny.shape[1]/2)+30) and not line[1][1] in range(round(canny.shape[1]/2)-30, round(canny.shape[1]/2)+30):
                    break
            right_half = False
            top_half = False
            if (line[0][1] + line[1][1])/2 > canny.shape[1]/2:
                right_half = True
            if (line[0][0] + line[1][0])/2 < canny.shape[0]/2:
                top_half = True
            line_length = sqrt((line[0][0]- line[1][0])**2 + (line[0][1]-line[1][1])**2)
            line_lengths = line_lengths + [line_length]
            if max1 == -1:
                max1 = line_length
                ang1 = angle
                right_half1 = right_half
                top_half1 = top_half
            else:
                if max1 > line_length and max2 < line_length:
                    max2 = line_length
                    ang2 = angle
                    right_half2 = right_half
                    top_half2 = top_half
            if line_length> max1:
                max2 = max1
                ang2 = ang1
                right_half2 = right_half1
                top_half2 = top_half1
                max1 = line_length
                ang1 = angle
                right_half1 = right_half
                top_half1 = top_half
        axes.plot(origin, (y0, y1), '-r')
    if max2 == -1:
        max2 = max1
        ang2 = ang1
        right_half2 = right_half1
        top_half2 = top_half1
    axes.set_xlim(origin)
    axes.set_ylim((img_in.shape[0], 0))
    # axes.set_axis_off()
    
    plt.show()
    ang_minutes = ang1
    if ang1>0:
        if (top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1 >= deg2rad(45)and ang1 <= deg2rad(90)) or (ang1 >= deg2rad(-90)and ang1<=deg2rad(-45)))):
            ang_minutes = ang1
        else:
            ang_minutes = deg2rad(180) + ang1
    else:
        if (not top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1>=deg2rad(45) and ang1<=deg2rad(90)) or (ang1 >= deg2rad(-90) and ang1 <= deg2rad(-45)))):
            ang_minutes = deg2rad(180) + ang1
        else:
            ang_minutes = deg2rad(360) + ang1
    
    minutes = round(ang_minutes/deg2rad(360)*60)
    ang_hours = ang2
    if ang2>0:
        if (top_half2 and ang2 >= deg2rad(-45) and ang2 <= deg2rad(45)) or ((right_half2 and ((ang2 >= deg2rad(45)and ang2 <= deg2rad(90)) or (ang2 >=deg2rad(-90) and ang2 <= deg2rad(-45))))):
            ang_hours = ang2
        else:
            ang_hours = deg2rad(180) + ang2
    else:
        if (not top_half2 and ang2 >= deg2rad(-45) and ang2 <= deg2rad(45)) or (right_half2 and ((ang2 >= deg2rad(45) and ang2 <= deg2rad(90)) or (ang2 >= deg2rad(-90) and ang2 <= deg2rad(-45)))):
            ang_hours = deg2rad(180) + ang2
        else:
            ang_hours = deg2rad(360) + ang2
    
    hours = int(floor(ang_hours/deg2rad(360)*12))
    if minutes == 60:
        hours = hours+1
        minutes=0
    if hours == 0:
        hours = 12
    return hours, minutes

def extract_time_bonus(img_in: np.array) -> tuple:
    canny = canny_edge_detection(img_in, 0.2, 0.2, 0.55)
    
    canny_seconds = np.copy(canny)
    canny_seconds[canny_seconds>np.max(canny_seconds)*3/4] = 0
    
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    ok=False
    left = 0
    s = 0
    for i in range(canny.shape[0]):
        if np.sum(canny[i, :]) != 0 and ok == False:
            s = i
            ok = True
        if np.sum(canny[i, :]) == 0 and ok == True:
            canny = canny[s:i, :]
            break
    ok=False
    left = 0
    s = 0
    for i in range(canny.shape[1]):
        if np.sum(canny[:, i]) != 0 and ok == False:
            ok = True
            s = i
        if np.sum(canny[:, i]) == 0 and ok == True:
            st = i
            canny = canny[:, s:i]
            break
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    canny[canny<np.max(canny)*3/4] = 0
    
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny, cmap='gray')
    
    [out, angles, distances] = skimage.transform.hough_line(canny)
    [intensity, peak_angles, peak_distances] = skimage.transform.hough_line_peaks(out, angles=angles, dists=distances, min_distance=20, threshold=0.4*amax(out), num_peaks=5)
    
    line_lengths = []
    
    fix, axes = plt.subplots(1, 1, figsize=(20, 8))
    axes.imshow(canny, cmap=plt.cm.gray)
    axes.set_title('Input image with detected lines')
    origin = np.array((0, img_in.shape[1]))
    max1 = -1
    ang1 = -1
    right_half1 = False
    top_half1 = False
    max2 = -1
    ang2 = -1
    right_half2 = False
    top_half2 = False
    
    for _, angle, dist in zip(intensity, peak_angles, peak_distances):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        lines = get_line_segments(canny, np.array([angle, dist]), 60, 20, 1)
        for line in lines:
            if not line[0][0] in range(round(canny.shape[0]/2)-30, round(canny.shape[0]/2)+30) and not line[1][0] in range(round(canny.shape[0]/2)-30, round(canny.shape[0]/2)+30):
                if not line[0][1] in range(round(canny.shape[1]/2)-30, round(canny.shape[1]/2)+30) and not line[1][1] in range(round(canny.shape[1]/2)-30, round(canny.shape[1]/2)+30):
                    break
            right_half = False
            top_half = False
            if (line[0][1] + line[1][1])/2 > canny.shape[1]/2:
                right_half = True
            if (line[0][0] + line[1][0])/2 < canny.shape[0]/2:
                top_half = True
            line_length = sqrt((line[0][0]- line[1][0])**2 + (line[0][1]-line[1][1])**2)
            line_lengths = line_lengths + [line_length]
            if max1 == -1:
                max1 = line_length
                ang1 = angle
                right_half1 = right_half
                top_half1 = top_half
            else:
                if max1 > line_length and max2 < line_length:
                    max2 = line_length
                    ang2 = angle
                    right_half2 = right_half
                    top_half2 = top_half
            if line_length> max1:
                max2 = max1
                ang2 = ang1
                right_half2 = right_half1
                top_half2 = top_half1
                max1 = line_length
                ang1 = angle
                right_half1 = right_half
                top_half1 = top_half
        axes.plot(origin, (y0, y1), '-r')
    if max2 == -1:
        max2 = max1
        ang2 = ang1
        right_half2 = right_half1
        top_half2 = top_half1
    axes.set_xlim(origin)
    axes.set_ylim((img_in.shape[0], 0))
    # axes.set_axis_off()
    
    plt.show()
    ang_minutes = ang1
    if ang1>0:
        if (top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1 >= deg2rad(45)and ang1 <= deg2rad(90)) or (ang1 >= deg2rad(-90)and ang1<=deg2rad(-45)))):
            ang_minutes = ang1
        else:
            ang_minutes = deg2rad(180) + ang1
    else:
        if (not top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1>=deg2rad(45) and ang1<=deg2rad(90)) or (ang1 >= deg2rad(-90) and ang1 <= deg2rad(-45)))):
            ang_minutes = deg2rad(180) + ang1
        else:
            ang_minutes = deg2rad(360) + ang1
    
    minutes = round(ang_minutes/deg2rad(360)*60)
    ang_hours = ang2
    if ang2>0:
        if (top_half2 and ang2 >= deg2rad(-45) and ang2 <= deg2rad(45)) or ((right_half2 and ((ang2 >= deg2rad(45)and ang2 <= deg2rad(90)) or (ang2 >=deg2rad(-90) and ang2 <= deg2rad(-45))))):
            ang_hours = ang2
        else:
            ang_hours = deg2rad(180) + ang2
    else:
        if (not top_half2 and ang2 >= deg2rad(-45) and ang2 <= deg2rad(45)) or (right_half2 and ((ang2 >= deg2rad(45) and ang2 <= deg2rad(90)) or (ang2 >= deg2rad(-90) and ang2 <= deg2rad(-45)))):
            ang_hours = deg2rad(180) + ang2
        else:
            ang_hours = deg2rad(360) + ang2
    
    hours = int(floor(ang_hours/deg2rad(360)*12))
    if minutes == 60:
        hours = hours+1
        minutes=0
    if hours == 0:
        hours = 12

    origin = np.array((0, img_in.shape[1]))
    max1 = -1
    ang1 = -1
    right_half1 = False
    top_half1 = False
    
    for _, angle, dist in zip(intensity, peak_angles, peak_distances):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        lines = get_line_segments(canny_seconds, np.array([angle, dist]), 30, 30, 3)
        for line in lines:
            if not line[0][0] in range(round(canny_seconds.shape[0]/2)-30, round(canny_seconds.shape[0]/2)+30) and not line[1][0] in range(round(canny_seconds.shape[0]/2)-30, round(canny_seconds.shape[0]/2)+30):
                if not line[0][1] in range(round(canny_seconds.shape[1]/2)-30, round(canny_seconds.shape[1]/2)+30) and not line[1][1] in range(round(canny_seconds.shape[1]/2)-30, round(canny_seconds.shape[1]/2)+30):
                    break
            right_half = False
            top_half = False
            if (line[0][1] + line[1][1])/2 > canny_seconds.shape[1]/2:
                right_half = True
            if (line[0][0] + line[1][0])/2 < canny_seconds.shape[0]/2:
                top_half = True
            line_length = sqrt((line[0][0]- line[1][0])**2 + (line[0][1]-line[1][1])**2)
            line_lengths = line_lengths + [line_length]
            if line_length> max1:
                max1 = line_length
                ang1 = angle
                right_half1 = right_half
                top_half1 = top_half
        # axes.plot(origin, (y0, y1), '-r')
    ang_seconds = ang1
    if ang1>0:
        if (top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1 >= deg2rad(45)and ang1 <= deg2rad(90)) or (ang1 >= deg2rad(-90)and ang1<=deg2rad(-45)))):
            ang_seconds = ang1
        else:
            ang_seconds = deg2rad(180) + ang1
    else:
        if (not top_half1 and ang1 >= deg2rad(-45) and ang1 <= deg2rad(45)) or (right_half1 and ((ang1>=deg2rad(45) and ang1<=deg2rad(90)) or (ang1 >= deg2rad(-90) and ang1 <= deg2rad(-45)))):
            ang_seconds = deg2rad(180) + ang1
        else:
            ang_seconds = deg2rad(360) + ang1
    
    seconds = round(ang_seconds/deg2rad(360)*60)
    if max1 == -1:
        seconds = -1
    plt.figure(figsize=(12, 9), dpi=80)
    io.imshow(canny_seconds, cmap='gray')
    return hours, minutes, seconds

if __name__ == "__main__":
    img_in = imread('../sekvence/clocks/clock8.jpg')
    
    img_in = color.rgb2gray(img_in)
    img_in = skimage.img_as_float(img_in)
    print(extract_time_bonus(img_in))
    # plt.figure()
    # io.imshow(img_in)
