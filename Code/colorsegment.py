import numpy as np 
import cv2
from imutils import contours
from constants import * 


def segment_color(img, color, gmm, num_gaussian):
    segment_function_map ={
        RED_BUOY: {
            1: segment_red_buoy_onegauss,
            3: segment_red_buoy_threegauss
        },
        GREEN_BUOY: {
            1: segment_green_buoy_onegauss,
            3: segment_green_buoy_threegauss
        },
        YELLOW_BUOY: {
            1: segment_yellow_buoy_onegauss,
            3: segment_yellow_buoy_threegauss
        }
    }
    try:
        segment_function_map[color][num_gaussian](img, gmm)
    except:
        msg = 'There is no implementation to detect ' + TITLE_MAP[color] + ' using ' + str(num_gaussian) + ' gaussian/s.'
        cv2.putText(img, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        



def segment_red_buoy_onegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img_r = img[:, :, RED_CHN]
    channel_img_g = img[:, :, GREEN_CHN]

    pdf_vals_r = gmm_r.gmm_pdf(channel_img_r)
    pdf_vals_g = gmm_g.gmm_pdf(channel_img_g)
    pdf_max = np.max(pdf_vals_r)
    bin_img = np.zeros((height, width), dtype = np.uint8)

    
    pdf_threshold_r = 0.007
    pdf_threshold_g = 0.005
    
    for i in range(height):
        for j in range(width):
            if pdf_vals_r[i,j] >= pdf_threshold_r and img[i,j,GREEN_CHN] <= 150 and img[i,j,BLUE_CHN] <= 120:
                bin_img[i,j] = 255

    
    kernel = np.ones((2,2), np.uint8)
    
    opening = bin_img #cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #closing = cv2.dilate(opening, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=1)

    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)

    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x)-4,int(y)-7)
        radius = int(radius)
        if 3 < radius < 45:
            radius = radius+4 if radius<=10 else radius
            center = (center[0]+1, center[1]-1)
            cv2.circle(img, center, radius+1, (0,0,255), 3)
    
    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)
       

def segment_red_buoy_threegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img_r = img[:, :, RED_CHN]
    channel_img_g = img[:, :, GREEN_CHN]

    pdf_vals_r = gmm_r.gmm_pdf(channel_img_r)
    pdf_vals_g = gmm_g.gmm_pdf(channel_img_g)
    pdf_max = np.max(pdf_vals_r)
    bin_img = np.zeros((height, width), dtype = np.uint8)

    pdf_threshold_r = 0.011 #0.006 #0.01 #0.009
    pdf_threshold_g = 0.0055 #0.009 #0.0055

    for i in range(height):
        for j in range(width):
            if pdf_vals_r[i,j] >= pdf_threshold_r and pdf_vals_g[i,j] <= pdf_threshold_g and img[i,j,BLUE_CHN] <= 130:
                bin_img[i,j] = 255
    
    kernel = np.ones((4,4), np.uint8)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=3)
    
    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)
    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x)-4,int(y)-7)
        radius = int(radius)
        if 4 < radius < 50:
            cv2.circle(img, center, radius+1, (0,0,255), 3)
    

    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)
        
    

def segment_green_buoy_onegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img = img[:, :, GREEN_CHN]

    pdf_vals_g = gmm_g.gmm_pdf(channel_img)
    bin_img = np.zeros((height, width), dtype = np.uint8)

    pdf_threshold_g = 0.001
    for i in range(height):
        for j in range(width):
            if pdf_vals_g[i,j] >= pdf_threshold_g and img[i,j,GREEN_CHN] >=220 and img[i,j,RED_CHN] <= 170 and img[i,j,BLUE_CHN] <= 140:
                bin_img[i,j] = 255

    
    kernel = np.ones((3,3), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=2)

    
    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)
    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y)+3)
        radius = int(radius)
        if radius > 10:
            cv2.circle(img, center, radius+1, (0,255,0), 3)
    
    
    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)
    

def segment_green_buoy_threegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img = img[:, :, GREEN_CHN]
    flatten_img = channel_img.flatten()

    pdf_vals_g = gmm_g.gmm_pdf(channel_img)
    bin_img = np.zeros((height, width), dtype = np.uint8)

    pdf_threshold_g = 0.01
    for i in range(height):
        for j in range(width):
            if pdf_vals_g[i,j] >= pdf_threshold_g and img[i,j,RED_CHN] <= 190 and img[i,j,BLUE_CHN] <= 150:
                bin_img[i,j] = 255

    kernel = np.ones((3,3), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=2)

    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)
    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y)+3)
        radius = int(radius)
        if 7 < radius < 15:
            cv2.circle(img, center, radius+1, (0,255,0), 3)
    
    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)


def segment_yellow_buoy_onegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img_g = img[:, :, GREEN_CHN]
    channel_img_r = img[:, :, RED_CHN]
    channel_img_y = channel_img_g + channel_img_r

    pdf_vals_y = gmm_y.gmm_pdf(channel_img_y)
    pdf_vals_r = gmm_r.gmm_pdf(channel_img_r)
    pdf_vals_g = gmm_g.gmm_pdf(channel_img_g)

    bin_img = np.zeros((height, width), dtype = np.uint8)

    pdf_threshold_y = 0.007 #0.01
    pdf_threshold_r = 0.007 #0.01
    pdf_threshold_g = 0.001 #0.01
    for i in range(height):
        for j in range(width):
            if pdf_vals_r[i,j] >= pdf_threshold_r and pdf_vals_g[i,j] >= pdf_threshold_g and img[i,j,BLUE_CHN] <= 115 and img[i,j,RED_CHN] >= 200 and img[i,j,GREEN_CHN] >= 230:
                bin_img[i,j] = 255
    
    kernel = np.ones((5,5), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=2)

    
    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)
    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y-3))
        radius = int(radius)
        cv2.circle(img, center, radius-4, (0,255,255), 3)
        #cv2.putText(img, str(radius), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)


def segment_yellow_buoy_threegauss(img, gmm):
    gmm_r, gmm_g, gmm_y = gmm[0], gmm[1], gmm[2]
    height, width = img.shape[0], img.shape[1] 
    channel_img_g = img[:, :, GREEN_CHN]
    channel_img_r = img[:, :, RED_CHN]
    channel_img_y = channel_img_g + channel_img_r

    pdf_vals_y = gmm_y.gmm_pdf(channel_img_y)
    pdf_vals_r = gmm_r.gmm_pdf(channel_img_r)
    pdf_vals_g = gmm_g.gmm_pdf(channel_img_g)

    bin_img = np.zeros((height, width), dtype = np.uint8)

    pdf_threshold_y = 0.01
    pdf_threshold_r = 0.01
    pdf_threshold_g = 0.01
    for i in range(height):
        for j in range(width):
            if pdf_vals_r[i,j] >= pdf_threshold_r and pdf_vals_g[i,j] >= pdf_threshold_g and img[i,j,BLUE_CHN] <= 130:
                bin_img[i,j] = 255
    
    kernel = np.ones((5,5), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.dilate(closing, kernel, iterations=2)

    
    ret, bin_threshold = cv2.threshold(closing, 240, 255, cv2.THRESH_BINARY)
    contour_list, _= cv2.findContours(bin_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contour_list:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y+1))
        radius = int(radius)
        if radius > 15:
            cv2.circle(img, center, radius-3, (0,255,255), 3)
        if radius > 30:
            cv2.circle(img, (center[0], center[1]-3), radius-3, (0,255,255), 3)
    
    #cv2.imshow('bin_img', bin_img)
    #cv2.imshow('closing', closing)
