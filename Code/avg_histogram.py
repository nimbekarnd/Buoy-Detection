import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob
import sys
from constants import *


def calc_intensity_freq(data, val_range=(0,256)):
    val_freq_map = np.zeros(val_range[1])
    for intensity in data:
        val_freq_map[intensity] += 1
    return val_freq_map

def calc_average_hist(image_color):
    path_map = {
        'r': 'frames/trainingframes/cropped/red/*.jpg', 
        'g': 'frames/cropped/green/*.png', #'frames/trainingframes/cropped/green/*.jpg',
        'y': 'frames/trainingframes/cropped/yellow/*.jpg' 
    }
    path = path_map[image_color]
    img_paths = glob.glob(path)

    avg_h = 25 #30
    avg_w = 25 #30
    b_channel_hists = [] 
    g_channel_hists = [] 
    r_channel_hists = [] 

    for path in img_paths:
        img = cv2.imread(path)
        
        img = cv2.resize(img, (avg_w,avg_h), interpolation = cv2.INTER_AREA)
        b_channel, g_channel, r_channel = cv2.split(img)
        b_channel, g_channel, r_channel = b_channel.flatten(), g_channel.flatten(), r_channel.flatten()

        b_channel_hists.append(calc_intensity_freq(b_channel))
        g_channel_hists.append(calc_intensity_freq(g_channel))
        r_channel_hists.append(calc_intensity_freq(r_channel))
        
    b_channel_hist = np.mean(b_channel_hists, axis=0)
    g_channel_hist = np.mean(g_channel_hists, axis=0)
    r_channel_hist = np.mean(r_channel_hists, axis=0)

    return b_channel_hist, g_channel_hist, r_channel_hist


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please enter the buoy color as an argument (r, g or y)')
    else:
        buoy_clr = str(sys.argv[1])
        b_channel_hist, g_channel_hist, r_channel_hist = calc_average_hist(buoy_clr)

        plt.title('Average histogram for ' + TITLE_MAP[buoy_clr])
        plt.xlabel('intensity')
        plt.ylabel('frequency')
        plt.plot(b_channel_hist, c='b')
        plt.plot(g_channel_hist, c='g')
        plt.plot(r_channel_hist, c='r')

        plt.savefig( 'avg_histogram_' + TITLE_MAP[buoy_clr] + '.png', bbox_inches='tight') 
        plt.show()
