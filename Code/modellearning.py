import numpy as np
import math
import glob
import cv2
import json
from constants import *
from gmm1d import GaussianMMOneD

def get_gmm_model_params(buoy_clr, num_gaussian):
    buoy_path = PATH_MAP[buoy_clr]
    img_paths = glob.glob(buoy_path)
    intensity_data = []

    avg_h = 25 #30
    avg_w = 25 #30

    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (avg_w,avg_h), interpolation = cv2.INTER_AREA)
        update_intensity_data(intensity_data, img, buoy_clr)

    input_data = np.zeros(len(intensity_data))
    input_data[:] = intensity_data

    orig_mean = np.mean(input_data)
    orig_sd = np.var(input_data)**0.5
    print(orig_mean, orig_sd)
 
    gmm_one_d = GaussianMMOneD(num_gaussian)
    num_iteration = 50
    gmm_one_d.train(input_data, num_iteration)

    write_param_json(gmm_one_d, buoy_clr)
    print(gmm_one_d.mu_list, gmm_one_d.sd_list)

    return input_data, gmm_one_d


def update_intensity_data(intensity_data, img, buoy_clr):
    if buoy_clr == YELLOW_BUOY:
        red_chn = img[:,:, GREEN_CHN].flatten()
        green_chn = img[:, :, GREEN_CHN].flatten()
        flatten_img = []
        for i in range(len(red_chn)):
            flatten_img.append(red_chn[i]+green_chn[i])
    else:
        color_chn = CHANNEL_MAP[buoy_clr]
        channel_img = img[:, :, color_chn]
        flatten_img = channel_img.flatten()

    intensity_data.extend(flatten_img)

def write_param_json(gmm_one_d, buoy_clr):
    params_data = {}
    
    ind = np.argsort(gmm_one_d.mu_list)
    gmm_one_d.mu_list = gmm_one_d.mu_list[ind]
    gmm_one_d.sd_list = gmm_one_d.sd_list[ind]
    gmm_one_d.mix_list = gmm_one_d.mix_list[ind]

    params_data['mu'] = list(gmm_one_d.mu_list)
    params_data['sd'] = list(gmm_one_d.sd_list)
    params_data['mix'] = list(gmm_one_d.mix_list)
    file_name = 'modelparams/gmmparams_' + buoy_clr + '_' + str(gmm_one_d.num_gaussian) + 'gauss.json' 
    with open(file_name, 'w') as outfile:
        json.dump(params_data, outfile)

