import json
import numpy as np
import cv2
import sys
from gmm1d import GaussianMMOneD
from constants import *
from colorsegment import segment_color 


def get_model_params(num_gauss_r, num_gauss_g, num_gauss_y):
    r_param_file = 'modelparams/gmmparams_' + RED_BUOY + '_' + str(num_gauss_r) + 'gauss.json'
    g_param_file = 'modelparams/gmmparams_' + GREEN_BUOY + '_' + str(num_gauss_g) + 'gauss.json'
    y_param_file = 'modelparams/gmmparams_' + YELLOW_BUOY + '_' + str(num_gauss_y) + 'gauss.json'
    files = [r_param_file, g_param_file, y_param_file]
    num_gauss_list = [num_gauss_r, num_gauss_g, num_gauss_y]
    buoys = [RED_BUOY, GREEN_BUOY, YELLOW_BUOY]
    buoy_gmm = np.array([read_params_from_file(filename, num_gauss, buoy_clr) for filename, num_gauss, buoy_clr in zip(files, num_gauss_list, buoys)])
    return buoy_gmm

def read_params_from_file(file_name, num_gauss, buoy_clr):
    try:
        with open(file_name) as json_file:
            params = json.load(json_file)
            gmm = GaussianMMOneD(num_gauss, params['mu'], params['sd'], params['mix'])
            gmm.update_mixture()
        return gmm
    except:
        print('No such training has been performed to detect ' + TITLE_MAP[buoy_clr] + ' using ' + str(num_gauss) + ' gaussian/s.' + '. Please refer README to train such model.')
        return None
    


def process_video(buoy_gmm, num_gauss_r=1, num_gauss_g=1, num_gauss_y=1):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("buoy_detection_using_" + str(num_gauss_r) + "_gaussian.avi", fourcc, 10.0, (640, 480))

    cap = cv2.VideoCapture("detectbuoy.avi")
    while True:
        ret, img = cap.read()
        if not ret:
            break

        segment_color(img, RED_BUOY, buoy_gmm, num_gauss_r)
        segment_color(img, GREEN_BUOY, buoy_gmm, num_gauss_g)
        segment_color(img, YELLOW_BUOY, buoy_gmm, num_gauss_y)
        
        cv2.imshow('Buoy Detection', img)
        out.write(img)

        if cv2.waitKey(1) ==27:
            break
    
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please enter the number of gaussians required for each buoy respectively eg: 1 1 1 for red, green and yellow buoy.')
    else:
        num_gauss_r = int(sys.argv[1])
        num_gauss_g = int(sys.argv[2])
        num_gauss_y = int(sys.argv[3])
        buoy_gmm = get_model_params(num_gauss_r = num_gauss_r, num_gauss_g = num_gauss_g, num_gauss_y = num_gauss_y)
        if np.all(buoy_gmm != None):
            process_video(buoy_gmm, num_gauss_r = num_gauss_r, num_gauss_g = num_gauss_g, num_gauss_y = num_gauss_y)






