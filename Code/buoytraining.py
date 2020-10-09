import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from constants import *
from modellearning import get_gmm_model_params



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Enter buoy color (r,g or y) and number of gaussians as arguments.')
    else:
        buoy_clr, num_gaussian = str(sys.argv[1]), int(sys.argv[2])
        input_data, gmm = get_gmm_model_params(buoy_clr, num_gaussian)

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        ind = np.argsort(gmm.mu_list)
        mu_list = gmm.mu_list[ind]
        sd_list = gmm.sd_list[ind]

        print(mu_list)
        print(sd_list)

        axis_title = '| '
        for i in range(num_gaussian):
            axis_title += 'mu' + str(i+1) + '=' + str(mu_list[i]) + ', sd' + str(i+1) + '=' + str(sd_list[i]) + ' | '

        ax.set_title(axis_title)
        ax.set_xlabel('$x$')
        ax.set_ylabel('pdf')

        ax1.set_xlabel('iterations')
        ax1.set_ylabel('log-likelihood')
        
        num_bins = 200
        ax.hist(input_data, num_bins, normed=1, facecolor='blue', alpha=0.5)

        ax1.plot(range(len(gmm.log_likelihood)), gmm.log_likelihood)

        plot_input_x = np.linspace(90,300,num=1000)

        plot_input_y_orig = []
        plot_input_y_gmm = gmm.gmm_pdf(plot_input_x) #[]
        
        ax.plot(plot_input_x, plot_input_y_gmm, c=buoy_clr, label='pdf')

        ax.scatter(input_data, 0.01*np.ones((len(input_data),1)), c='b', label='data point')

        ax.legend()
        plt.savefig('gmmmodel_' + buoy_clr + '_' + str(num_gaussian) + 'gauss.png', bbox_inches='tight') 
        plt.show()



