import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
from gmm1d import GaussianMMOneD


if __name__ == '__main__':
    n_samples = 50
    
    X0 = np.random.normal(0, 2, n_samples) 
    X1 = np.random.normal(10, 3, n_samples) 
    X2 = np.random.normal(-10, 2, n_samples) 
    X_tot = np.stack((X0,X1,X2)).flatten()

    mean_orig = [np.mean(X0), np.mean(X1), np.mean(X2)]
    sd_orig = [math.sqrt((np.var(X0))), math.sqrt((np.var(X1))), math.sqrt((np.var(X2)))]
    gauss_orig = [norm(mean_orig[i],sd_orig[i]) for i in range(3)]
    
    num_gaussian = 3
    gmm_one_d = GaussianMMOneD(num_gaussian)
    num_iteration = 50
    gmm_one_d.train(X_tot, num_iteration)

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ind = np.argsort(gmm_one_d.mu_list)
    mu_list = gmm_one_d.mu_list[ind]
    sd_list = gmm_one_d.sd_list[ind]
    print(mu_list, sd_list)

    axis_title = '| '
    for i in range(num_gaussian):
        axis_title += 'mu' + str(i+1) + '=' + str(mu_list[i]) + ', sd' + str(i+1) + '=' + str(sd_list[i]) + ' | '

    ax.set_title(axis_title)

    ax.set_xlabel('$x$')
    ax.set_ylabel('pdf')

    ax1.set_xlabel('iterations')
    ax1.set_ylabel('log-likelihood')
    

    num_bins = int(0.6*len(X_tot))
    ax.hist(X_tot, num_bins, normed=1, facecolor='blue', alpha=0.5)

    ax1.plot(range(len(gmm_one_d.log_likelihood)), gmm_one_d.log_likelihood)

    plot_input_x = np.linspace(-20,20,num=200)
    plot_input_y_orig = []
    plot_input_y_gmm = []

    for x in plot_input_x:
        plot_input_y_orig.append(np.max([gauss_orig[c].pdf(x) for c in range(num_gaussian)]))
        plot_input_y_gmm.append(gmm_one_d.gmm_pdf(x))
    
    ax.plot(plot_input_x, plot_input_y_orig, c='green', label='Original GMM PDF')
    ax.plot(plot_input_x, plot_input_y_gmm, c='orange', label='Calculated GMM PDF')

    ax.scatter(X_tot, 0.05*np.ones((len(X_tot),1)), c='b', label='data point')

    ax.legend()
    plt.savefig('sample_em_learning_model.png', bbox_inches='tight') 
    plt.show()