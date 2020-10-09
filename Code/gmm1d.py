import numpy as np
import math
from scipy.stats import norm


class GaussianMMOneD:
    def __init__(self, number_gaussian=1, mu_list=None, sd_list=None, mix_list=None):
        self.num_gaussian = number_gaussian
        self.mu_list = mu_list
        self.sd_list = sd_list
        self.mix_list = mix_list
        self.gauss_mixture = None
        self.log_likelihood = []
           
    def train(self, data, num_iterations):
        eps=1e-8  # to take care of the singularities
        k = self.num_gaussian
        weights = np.ones((k)) / k
        means = np.random.choice(data, k)
        variances = np.random.random_sample(size=k)
        bins = np.linspace(np.min(data),np.max(data),100)
        log_likelihood_threshold = 0.0001
        for _ in range(num_iterations):

            # calculate the maximum likelihood of each observation xi
            likelihood = []

            # Expectation step
            for j in range(k):
                likelihood.append(weights[j]*self.norm_pdf(data, means[j], (variances[j])))
            likelihood = np.array(likelihood)


            sum_likelihood = np.sum(likelihood, axis=0) #np.sum([likelihood[i] for i in range(k)], axis=0)
            self.log_likelihood.append(np.sum(np.log(sum_likelihood+eps)))
            

            b = []
            # Maximization step 
            for j in range(k):
                # use the current values for the parameters to evaluate the posterior
                #   probabilities of the data to have been generanted by each gaussian    
                b.append((likelihood[j]) / (sum_likelihood+eps))

                # updage mean and variance
                means[j] = np.sum(b[j] * data) / (np.sum(b[j]+eps))
                variances[j] = np.sum(b[j] * np.square(data - means[j])) / (np.sum(b[j]+eps))

                # update the weights
                weights[j] = np.mean(b[j])
            
            # Check for convergence
            if len(self.log_likelihood) > 1:
                if abs(self.log_likelihood[-1] - self.log_likelihood[-2]) <= log_likelihood_threshold:
                    break
        
        
        self.mu_list = means
        self.sd_list = variances**0.5
        self.mix_list = weights
        self.update_mixture()
          
    def norm_pdf(self, data, mean, variance):
        s1 = 1/(np.sqrt(2*np.pi*variance))
        s2 = np.exp(-(np.square(data - mean)/(2*variance)))
        return s1 * s2
            
    def update_mixture(self):
        self.gauss_mixture = [norm(loc=mu,scale=(sd)) for mu,sd in zip(self.mu_list, self.sd_list)]

    def gmm_pdf(self, data):
        #return np.max([self.gauss_mixture[c].pdf(data) for c in range(self.num_gaussian)], axis=0)
        return np.sum([self.mix_list[c]*self.gauss_mixture[c].pdf(data) for c in range(self.num_gaussian)], axis=0)
        
