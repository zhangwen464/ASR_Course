# Author: Sining Sun , Zhanheng Yang,Wen Zhang

import numpy as np
from utils import *
import scipy.cluster.vq as vq
from scipy.stats import multivariate_normal

num_gaussian = 7
num_iterations = 20
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu , self.sigma , self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')   
        print ("the shape of data is:",data.shape) # is a 18593 * 39 array

        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)] # clusters = [[], [], [], [], []]
        for (l,d) in zip(labels,data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return mu , sigma , pi
    
    def gaussian(self , x , mu , sigma):
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]     # D dimension Guassion distribution
        det_sigma = np.linalg.det(sigma)      # a scaler is |sigma| 
        inv_sigma = np.linalg.inv(sigma + 0.0001)     # a matrix is the inv of sigma
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)


    def calc_log_likelihood(self , X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """
        log_llh = 0.0

        # firstly, we must update r(Znk)
        n_observ, n_clusters = len(X), len(self.pi)
        pdfs = np.zeros(((n_observ, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(X, self.mu[i], np.diag(self.sigma[i]))
        r_Znk = pdfs / pdfs.sum(axis=1).reshape(-1,1)

        # secondly, according to the r(Znk), we update the pi(k)
        pi = r_Znk.sum(axis=0) / r_Znk.sum()

        # thirdly, calculate the log likelihood of GMM
        n_observ, n_clusters = len(X), len(pi)
        pdfs = np.zeros(((n_observ, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, self.mu[i],np.diag(self.sigma[i]))
        log_llh = np.mean(np.log(pdfs.sum(axis=1)))

        return log_llh,r_Znk

    def em_estimator(self , X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model 
        """

        log_llh = 0.0
        log_llh, r_Znk = self.calc_log_likelihood(X)

        # firstly, udapte mu
        D = X.shape[1]
        n_clusters = len(self.pi)  #n_clusters = r_Znk.shape[1]
        mu = np.zeros((n_clusters,D))
        for i in range(n_clusters):
            mu[i] = np.average(X, axis=0, weights = r_Znk[:, i])

        #secondly, udapte sigma
        n_clusters = len(self.pi)
        sigma = np.zeros((n_clusters, D))
        for i in range(n_clusters):
            sigma[i] = np.average((X - mu[i]) ** 2, axis=0, weights = r_Znk[:, i])
        
        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
