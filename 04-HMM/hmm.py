# Author: Kaituo Xu, Fan Yu, Wen Zhang

import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0

    # Begin Assignment
    a = np.array(A)
    b = np.array(B)

    BB = [[0.5, 0.5, 0.5],
         [0.4, 0.6, 0.4],
         [0.7, 0.3, 0.7]]
    bb = np.array(BB)

    pi = np.array(pi)

    alpha = np.zeros(((N, T)))
    # initialization
    alpha [:,0] = pi[:] * bb[:,0]
    #print ("alpha initialization:",alpha)

    # from 2 to T, j means time, and i means state
    for j in range(1,T):
        for i in range(N):
            for ii in range (N):
                alpha[i,j] += alpha[ii,j-1] * a[ii,i]
            alpha[i,j] = alpha[i,j] * bb[i,j]
    print ("alpha:",alpha)

    for i in range (N): 
        prob +=alpha[i,2]
    # End Assignment

    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    a = np.array(A)
    b = np.array(B)

    BB = [[0.5, 0.5, 0.5],
         [0.4, 0.6, 0.4],
         [0.7, 0.3, 0.7]]
    bb = np.array(BB)
    pi = np.array(pi)

    beta = np.zeros(((N, T)))

    # initialization
    beta [:,T-1] = [1,1,1]
    #print ("beta initialization:",beta)
    
    # from T to 2, j means time, and i means state
    for j in range(T-2,-1,-1):
        for i in range(N):
            for ii in range (N):
                beta[i,j] += beta[ii,j+1] * a[i,ii] * bb[ii,j+1]
    print ("beta:",beta)
    for i in range (N): 
        prob +=beta[i,0]*bb[i,0]*pi[i]
    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob = 0
    best_path = np.zeros(((T)))
    # Begin Assignment

    a = np.array(A)
    b = np.array(B)

    BB = [[0.5, 0.5, 0.5],
         [0.4, 0.6, 0.4],
         [0.7, 0.3, 0.7]]
    bb = np.array(BB)
    pi = np.array(pi)

    theta = np.zeros(((N, T)))

    # initialization
    theta [:,0] = pi[:] * bb[:,0]
    phi  = np.zeros(((N, T)))  
    #print ("theta initialization:",theta)
    temp = np.zeros(((N)))

    # from 2 to N
    for j in range(1,T):
        for i in range(N):
            for ii in range (N):
                temp[ii] = theta[ii,j-1] * a[ii,i]

            temp_max = max(temp)
            index_max = np.argmax(temp)+1           
            theta[i,j] = temp_max * bb[i,j]
            phi[i,j] = index_max
           
    print ("theta:",theta)
    print ("phi:",phi)
 
    for i in range (N): 
        for j in range (T-1,-1,-1):
            prob = max (theta[:,j])
            best_path[j] = np.argmax(theta[:,2])+1
    # End Assignment
    best_prob = max (theta[:,2])
    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]

    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)
