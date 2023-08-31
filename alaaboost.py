import numpy as np
import sys
import time
import math
#random.seed(7)


def CaraIdxCoreset(P, u,dtype = 'float64'):
    while 1 :
        n = np.count_nonzero(u)
        d = P.shape[1]
        u_non_zero = np.nonzero(u)
        if n <= d + 1: return P, u
        A = P[u_non_zero];
        reduced_vec = np.outer(A[0], np.ones(A.shape[0]-1, dtype = dtype))
        A = A[1:].T - reduced_vec

        idx_of_try = 0; const = 10000;  diff = np.infty ; cond = sys.float_info.min  

        _, _, V = np.linalg.svd(A, full_matrices=True)
        v=V[-1]
        diff = np.max(np.abs(np.dot(A, v ))) 
        v = np.insert(v, [0],   -1 * np.sum(v))


        idx_good_alpha = np.nonzero(v > 0)
        alpha = np.min(u[u_non_zero][idx_good_alpha]/v[idx_good_alpha])
       

        w = np.zeros(u.shape[0] , dtype = dtype)
        tmp = u[u_non_zero] - alpha * v
        tmp[np.argmin(tmp)] = 0.0
        w[u_non_zero] = tmp
        w[u_non_zero][np.argmin(w[u_non_zero] )] = 0
        u = w


    return CaraIdxCoreset(P, w)
def updated_cara(P,w,coreset_size, dtype = 'float64'):
    d = P.shape[1] ; n= P.shape[0]; m = 2*d +2;  #print (coreset_size,dtype)
    if n <= d + 1 : return (P,w, np.array(list(range(0,P.shape[0]))) ) #carateodory return coreset in size of at least d+1, so if smaller than C=P
    wconst = 1
    w_sum = np.sum(w)
    w = wconst* w/w_sum 
    chunk_size = math.ceil(n /m)
    current_m =  math.ceil( n/chunk_size)

    add_z = chunk_size - int (n%chunk_size)
    w = w.reshape(-1,1)
    f = time.time()
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype = dtype)
        P = np.concatenate((P,zeros ))
        f3 = time.time();
        zeros = np.zeros((add_z, w.shape[1]), dtype = dtype)
        w = np.concatenate((w, zeros))
    
    idxarray = np.array(range(P.shape[0]) )
    
    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    w_groups = w.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    w_nonzero = np.count_nonzero(w) ;counter = 1 ; #print (w_nonzero, w)

    if not coreset_size : coreset_size = d+1
    while w_nonzero > coreset_size:
        s0 = time.time()
        counter +=1
        groups_means = np.einsum('ijk,ij->ik',p_groups, w_groups)
        group_weigts = np.ones(groups_means.shape[0], dtype = dtype)*1/current_m
        
        Cara_p, Cara_w_idx = CaraIdxCoreset(groups_means , group_weigts,dtype = dtype )# real caratehodory

      
        IDX = np.nonzero(Cara_w_idx)

        new_P = p_groups[IDX].reshape(-1,d)

        new_w = (current_m * w_groups[IDX] * Cara_w_idx[IDX][:, np.newaxis]).reshape(-1, 1)
        new_idx_array = idx_group[IDX].reshape(-1,1)
        ##############################################################################3
        w_nonzero = np.count_nonzero(new_w)
        chunk_size = math.ceil(new_P.shape[0]/ m)
        current_m =  math.ceil(new_P.shape[0]/ chunk_size)

        add_z = chunk_size - int(new_P.shape[0] % chunk_size)
        if add_z != chunk_size :
            new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype = dtype)))
            new_w = np.concatenate((new_w, np.zeros((add_z, new_w.shape[1]),dtype = dtype)))
            new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]),dtype = dtype)))
        p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
        w_groups = new_w.reshape(current_m, chunk_size)
        idx_group = new_idx_array.reshape(current_m , chunk_size)
        ###########################################################
    
    return new_P, w_sum * new_w/wconst, new_idx_array.reshape(-1).astype(int)


def  check_cara_out(P,w, Cara_p,Cara_w_Idx, groups_means,group_weigts,w_sum,current_m,Cara_S, Cara_w ):

    Cara_w_Idx = Cara_w_Idx.reshape(-1,1)
    print(Cara_p.shape, Cara_w_Idx.shape,groups_means.shape,group_weigts.shape)
    print ("checker", groups_means.T.dot(group_weigts.reshape(-1,1)) - Cara_p.T.dot(Cara_w_Idx)); #input()   
