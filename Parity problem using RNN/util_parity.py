import numpy as np 


def init_weights(M1, M2):
  return np.random.randn(M1,M2)/np.sqrt(M1+M2)
  
def error_rate(Y, T):
  return np.mean(Y!=T)

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    Y_ind = np.zeros([N, K])
    Y = Y.astype(np.int32)
    for i in range(N):
        Y_ind[i, Y[i]] = 1
    return Y_ind
  
def parity_pairs(n_bits):
  N = 2**n_bits
  n_total = N + 100 - N%100 #To make total number of samples a multiple of 100
  X = np.zeros([n_total, n_bits])
  Y = np.zeros([n_total])
  
  for i in range(n_total):
    num = i%N
    for j in range(n_bits):
      if num%(2**(j+1))!=0:
        num-=2**j
        X[i,j]=1
    Y[i] = X[i].sum()%2
  return X, Y

def parity_pairs_with_labels(n_bits):
	X, Y = parity_pairs(n_bits)
	Y_t = np.zeros(X.shape)
	N, D = X.shape

	for i in range(N):
		count = 0
		for j in range(D):
			if X[i,j] == 1:
				count+=1
			if count%2 == 1:
				Y_t[i,j] = 1

	X = X.reshape(N, D, 1).astype(np.float32)
	Y_t = Y_t.astype(np.int32)
	return X, Y_t
