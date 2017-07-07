import numpy as np

# def convolve2d(X,W):
# 	n1, n2 = X.shape
# 	m1, m2 = W.shape
# 	con2d = np.zeros((n1+m1-1,n2+m2-1))
# 	print(con2d.shape)
# 	for i1 in range(n1):
# 		for i2 in range(m1):
# 			for j1 in range(n2):
# 				for j2 in range(m2):
# 					if i1>=i2 and j1>=j2 and i1-i2<n1 and j1-j2<n2:
# 						con2d[i1,j1] += W[i2,j2]*X[i1-i2,j1-j2]
# 	return con2d



def convolve2d(X,W):
	n1, n2 = X.shape
	m1, m2 = W.shape
	con2d = np.zeros((n1+m1-1,n2+m2-1))
	for i in range(n1):
		for j in range(n2):
			con2d[i:i+m1,j:j+m2] += W*X[i,j]

	Y = con2d[m1//2:-m1//2+1,m2//2:-m2//2+1]		#To makeoutput same size as input
	print(con2d)
	assert(Y.shape == X.shape)
	return Y

# X = np.random.randn(2,2)
# W = np.random.randn(5,5)

# print( convolve2d(X,W) )