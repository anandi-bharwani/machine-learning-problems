import numpy as np 
import theano
import theano.tensor as T 
from util_parity import parity_pairs_with_labels, init_weights


def y2indicator(Y):
    N = len(Y)
    K = 2
    Y_ind = np.zeros([N, K])
    Y = Y.astype(np.int32)
    for i in range(N):
        Y_ind[i, Y[i]] = 1
    return Y_ind


class RNN(object):
	def __init__(self, M):
		self.M = M

	def fit(self, X, Y, lr=0.0001, mu=0.99):
		#print(X.shape)
		D = X[0].shape[1]			# X -> N x T x D matrix
		N = len(Y)
		K = len(set(Y.flatten()))	# K=2
		lr = np.float32(lr)
		mu = np.float32(mu)

		print("D:", D, "M:", self.M, "K:", K)

		#Weights
		Wx = init_weights(D,self.M).astype(np.float32)
		Wh = init_weights(self.M,self.M).astype(np.float32)
		bh = np.zeros(self.M).astype(np.float32)
		h0 = np.zeros(self.M).astype(np.float32)
		Wo = init_weights(self.M,K).astype(np.float32)
		bo = np.random.randn(K).astype(np.float32)

		#Theano variables
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)

		self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
		
		thX = T.fmatrix('X')		# 3 x 1
		thY = T.imatrix('Y')		# 3 x 2

		dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]

		#Scan the sequence
		def recurrence(x_t, h_t_prev):
			h_t = T.tanh(x_t.dot(self.Wx) + h_t_prev.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h , y],_ = theano.scan(
				fn=recurrence,
				sequences = thX,
				outputs_info = [self.h0,None],
				n_steps = thX.shape[0],
			)

		pY = y[:,0,:]		# T x D(1) x K(2)
		pred = T.argmax(pY, axis=1)
		#cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
		cost = - (thY * T.log(pY)).sum()

		#weight and momentum parameter updates
		grads = [T.grad(cost, p) for p in self.params]
		updates = [
			(p, p + mu*d - lr*g) for p,d,g in zip(self.params, dparams, grads)
			] + [
			(d, mu*d - lr*g) for d,g in zip(dparams, grads)
			]

		#Train function
		train = theano.function(
				inputs=[thX, thY],
				outputs=[pred, cost, y],
				updates=updates,
			)

		#Loop through the inputs
		for i in range(100):
			tot_cost = 0
			n_correct = 0
			for n in range(N):
				Y_n = y2indicator(Y[n].flatten()).astype(np.int32)
				pY, c, y= train(X[n], Y_n)
				#print(pY.shape, y.shape)
				#print(pY[-1], Y[n, -1])
				if pY[-1]==Y[n,-1]:
					n_correct+=1
				tot_cost+=c
			print("Iteration:", i, "cost: ", tot_cost, "classification_rate:", n_correct/N)



def main():
	X,Y = parity_pairs_with_labels(12)

	#X -> N x T x D
	#Y -> N x T

	model = RNN(10)
	model.fit(X,Y)

if __name__=='__main__':
	main()