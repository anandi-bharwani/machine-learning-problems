import numpy as np 
import theano.tensor as T 
import theano
from util_parity import init_weights, y2indicator, parity_pairs, error_rate
    
  
class HiddenLayer(object):
  def __init__(self, M1, M2, i):
    self.M1 = M1
    self.M2 = M2
    W = init_weights(M1, M2)
    b = np.zeros(M2)
    self.W = theano.shared(W, 'W_%s' %i)
    self.b = theano.shared(b, 'b_%s' %i)
    self.params = [self.W, self.b]
    
  def forward(self, X):
  	return T.nnet.relu(X.dot(self.W) + self.b)
    
class ANN(object):
  def __init__(self, hidden_layer_sizes):
    self.hidden_layer_sizes = hidden_layer_sizes
    
  def forward(self, X):
    z = X
    for h in self.hidden_layers:
      z = h.forward(z)
    return T.nnet.softmax(z.dot(self.W) + self.b)
    
  def fit(self, X, Y, lr=0.01, mu=0.99, batch_sz=50):
    Y = Y.astype(np.int32)
    N, D = X.shape
    K = len(set(Y))
    
    #Create the hidden layers
    self.hidden_layers = []
    m1 = D
    count = 0
    for m2 in self.hidden_layer_sizes:
      h = HiddenLayer(m1, m2, count)
      self.hidden_layers.append(h)
      m1 = m2 
      count+=1
      
    W = init_weights(m2, K)    #Logistic reg layer
    b = np.zeros([K])
    
    #Create theano variables
    thX = T.matrix('X')
    thY = T.ivector('Y')
    
    self.W = theano.shared(W, 'W_log')
    self.b = theano.shared(b, 'b_log')
    
    #Create parameter array for updates
    params = [self.W, self.b]
    for h in self.hidden_layers:
      params += h.params
    
    #Momentum parameters
    dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in params]

    #Forward pass
    pY = self.forward(thX)
    P = T.argmax(pY, axis=1)
    #cost = -(thY * T.log(pY)).sum()
    cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))

    #Weight updates
    updates = [
    		(p, p + mu*d - lr*T.grad(cost, p)) for p,d in zip(params, dparams)
   		] + [
    		(d, mu*d - lr*T.grad(cost, p)) for p,d in zip(params, dparams)
    	]
    #Theano function for training and predicting and calculating cost
    train = theano.function(
        inputs=[thX, thY],
        updates=updates,
      )
      
    get_cost_prediction = theano.function(
        inputs=[thX, thY],
        outputs=[P, cost]
      )
    
    #Loop for Batch grad descent
    no_batches = int(N/batch_sz)
    for i in range(5000):
      for n in range(no_batches):
        Xbatch = X[n*batch_sz:(n*batch_sz+batch_sz)]
        Ybatch = Y[n*batch_sz:(n*batch_sz+batch_sz)]
        #print(Xbatch.shape, Ybatch.shape)
        train(Xbatch, Ybatch)
        if n%100==0:
          P, c = get_cost_prediction(Xbatch, Ybatch)
          #print(P.shape, Ybatch.shape)
          er = error_rate(P, Ybatch)
          print("iteration:", i, "cost:", c, "error rate:", er)

def model1(X, Y):
  model = ANN([2048])
  model.fit(X, Y)

def model2(X, Y):
  model = ANN([1024]*2)
  model.fit(X, Y)

def model3(X, Y):
  model = ANN([50]*5)
  model.fit(X, Y)
  

def main():
  X,Y = parity_pairs(6)
  #model1(X,Y)
  #model2(X,Y)
  model3(X,Y)

main()
