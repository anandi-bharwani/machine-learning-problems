import theano
import theano.tensor as T 
import numpy as np 


x = T.vector('x')

# def square(x):
#   x*x 
  

outputs, updates = theano.scan(
      fn=lambda x:x*x,
      sequences=x,
      n_steps=x.shape[0]
   )
   
square_op = theano.function(
      inputs=[x],
      outputs=[outputs]
  )

A = np.array([1,2,3,4,5])
print(square_op(A))