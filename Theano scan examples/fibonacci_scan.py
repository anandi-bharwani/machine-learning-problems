import theano
import theano.tensor as T


N  = T.iscalar('n')

def fibonacci(n, x, x_prev):
  return x+x_prev, x
  

outputs, updates = theano.scan(
      fn=fibonacci,
      sequences=T.arange(N),
      n_steps=N,
      outputs_info=[1, 1],
   )
   
fib_op = theano.function(
      inputs=[N],
      outputs=outputs
  )


print(fib_op(5))