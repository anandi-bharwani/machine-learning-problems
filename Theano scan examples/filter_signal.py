import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt

#Sin curve with noise
X = np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))

#PLot the input signal
plt.plot(X)
plt.show()

decay = T.scalar('d')
sequence = T.vector('s')

def recurrence(x, last, decay):
	return (1-decay)*x + decay*last

outputs, _  = theano.scan(
		fn=recurrence,
		sequences=sequence,
		n_steps=sequence.shape[0],
		outputs_info=[np.float64(0)],
		non_sequences=[decay],
	)

fil = theano.function(
		inputs=[sequence,decay],
		outputs=outputs,
	)

out = fil(X, 0.99)

plt.plot(out)
plt.show()
