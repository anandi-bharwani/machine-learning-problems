import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
#from scipy.signal import convolve2d
from convolve2d import convolve2d

#Read the lena image
img = mpimg.imread('lena.png')
plt.imshow(img)
plt.show()

#Convert to black and white
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

#Filter for edge detection
Hx = np.array([
	[-1,0,1],
	[-2,0,2],
	[-1,0,1],
], dtype=np.float32)

Hy = Hx.T

Gx = convolve2d(bw,Hx)		#Calculates horizontal edges
Gy = convolve2d(bw,Hy)		#Calculates vertical edges

#Adding them together
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()
mpimg.imsave('edge.png', G, cmap='gray')

theta = np.arctan2(Gy,Gx)
plt.imshow(theta)
plt.show()