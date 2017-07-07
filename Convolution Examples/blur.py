import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
import numpy as np

#Read the lena image
img = mpimg.imread('lena.png')
plt.imshow(img)
plt.show()

#Convert to black and white
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

#Create a 2d guassian image
guassian_filter = np.zeros((20,20))

for i in range(20):
	for j in range(20):
		dist = (i-9.5)**2 + (j-9.5)**2
		guassian_filter[i,j] = np.exp(-dist/100)

plt.imshow(guassian_filter, cmap='gray')
plt.show()

#Do the convolution and save the file
out = convolve2d(bw, guassian_filter)
mpimg.imsave('blur_lena.png', out, cmap='gray')