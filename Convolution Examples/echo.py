#Here we convolve the helloworld audio file with two different filters
#1. Impulse function: Doing this diowe get back the same audio effect on the auio signal
#2. Echo Filter to create an echo effect

import wave
import numpy as np
import sys
import matplotlib.pyplot as plt
#from scipy.io.wavfile import write

#Read the plot audio file
f = wave.open("helloworld.wav", 'r')
data = np.fromstring(f.readframes(-1), dtype='Int16')
print("chaennels: ",f.getnchannels(), "sampwidth: ", f.getsampwidth(), "framrate: ", f.getframerate(), "nframes", f.getnframes())

f.close()

print(data.shape)
plt.plot(data)
plt.show()

#Perform convolution on the audio file and the impulse function
impulse = np.zeros(16000)
impulse[0] = 1

noecho = np.convolve(data, impulse)
noecho = noecho.astype(np.float32)

print(noecho.shape)			#Result will be same as the original
plt.plot(noecho)			#Although length increases
plt.show()

#Write to file
noecho = noecho.astype(np.int16)
f = wave.open("noecho.wav", 'w')
f.setparams((1,2,16000,0,'NONE','not compressed'))
f.writeframesraw(noecho)
f.close()

#Convolution to create echo
echoFilter = np.zeros(16000)
echoFilter[0]=1
echoFilter[3000]=0.8
echoFilter[5000]=0.6
echoFilter[9000]=0.3
echoFilter[12000]=0.2
echoFilter[15999]=0.1

with_echo = np.convolve(data, echoFilter)
with_echo = with_echo.astype(np.float32)

# print(with_echo.shape)			#Result will be same as the original
# plt.plot(with_echo)			#Although size increases
# plt.show()

#Write to file
with_echo = with_echo.astype(np.int16)
f = wave.open("helloworld_withecho.wav", 'w')
f.setparams((1,2,16000,0,'NONE','not compressed'))
f.writeframesraw(with_echo)
f.close()