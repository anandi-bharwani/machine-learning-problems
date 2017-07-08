# machine-learning-problems
This repository contains all the basic machine learning examples which I had written as part of learning the various concepts.

Linear Regression, Logistic Regression and Multi-Class Logistic Regression are the excercises I implemented as part of the Machine Learning online course by Andrew Ng.

In 'ANN on a random guassian distribution', I created 3 random guassian distributions centered around different points which served as 3 classes for classification. I then used backpropagation with cross entropy cost function to train the Neural Network.

In 'Convolution Examples' there are three examples of how convolution can be used to alter image/audio files:-
1. Echo effect
2. Blurring of an image
3. Edge detection in an image 
This i did to get an intuition of how using Convolution in an ANN helps in extracting features.

'Theano scan example' contains simple examples which uses the theano Scan function. This was done to get an understanding of how theano scan functions work.

In 'Parity problem using RNN' I looked to solve the common parity problem where you output 1 if total number of 1's in the sequence is odd, 0 otherwise.  
1. ANN could solve the problem for smaller sequences, but failed if no of bits in the sequence was large. To solve the larger sequences the ANN had to made deeper and deeper.
2. On the other hand, a simple RNN with just one hidden layer was able to solve the parity problem for much longer sequences. This was because of its ability to treate input as sequences rather than independent data.

