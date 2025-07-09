## Introduction to EEG and Machine Learning
This is the beginning of my notes for EEG signal processing and neural networks. I will mainly be taking notes from 2 sources 1) EEG Signal Processing and Machine Learning Textbook, 2) Coursera Courses about neural networks.

```python
print("Hello, GitHub!")
```

# Neural Networks and Computer Vision (UMICH EE CS) July 6th
Hubel and Wiesel, 1959 - different electrical recordings of the cat's brain when shown different slides.  
They discovered simple cells - responsible for detecting edges at particular positions in the cat's visual field. Neuron may fire when edge is detected. Complex cells - light orientation and movement. Hypercomplex cells - movement.

Hierarchichal representation of visual system. Building from simple cells to complex cells! Later we will do MNIST which will be very helpful.
Larry Roberts, 1963 - original picture -> differentiated picture -> feature points selected -> 3d gemometry.
1970s - detect not just basic shapes and edges, pictorial structures
1980s - recognition via edge detection. Matching edges through comparing edges in images.
1990s - recognition via grouping semantically meaningful chunks.
2000s - extracting feature vectors (that somehow presersves info about the image) that can later be compared and manipulated.
- Face detection (algorithms)
2010s - Hinton - deep convolutional neural network called AlexNet
History of Deep Learning
Perceptron - it was a physical machine! The weights were potentiometers (I suppose changing electric current with respect to how important a signal is)
- linear classifier
Multi-layer perceptron - can learn many types of functions
1986 - hinton, method for training multi-layer percpetrons through back propagation. m - alpha * dL/dm
CNN -  is a specialized deep learning architecture designed for processing structured grid data, such as images, videos, and even audio spectrograms. CNNs are particularly effective for tasks like image classification, object detection, segmentation, and more, due to their ability to automatically learn hierarchical features from input data.

# Intro to PyTorch (NTU Lectures) 
PyTorch
- a machine learning framework in Python
Two nice features: 
- 1. n-dimensional tensor computation on GPUs (good for parallel processing)
- 2. automatic differentiation for training deep neural nets
3 things go into training a neural network: define the NN, the lost function, optimization algorithm.
Training -> Validation (loop) -> Testing
<img width="904" alt="Screenshot 2025-07-06 at 1 40 05 PM" src="https://github.com/user-attachments/assets/3c2d9939-dddd-4905-8250-397636279a8d" />

```python
import tensorflow as tf
x = torch.tensor ([[1,-1],[-1,1]]) 
x = torch.from_numpy (np.array[[1,-1],[-1,1]])

x = torch.zeros ([2,2])
x = torch.ones([1,2,5])
```
It's very easy to calculate the gradient of a function in tensor. For example:


```python
import tensorflow as tf
x = torch.tensor ([[1.,0.],[-1.,1.]]) 
z = x.pow(2).sum()
z.backward () # This step calculates the partial derivative of z with respect to x. ∂z/∂xᵢ for each element xᵢ in x. Since z = x₁² + x₂² + x₃² + x₄²,  ∂z/∂xᵢ = 2xᵢ = 2 times the whole matrix.
z.backward stores the partial derivative matrix inside x.grad.
x.grad #This step gives the gradient matrix of the partial derivative of z w.r.t. x
```
As you can see, there are a couple steps invovled in our z.backward() function. z.backward() does not just differentiate the scalar, but goes back to represent each x as a variable, and then gets the expression of z. Then finds the partial derivatives of z, then computes the gradient matrix based on that.

PyTorch’s backward() is designed to start from a scalar by default. When z is a scalar, PyTorch implicitly assumes you want to compute the gradient of z with respect to all inputs (x).

# Lecutre 1 Notes - Image Classification
Each color is represented by a number between 0 and 255.
The image is a big matrix consisting of those numbers in a matrix. Each pixel is one number.
There are several challenges to image classification, for example: Viewpoint (camera angle) variation, interclass variation, fine-grained categories (like different species of cats), background clutter, illumination changes (light), deformation (objects in image may be in diff. poses), when the object is blocked)

Image classification is also an important building block towards other things. e.g. object detection.

Process for Machine Learing: the data driven approach.
We first need a large set of images as training data. e.g. cat images vs. dog images. Next, we need human labels on these images. Two parts:
1. train: we input images and their associated labels, and we form a prediction algorithm.
2. predict. input the model we made for prediction, and new images we wanna classify.

 # Lecture 2 Notes - (UM EECS)
 We will work with CIFAR10. Mediumly challenging.

 First algorithm: Nearest Neighbor

 Ok right now, I will pause. The UM Course seem to be a bit too challenging for now, especially the coding and OOP. I would like to switch to a more basic course (Andrew Ng deep learning to build up foundations. Let's go.

 



 
 
