# TRISEP 2024 - Machine Learning Tutorial and Excercises

## Prerequisites

Prerequisites for the course include basic knowledge of GitHub, Colab and python. It is thus required before the course to go through [these](https://github.com/makagan/TRISEP_Tutorial/blob/main/GettingStarted.pdf) slides as well as the following two python basics notebooks:

* [`python_intro_part1.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/python_basics/python_intro_part1.ipynb)
    * Quickstart
    * Indentation
    * Comments
    * Variables
    * Conditions and `if` statements
    * Arrays
    * Strings
    * Loops: `while` and `for`
    * Dictionaries
* [`python_intro_part2.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/python_basics/python_intro_part2.ipynb)
    * Functions
    * Classes/Objects
    * Inheritance
    * Modules
    * JSON data format
    * Exception Handling
    * File Handling

## Tutorials

A variety of tutorial notebooks below will introduce you to advanced python, PyTorch. The later excercises will not focus on PyTorch Geometric for using Graph Neural Networks or Decision Tree Models, but we have added a tutorial in case you would like to explore.

### General: Advanced Python

* Intro to Numpy: [`numpy_intro.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/python_advanced/numpy_intro.ipynb)
* Intro to Pandas: [`pandas_intro.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/python_advanced/pandas_intro.ipynb)
* Intro to Matplotlib: [`matplotlib_intro.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/python_advanced/matplotlib_intro.ipynb)

### General: Introduction to PyTorch

* Intro to PyTorch: [`pytorch_intro.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/pytorch_basics/pytorch_intro.ipynb) and [`pytorch_NeuralNetworks.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/pytorch_basics/pytorch_NeuralNetworks.ipynb)


### PyTorch Geometric (PyG)
* Intro to PyTorch Geometric: [`1.IntroToPyG.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/pytorch_geometric_intro/1.IntroToPyG.ipynb)
* Node classification with PyG on Cora citation dataset: [`2.KCNodeClassificationPyG.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/pytorch_geometric_intro/2.KCNodeClassificationPyG.ipynb)
* Graph classification with PyG on molecular prediction dataset: [`3.TUGraphClassification.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/pytorch_geometric_intro/3.TUGraphClassification.ipynb)

### Decision Trees with scikit-learn
* Decision tree models sxample: [`DecisionTrees.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/sklearn_trees/DecisionTrees.ipynb)


## Excercises

The excercises are organized by day to follow along with you lecture materials. Each notebook will have open questions and code for you to fill in. They are roughly numbered in the order to be explored. The solutions are also provided:

### Day 1: Linear Models
* Fitting models and bias-variance tradeoff: [`1.1.Fitting-and-Bias-Variance-Tradeoff.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/1.1.Fitting-and-Bias-Variance-Tradeoff.ipynb)
* Logistic and Linear Regression trained with gradient descent: [`1.2.Intro-LinearModels-SGD.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/1.2.Intro-LinearModels-SGD.ipynb)

### Day 2: Neural Networks
* Make a neural network by hand: [`2.1.First-NN.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/2.1.First-NN.ipynb)
* Train a neural network with PyTorch:[`2.2.Intro-NN-pytorch.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/2.2.Intro-NN-pytorch.ipynb)
* Train a neural network on the MNIST data set: [`2.3.NN-MLP-MNIST.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/2.3.NN-MLP-MNIST.ipynb)

### Day 3: Deep Neural Networks
* Train a multi-layer MLP: [`3.1.Going-Deeper.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/3.1.Going-Deeper.ipynb)
* Train a convolutional neural network on MNIST: [`3.2.ConvNet-MNIST.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/3.2.ConvNet-MNIST.ipynb)

### Day 4: Unsupervised Learning
* Apply principle components analysis to MNIST:[`4.1.PCA-MNIST.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/4.1.PCA-MNIST.ipynb)
* Train an autoencoder on MNIST: [`4.2.AutoEncoder-MNIST.ipynb`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/4.2.AutoEncoder-MNIST.ipynb)

### Day 5: Deep Generative Models
* Train a variation autoencoder and conditional variational autoencoder on MNIST: [`5.VariationalAutoEncoder-MNIST`](https://github.com/makagan/TRISEP_Tutorial/blob/main/Exercises/5.VariationalAutoEncoder-MNIST.ipynb)


## Other Resources

* Pattern Recognition and Machine Learning, Bishop (2006) -- ['link'](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
* Deep Learning, Goodfellow et al. (2016) -- [`link`](https://www.deeplearningbook.org/)
* Introduction to machine learning, Murray (2010) -- [`video lectures`](http://videolectures.net/bootcamp2010_murray_iml/)
* Stanford ML courses -- [`link`](https://ai.stanford.edu/stanford-ai-courses/)
* Francois Fleuret course on deep learning -- [`link`](https://fleuret.org/dlc/)
* Gilles Louppe course on deep learning -- [`link`](https://github.com/glouppe/info8010-deep-learning)
