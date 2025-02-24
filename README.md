# PyCNN: Image Processing with Cellular Neural Networks in Python

[![Build Status](https://travis-ci.org/ankitaggarwal011/PyCNN.svg?branch=master)](https://travis-ci.org/ankitaggarwal011/PyCNN)
[![Coverage Status](https://codecov.io/gh/ankitaggarwal011/PyCNN/coverage.svg?branch=master)](https://codecov.io/gh/ankitaggarwal011/PyCNN)

**Cellular Neural Networks (CNN)** [[wikipedia]](https://en.wikipedia.org/wiki/Cellular_neural_network) [[paper]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7600) are a parallel computing paradigm that was first proposed in 1988. Cellular neural networks are similar to neural networks, with the difference that communication is allowed only between neighboring units. Image Processing is one of its [applications](https://en.wikipedia.org/wiki/Cellular_neural_network#Applications). CNN processors were designed to perform image processing; specifically, the original application of CNN processors was to perform real-time ultra-high frame-rate (>10,000 frame/s) processing unachievable by digital processors.

This python library is the implementation of CNN for the application of **Image Processing**.

**Note**: The library has been **cited** in the research published on [Using Python and Julia for Efficient Implementation of Natural Computing and Complexity Related Algorithms](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7168488), look for the reference #19 in the references section. I'm glad that this library could be of help to the community.

**Note**: Cellular neural network (CNN) must not be confused with completely different convolutional neural network (ConvNet).

![alt text](images/blockdiagram.png "CNN Architecture")

As shown in the above diagram, imagine a control system with a feedback loop. f(x) is the piece-wise linear sigmoid function. The control (template B) and the feedback (template A) templates (coefficients) are configurable and controls the output of the system. Significant research had been done in determining the templates for common image processing techniques, these templates are published in this [Template Library](http://cnn-technology.itk.ppke.hu/Template_library_v4.0alpha1.pdf).

### Further reading:
- [Methods for image processing and pattern formation in Cellular Neural Networks: A Tutorial](http://ai.pku.edu.cn/aiwebsite/research.files/collected%20papers%20-%20others/Methods%20for%20image%20processing%20and%20pattern%20formation%20in%20Cellular%20Neural%20Networks%20-%20a%20tutorial.pdf)

## Motivation

This is an extension of a demo at 14th Cellular Nanoscale Networks and Applications (CNNA) Conference 2014. I have written a blog post, available at [Image Processing in CNN with Python on Raspberry Pi](http://blog.ankitaggarwal.me/technology/image-processing-with-cellular-neural-networks-in-python-on-raspberry-pi).
The library was used in my paper [B3: A plug-n-play internet enabled platform for real time image processing](http://ieeexplore.ieee.org/document/6888614/) published in IEEE Xplore.

## Dependencies

The library is supported for Python >= 2.7 and Python >= 3.3.

The python modules needed in order to use this library.
```
Pillow: 3.3.1
Scipy: 0.18.0
Numpy: 1.11.1 + mkl
```
Note: Scipy and Numpy can be installed on a Windows machines using binaries provided over [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs).

## Example 1

```sh
$ python example.py
```

#### OR

```python
from pycnn import PyCNN

cnn = PyCNN()
```

**Input:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/input1.bmp)

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/input3.bmp)

**Edge Detection:**

```python
cnn.edgeDetection('images/input1.bmp', 'images/output1.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output1.png)

**Grayscale Edge Detection**

```python
cnn.grayScaleEdgeDetection('images/input1.bmp', 'images/output2.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output2.png)

**Corner Detection:**

```python
cnn.cornerDetection('images/input1.bmp', 'images/output3.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output3.png)

**Diagonal line Detection:**

```python
cnn.diagonalLineDetection('images/input1.bmp', 'images/output4.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output4.png)

**Inversion (Logic NOT):**

```python
cnn.inversion('images/input1.bmp', 'images/output5.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output5.png)

**Optimal Edge Detection:**

```python
cnn.optimalEdgeDetection('images/input3.bmp', 'images/output6.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output6.png)

## Example 2

```sh
$ python example_lenna.py
```

#### OR

```python
from pycnn import PyCNN

cnn = PyCNN()
```

**Input:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna.gif)

**Edge Detection:**

```python
cnn.edgeDetection('images/lenna.gif', 'images/lenna_edge.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_edge.png)

**Diagonal line Detection:**

```python
cnn.diagonalLineDetection('images/lenna.gif', 'images/lenna_diagonal.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_diagonal.png)

## Usage

Import module

```python
from pycnn import PyCNN
```

Initialize object

```python
cnn = PyCNN()

# object variables: 
# m: width of the image (number of columns)
# n: height of image (number of rows)
```

```python
# name: name of image processing method (say, Edge detection); type: string
# inputImageLocation: location of the input image; type: string.
# outputImageLocation: location of the output image; type: string.
# tempA_A: feedback template; type: n x n list, e.g. 3 x 3, 5 x 5.
# tempB_B: control template; type: n x n list, e.g. 3 x 3, 5 x 5.
# initialCondition: initial condition, type: float.
# Ib_b: bias, type: float.
# t: time points for integration, type: ndarray. 
  # Note: Some image processing methods might need more time point samples than default.
  #       Display the output with each time point to see the evolution until the final convergence 
  #       to the output, looks pretty cool.
```

General image processing

```python
cnn.generalTemplates(name, inputImageLocation, outputImageLocation, tempA_A, tempB_B, 
                      initialCondition, Ib_b, t)
```

Edge detection

```python
cnn.edgeDetection(inputImageLocation, outputImageLocation)
```

Grayscale edge detection

```python
cnn.grayScaleEdgeDetection(inputImageLocation, outputImageLocation)
```

Corner detection

```python
cnn.cornerDetection(inputImageLocation, outputImageLocation)
```

Diagonal line detection

```python
cnn.diagonalLineDetection(inputImageLocation, outputImageLocation)
```

Inversion (Login NOT)

```python
cnn.inversion(inputImageLocation, outputImageLocation)
```

Optimal Edge Detection

```python
cnn.optimalEdgeDetection(inputImageLocation, outputImageLocation)
```

## License

[MIT License](https://github.com/ankitaggarwal011/PyCNN/blob/master/LICENSE)

## Contribute

Want to work on the project? Any kind of contribution is welcome!

Follow these steps:
- Fork the project.
- Create a new branch.
- Make your changes and write tests when practical.
- Commit your changes to the new branch.
- Send a pull request.
