# MA322_BE Image Processing Library

This repository contains a Python library for image processing using diffusion equations. 
It was developed as part of MA322 course by Nathan AZEDO, David BURGUN, and Laurène REINHART.

## Features

- Diffusion filtering on color images (RK4 integration)
- Gradient-based filtering
- Laplacian-based filtering
- Brightness modification
- Random pattern generation ("Prince de Galles")
- Anisotropic diffusion

## Requirements

- Python 3.7+
- numpy
- opencv-python

## Usage

```python
import cv2
from MA322_BE_lib import RKimage, f

# Load an image
U0 = cv2.imread('image1.jpg')

# Apply RK4 diffusion filter
output = cv2.convertScaleAbs(RKimage(f, U0, t0=0, h=0.01, nbiter=100))

cv2.imshow("Before filter", U0)
cv2.imshow("After filter", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
