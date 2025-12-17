# -*- coding: utf-8 -*-
"""
Example usage of MA322_BE library

Example image files available (place them in the same folder as this script):
- image1.jpg      # Eagle
- image2.jpg      # La Nuit étoilée (Van Gogh)
- image3.jpg      # Road 

@author : Nathan_AZO
"""

import cv2
from MA322_BE_lib import RKimage, f

# Example image
U0 = cv2.imread('image1.jpg')

cv2.imshow("Before filter", U0)
output = cv2.convertScaleAbs(RKimage(f, U0, t0=0, h=0.01, nbiter=100))
cv2.imshow("After filter", output)

cv2.waitKey(0)
cv2.destroyAllWindows()

