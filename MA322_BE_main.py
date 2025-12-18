# -*- coding: utf-8 -*-
"""
This script demonstrates how to apply one of the available image filters.
Other filters are available and can be applied in the same way.

Available filters:
- RKimage: diffusion filter
- RKimage_normgrad: gradient-based diffusion
- RKimage_normlaplace: Laplacian-based diffusion
- RKimage_luminosité: brightness modification
- RKimage (fPdeGalles): "Prince de Galles" random pattern


Example image files available (place them in the same folder as this script):
- image1.jpg      # Eagle
- image2.jpg      # La Nuit étoilée (Van Gogh)
- image3.jpg      # Road 

@author : Nathan_AZO
"""

import cv2
from MA322_BE_lib import *  # Import one filter as example

# Load an example image
U0 = cv2.imread('image1.jpg')

# Display the original image
cv2.imshow("Original Image", U0)

# -----------------------
# Choose one filter to apply
# -----------------------
output = cv2.convertScaleAbs(RKimage(f, U0, t0=0, h=0.01, nbiter=100))  # Diffusion filter (example)
# output = cv2.convertScaleAbs(RKimage_normgrad(fgrad, U0, t0=0, h=0.01, nbiter=100))  # Gradient-based diffusion
# output = cv2.convertScaleAbs(RKimage_normlaplace(flaplace, U0, t0=0, h=0.01, nbiter=100))  # Laplacian-based diffusion
# output = cv2.convertScaleAbs(RKimage_luminosité(fluminosité, U0, t0=0, h=0.01, nbiter=10))  # Brightness modification
# output = cv2.convertScaleAbs(RKimage(fPdeGalles, U0, t0=0, h=0.01, nbiter=10))  # "Prince de Galles" random pattern

# Display the filtered image
cv2.imshow("Filtered Image (Diffusion)", output)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
