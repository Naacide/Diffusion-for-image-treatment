# Image Diffusion Filters

This repository contains a Python library for image processing using diffusion equations.


The project demonstrates various image filtering techniques based on partial differential equations (PDEs) and numerical integration.

## Features

The library provides several filters for color images:

- **Diffusion filter (`RKimage`)**: Standard diffusion-based smoothing.
- **Gradient-based diffusion (`RKimage_normgrad`)**: Filter using the gradient norm.
- **Laplacian-based diffusion (`RKimage_normlaplace`)**: Filter using the Laplacian norm.
- **Brightness modification (`RKimage_luminosité`)**: Custom brightness adjustment.
- **Random pattern / "Prince de Galles" (`RKimage(fPdeGalles)`)**: Generates a random pattern for artistic effect.
- **Anisotropic diffusion (`fani`)**: Optional function for advanced diffusion (available in the library).

## Requirements

- Python 3.7+
- numpy
- opencv-python
- tqdm (for progress bars)

Install dependencies with:

```bash
pip install numpy opencv-python tqdm
