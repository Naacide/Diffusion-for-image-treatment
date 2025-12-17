# -*- coding: utf-8 -*-
"""
MA322_BE Library: Image processing using diffusion equations

@author : Nathan_AZO
"""

import numpy as np
import random
from tqdm import tqdm

def completion(U: np.ndarray) -> np.ndarray:
    """
    Extends a 2D array by replicating its border values.

    Parameters:
        U (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Extended array with replicated borders.
    """
    l, c = U.shape
    Ut = np.vstack((U[0, :], U))  # replicate first row at top
    Ut = np.vstack((Ut, U[l - 1, :]))  # replicate last row at bottom
    Ut = np.hstack((Ut[:, 0].reshape(l + 2, 1), Ut))  # replicate first column
    Ut = np.hstack((Ut, Ut[:, c].reshape(l + 2, 1)))  # replicate last column
    return Ut


def Dmat(n: int) -> np.ndarray:
    """
    Constructs a finite difference matrix for diffusion.

    Parameters:
        n (int): Size of the matrix.

    Returns:
        np.ndarray: (n x n+2) finite difference matrix.
    """
    D = np.eye(n, n + 2)
    for j in range(n + 2):
        for i in range(n):
            if i + 1 == j:
                D[i, j] = -2
            elif i + 2 == j:
                D[i, j] = 1
    return D


def f(t: float, U: np.ndarray) -> np.ndarray:
    """
    Computes the diffusion update for a 2D array.

    Parameters:
        t (float): Time parameter (not used here, for RK4 compatibility)
        U (np.ndarray): 2D input array

    Returns:
        np.ndarray: Diffusion update
    """
    l, c = U.shape
    return Dmat(l) @ completion(U)[:, 1:c + 1] + completion(U)[1:l + 1, :] @ Dmat(c).T



def RKimage(f, U0: np.ndarray, t0: float, h: float, nbiter: int) -> np.ndarray:
    """
    Applies RK4 integration to each color channel of an image with a progress bar.

    Parameters:
        f (function): Update function (e.g., diffusion)
        U0 (np.ndarray): Input image (HxWx3)
        t0 (float): Initial time
        h (float): Time step
        nbiter (int): Number of iterations

    Returns:
        np.ndarray: Processed image
    """
    l, c, d = U0.shape
    Us = np.zeros((l, c, d))
    for k in range(3):
        Ucouchek = U0[:, :, k]
        for i in tqdm(range(nbiter), desc=f"Channel {k+1}/3"):
            k1 = h * f(t0, Ucouchek)
            k2 = h * f(t0 + h / 2, Ucouchek + k1 / 2)
            k3 = h * f(t0 + h / 2, Ucouchek + k2 / 2)
            k4 = h * f(t0 + h, Ucouchek + k3)
            Ucouchek = Ucouchek + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Us[:, :, k] = Ucouchek
    return Us


def RKimage_normgrad(f, U0: np.ndarray, t0: float, h: float, nbiter: int) -> np.ndarray:
    l, c, d = U0.shape
    Us = np.zeros((l, c, d))
    for k in range(3):
        Ucouchek = U0[:, :, k]
        for i in tqdm(range(nbiter), desc=f"Channel {k+1}/3 (Gradient)"):
            k1 = h * f(t0, Ucouchek)
            k2 = h * f(t0 + h / 2, Ucouchek + k1 / 2)
            k3 = h * f(t0 + h / 2, Ucouchek + k2 / 2)
            k4 = h * f(t0 + h, Ucouchek + k3)
            Ucouchek = Ucouchek + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Us[:, :, k] = Ucouchek
    return Us


def RKimage_normlaplace(f, U0: np.ndarray, t0: float, h: float, nbiter: int) -> np.ndarray:
    l, c, d = U0.shape
    Us = np.zeros((l, c, d))
    for k in range(3):
        Ucouchek = U0[:, :, k]
        for i in tqdm(range(nbiter), desc=f"Channel {k+1}/3 (Laplace)"):
            k1 = h * f(t0, Ucouchek)
            k2 = h * f(t0 + h / 2, Ucouchek + k1 / 2)
            k3 = h * f(t0 + h / 2, Ucouchek + k2 / 2)
            k4 = h * f(t0 + h, Ucouchek + k3)
            Ucouchek = Ucouchek + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Us[:, :, k] = Ucouchek
    return Us


def RKimage_luminosité(f, U0: np.ndarray, t0: float, h: float, nbiter: int) -> np.ndarray:
    l, c, d = U0.shape
    Us = np.zeros((l, c, d))
    for k in range(3):
        Ucouchek = U0[:, :, k]
        for i in tqdm(range(nbiter), desc=f"Channel {k+1}/3 (Luminosity)"):
            k1 = h * f(t0, Ucouchek)
            k2 = h * f(t0 + h / 2, Ucouchek + k1 / 2)
            k3 = h * f(t0 + h / 2, Ucouchek + k2 / 2)
            k4 = h * f(t0 + h, Ucouchek + k3)
            Ucouchek = Ucouchek + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Us[:, :, k] = Ucouchek
    return Us



def Dgrad(n: int) -> np.ndarray:
    """
    Finite difference matrix for gradient calculation.
    """
    D = np.zeros((n, n + 2))
    for j in range(n + 2):
        for i in range(n):
            if i == j:
                D[i, j] = -0.5
            elif i + 2 == j:
                D[i, j] = 0.5
    return D


def fgrad(t: float, U: np.ndarray) -> np.ndarray:
    """
    Computes the gradient norm update for an image.
    """
    l, c = U.shape
    return np.sqrt((Dgrad(l) @ completion(U)[:, 1:c + 1]) ** 2 +
                   (completion(U)[1:l + 1, :] @ Dgrad(c).T) ** 2)


def flaplace(t: float, U: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian norm update for an image.
    """
    l, c = U.shape
    return np.sqrt((Dmat(l) @ completion(U)[:, 1:c + 1]) ** 2 +
                   (completion(U)[1:l + 1, :] @ Dmat(c).T) ** 2)


def Dluminosité(n: int) -> np.ndarray:
    """
    Custom finite difference matrix for brightness modification.
    """
    D = np.zeros((n, n + 2))
    for j in range(n + 2):
        for i in range(n):
            if i == j:
                D[i, j] = -5
            elif i + 2 == j:
                D[i, j] = 12
    return D


def fluminosité(t: float, U: np.ndarray) -> np.ndarray:
    """
    Brightness modification function.
    """
    l, c = U.shape
    return Dluminosité(l) @ completion(U)[:, 1:c + 1] + completion(U)[1:l + 1, :] @ Dluminosité(c).T


def fmod2(t: float, U: np.ndarray) -> np.ndarray:
    """
    Computes modified diffusion with fourth power.
    """
    l, c = U.shape
    return ((Dmat(l) @ completion(U)[:, 1:c + 1]) ** 4 + 
            (completion(U)[1:l + 1, :] @ Dmat(c).T) ** 4) ** 2


def DPdeGalles(n: int) -> np.ndarray:
    """
    Random difference matrix for "Prince de Galles" effect.
    """
    D = np.zeros((n, n + 2))
    for j in range(n + 2):
        for i in range(n):
            D[i, j] = random.randint(-2, 2)
    return D


def fPdeGalles(t: float, U: np.ndarray) -> np.ndarray:
    """
    Computes the Prince de Galles pattern update.
    """
    l, c = U.shape
    return DPdeGalles(l) @ completion(U)[:, 1:c + 1] + completion(U)[1:l + 1, :] @ DPdeGalles(c).T


def c(s: float, l: float) -> float:
    """
    Anisotropic diffusion coefficient.
    """
    return 1 / (1 + (s ** 2 / l ** 2))


def fani(t: float, U: np.ndarray) -> np.ndarray:
    """
    Anisotropic diffusion function.
    """
    l, c = U.shape
    return Dmat(l) @ completion(U)[:, 1:c + 1] + completion(U)[1:l + 1, :] @ Dmat(c).T
