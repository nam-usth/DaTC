"""Turn angles of a rim: Darboux (alpha, beta) and Frenet (kappa, tau).

alpha  turn of the tangent along g = n x t, inside the surface
beta   turn of the tangent along the surface normal n, out of the surface
kappa  curvature  |r' x r''| / |r'|^3              (second derivative)
tau    torsion    (r' x r'') . r''' / |r' x r''|^2  (third derivative)

alpha and beta are dimensionless angles per sample; kappa and tau carry units of
1/length. Both pairs are invariant under rigid motion.
"""
from __future__ import annotations

import numpy as np

from .geometry import smooth_curve


def darboux_ab(pts: np.ndarray, nrm: np.ndarray):
    """(alpha, beta), one value per interior sample (length n - 2)."""
    T = np.diff(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    if len(T) < 2:
        return np.zeros(0), np.zeros(0)
    t0, t1, nv = T[:-1], T[1:], nrm[1:len(T)]
    nv = nv - np.einsum("ij,ij->i", nv, t0)[:, None] * t0
    nv = nv / (np.linalg.norm(nv, axis=1, keepdims=True) + 1e-12)
    g = np.cross(nv, t0)
    dot = np.einsum("ij,ij->i", t1, t0)
    w = t1 - dot[:, None] * t0
    nw = np.linalg.norm(w, axis=1)
    th = np.arctan2(nw, dot)
    safe = nw > 1e-9
    den = np.where(safe, nw, 1.0)
    alpha = np.where(safe, th * np.einsum("ij,ij->i", w, g) / den, 0.0)
    beta = np.where(safe, th * np.einsum("ij,ij->i", w, nv) / den, 0.0)
    return alpha, beta


def frenet_kt(pts: np.ndarray, ds: float):
    """(kappa, tau) by three chained finite differences (length n - 2)."""
    if len(pts) < 5:
        return np.zeros(0), np.zeros(0)
    d1 = np.gradient(pts, ds, axis=0)
    d2 = np.gradient(d1, ds, axis=0)
    d3 = np.gradient(d2, ds, axis=0)
    cr = np.cross(d1, d2)
    ncr = np.linalg.norm(cr, axis=1)
    kap = ncr / (np.linalg.norm(d1, axis=1) ** 3 + 1e-12)
    tau = np.einsum("ij,ij->i", cr, d3) / (ncr ** 2 + 1e-12)
    return kap[1:-1], tau[1:-1]


def descriptors(P: np.ndarray, N: np.ndarray, ds: float, sm: float = 0.0,
                closed: bool = True):
    """(alpha, beta, kappa, tau) after Gaussian smoothing of the points.

    The points are smoothed and then differentiated; the normals are not
    smoothed. Smoothing the descriptors afterwards would be another operator.
    """
    Ps = smooth_curve(P, sm, closed)
    a, b = darboux_ab(Ps, N)
    k, t = frenet_kt(Ps, ds)
    return a, b, k, t
