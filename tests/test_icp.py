"""ICP recovers a small rigid offset of a curved surface patch, and RANSAC + ICP compose."""
import numpy as np
from scipy.spatial.transform import Rotation

from datc import pose as PO


def _patch(n=60):
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    return np.c_[u.ravel(), v.ravel(), 0.3 * u.ravel() ** 2 - 0.2 * v.ravel() ** 2 + 0.1 * u.ravel() * v.ravel()]


def _normals(Q):
    u, v = Q[:, 0], Q[:, 1]
    N = np.c_[-(0.6 * u + 0.1 * v), -(-0.4 * v + 0.1 * u), np.ones(len(Q))]
    return N / np.linalg.norm(N, axis=1, keepdims=True)


R_TRUE = Rotation.from_euler("xyz", [4, -3, 5], degrees=True).as_matrix()
T_TRUE = np.array([0.03, -0.02, 0.04])


def test_point_to_plane_recovers_small_offset():
    Q = _patch()
    P = (Q - T_TRUE) @ R_TRUE            # P moved by (R_TRUE, T_TRUE) lands on Q
    r = PO.icp(P, Q, np.eye(3), np.zeros(3), tau=0.5, dst_normals=_normals(Q))
    assert r is not None and r["converged"] and r["metric"] == "point-to-plane"
    assert PO.rotation_angle(r["R"] @ R_TRUE.T) < 0.2
    assert np.abs(r["t"] - T_TRUE).max() < 5e-3


def test_point_to_point_reduces_error():
    Q = _patch()
    P = (Q - T_TRUE) @ R_TRUE
    r = PO.icp(P, Q, np.eye(3), np.zeros(3), tau=0.5)
    start = PO.rotation_angle(R_TRUE)
    assert r["rms"] < r["rms0"] and PO.rotation_angle(r["R"] @ R_TRUE.T) < start / 3


def test_icp_refuses_without_overlap():
    Q = _patch()
    assert PO.icp(Q + 10.0, Q, np.eye(3), np.zeros(3), tau=0.1) is None
