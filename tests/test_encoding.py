"""Properties the method rests on: rigid invariance, mate traversal, tie rule."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datc import stats as ST  # noqa: E402
from datc.alphabet import Alphabet  # noqa: E402
from datc.descriptors import darboux_ab, descriptors, frenet_kt  # noqa: E402


def helix_on_cylinder(n=200, r=1.0, pitch=0.3, turns=1.5):
    t = np.linspace(0, 2 * np.pi * turns, n)
    P = np.c_[r * np.cos(t), r * np.sin(t), pitch * t]
    N = np.c_[np.cos(t), np.sin(t), np.zeros(n)]       # cylinder normal
    return P, N


def rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_rigid_invariance():
    rng = np.random.default_rng(0)
    P, N = helix_on_cylinder()
    a0, b0, k0, t0 = descriptors(P, N, 0.05, sm=2.0, closed=False)
    R, t = rotation(rng), rng.normal(size=3)
    a1, b1, k1, t1 = descriptors(P @ R.T + t, N @ R.T, 0.05, sm=2.0, closed=False)
    for u, v in ((a0, a1), (b0, b1), (k0, k1), (t0, t1)):
        assert np.allclose(u, v, atol=1e-9)


def test_beta_zero_on_plane():
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    P = np.c_[np.cos(t), np.sin(t), np.zeros(100)]
    N = np.tile([0.0, 0.0, 1.0], (100, 1))
    _, b = darboux_ab(P, N)
    assert np.abs(b).max() < 1e-12


def test_mate_traversal_negates_alpha():
    """Walking the curve backwards on the same surface negates alpha and keeps beta."""
    P, N = helix_on_cylinder()
    a, b = darboux_ab(P, N)
    ar, br = darboux_ab(P[::-1], N[::-1])
    assert np.allclose(ar, -a[::-1], atol=1e-12)
    assert np.allclose(br, b[::-1], atol=1e-12)


def test_symmetric_alphabet_reverse_is_exact():
    rng = np.random.default_rng(1)
    al = Alphabet(9, 5).fit([rng.normal(0.1, 1, 500)], [rng.normal(size=500)])
    a, b = rng.normal(size=50), rng.normal(size=50)
    lhs = al.reverse(al.encode(a, b))
    rhs = al.encode(-a[::-1], b[::-1])
    # mirroring the bin equals negating alpha except exactly on a bin edge
    assert (lhs == rhs).mean() == 1.0


def test_frenet_helix_constants():
    r, c = 1.0, 0.3
    t = np.linspace(0, 6 * np.pi, 3000)
    P = np.c_[r * np.cos(t), r * np.sin(t), c * t]
    ds = float(np.linalg.norm(np.diff(P, axis=0), axis=1).mean())
    k, tau = frenet_kt(P, ds)
    assert np.allclose(np.median(k), r / (r * r + c * c), rtol=1e-3)
    assert np.allclose(np.median(tau), c / (r * r + c * c), rtol=1e-2)


def test_expected_tie_rule():
    S = np.array([[-np.inf, 1.0, 1.0, 1.0, 0.0],
                  [1.0, -np.inf, 0, 0, 0],
                  [1.0, 0, -np.inf, 0, 0],
                  [1.0, 0, 0, -np.inf, 0],
                  [0.0, 0, 0, 0, -np.inf]])
    q = ST.per_query(S, {0: 1})
    assert q[0]["better"] == 0 and q[0]["tied"] == 2
    assert abs(q[0]["top1"] - 1 / 3) < 1e-12
    assert q[0]["mid"] == 2.0


def test_sign_test():
    assert abs(ST.sign_test(3, 10) - 0.34375) < 1e-12
    assert ST.sign_test(0, 0) == 1.0
