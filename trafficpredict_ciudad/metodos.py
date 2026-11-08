"""Métodos numéricos implementados desde cero: Lagrange, Newton y mínimos cuadrados."""
import numpy as np


def lagrange(xs, ys, x):
    """P(x) = sum_i y_i * prod_{j!=i} (x - x_j)/(x_i - x_j)"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x = np.asarray(x, float)
    total = np.zeros_like(x)
    for i in range(len(xs)):
        li = np.ones_like(x)
        for j in range(len(xs)):
            if j != i:
                li *= (x - xs[j]) / (xs[i] - xs[j])
        total += ys[i] * li
    return total


def diferencias_divididas(xs, ys):
    """Tabla triangular de diferencias divididas (matriz n x n)."""
    n = len(xs)
    T = np.zeros((n, n))
    T[:, 0] = ys
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = (T[i + 1, j - 1] - T[i, j - 1]) / (xs[i + j] - xs[i])
    return T


def newton(xs, ys, x):
    """P(x) = a0 + a1(x-x0) + a2(x-x0)(x-x1) + ...  con a_k = f[x0..xk]"""
    xs = np.asarray(xs, float)
    x = np.asarray(x, float)
    T = diferencias_divididas(xs, np.asarray(ys, float))
    coef = T[0, :]
    res = np.full_like(x, coef[-1])
    for k in range(len(xs) - 2, -1, -1):           # esquema de Horner
        res = res * (x - xs[k]) + coef[k]
    return res, coef, T


def minimos_cuadrados(xs, ys, grado):
    """Ajuste polinómico por ecuaciones normales (con xs centrado para estabilidad)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x0 = xs.mean()
    A = np.vander(xs - x0, grado + 1)
    coef = np.linalg.solve(A.T @ A, A.T @ ys)
    pred = lambda x: np.vander(np.asarray(x, float) - x0, grado + 1) @ coef
    r = ys - pred(xs)
    return coef, pred, float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2)))
