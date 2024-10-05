"""
@author: Yujie Zhang
"""


import Bases
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from itertools import product
from pypoman import compute_polytope_vertices

'''This script is a collection of function and SDP for certifying and quantifying 
contextual measurement'''

def GellmannBasisElement(i, j, d):
    '''Converting effects into real vectors'''
    if i > j:
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = 1
        L[j - 1][i - 1] = 1
    elif i < j:
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = -1.0j
        L[j - 1][i - 1] = 1.0j
    elif i == j and i < d:
        L = np.sqrt(2 / (i * (i + 1))) * np.diag(
            [1 if n <= i else (-i if n == (i + 1) else 0) for n in range(1, d + 1)]
        )
    else:  # identity
        L = np.eye(d)
    return np.array(L / np.sqrt((L @ L).trace()))

def GelmannBasis(d):
    return [
        GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)
    ]


def Dvertices(effects):
    '''Enumerating all extreme point for the polytope determined by measurement operational indentity'''
    d = effects[0].shape[0]
    basis = GelmannBasis(d)
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    A=np.array([to_gellmann(v) for v in effects]).T.real
    alpha=null_space(A).T
    n = len(alpha[0])
    A_eq = np.vstack([np.ones(n), alpha])
    b_eq = np.concatenate([[1], np.zeros(len(alpha))])
    N = null_space(A_eq)
    A_eq_inv = np.linalg.pinv(A_eq)
    p0 = A_eq_inv @ b_eq
    A_ub = -np.eye(n)
    b_ub = np.zeros(n)
    A = A_ub @ N
    b = b_ub - A_ub @ p0
    vertices_lower_dim = compute_polytope_vertices(A, b)
    vertices = [p0 + N @ v for v in vertices_lower_dim]
    return np.array(vertices)


def jrdual(Ma):
    '''Dual SDP for certifying contextual measurement Eq.(18)'''
    dA = Ma.shape[2]
    k = Ma.shape[0]
    Fa = [cp.Variable((dA, dA), hermitian=True) for _ in range(k)]
    tr_Fsig = 0
    trF_trsig=0
    for j in range(k):
        sig_ax = Ma[j,:, :]
        tr_Fsig += cp.real(cp.trace(Fa[j] @ sig_ax))
        trF_trsig += cp.real(cp.trace(Fa[j]) * cp.trace(sig_ax))
    constraints = []
    D=Dvertices(Ma)
    for l in range(len(D)):
        U = cp.Constant(np.zeros((dA, dA)))
        for i in range(k):
             U = U+D[l][i]*Fa[i]
        constraints.append(U >> 0)

    constraints.append(1 + tr_Fsig - (1/dA) * trF_trsig >= 0)

    J = cp.Minimize(1+cp.real(tr_Fsig))
    problem = cp.Problem(J, constraints)
    problem.solve(solver=cp.MOSEK)
    F_a_value = np.array([Fa[j].value for j in range(k)])

    return F_a_value, problem.value
def jwdual(Ma):
    '''Dual SDP for quantifying weight of contextual measurement Eq.(24)'''
    dA = Ma.shape[2]
    k = Ma.shape[0]
    Fa = [cp.Variable((dA, dA), hermitian=True) for _ in range(k)]
    tr_Fsig = 0
    for j in range(k):
        sig_ax = Ma[j,:, :]
        tr_Fsig += cp.real(cp.trace(Fa[j] @ sig_ax))
    constraints = []
    D=Dvertices(Ma)
    for l in range(len(D)):
        U = cp.Constant(np.zeros((dA, dA)))
        for i in range(k):
             U = U+D[l][i]*Fa[i]
        constraints.append(U-np.eye(dA)/dA >> 0)
    for i in range(k):
        constraints.append(Fa[i]>> 0)

    J = cp.Maximize(1-cp.real(tr_Fsig))  # Ensure the objective is real
    problem = cp.Problem(J, constraints)
    problem.solve(solver=cp.MOSEK)
    F_a_value = np.array([Fa[j].value for j in range(k)])

    return F_a_value, problem.value



