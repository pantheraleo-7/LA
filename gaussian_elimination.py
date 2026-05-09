import numpy as np
from sympy import Symbol
from math import isclose

def det(mat):
    mat = np.array(mat, dtype=float)
    r, c = mat.shape
    if r!=c:
        raise ValueError('matrix must be square')
    
    val = 1
    for i in range(r):
        pivot = mat[i, i]
        for next_row in range(i+1, r):
            below_pivot = mat[next_row, i]
            if not isclose(below_pivot, 0, abs_tol=1e-08):
                if not isclose(pivot, 0, abs_tol=1e-08):
                    mat[next_row] -= (below_pivot/pivot)*mat[i]
                else:
                    mat[[i, next_row]] = mat[[next_row, i]]
                    pivot = below_pivot
                    val *= -1
        
        if isclose(pivot, 0, abs_tol=1e-08):
            return 0.
        val *= pivot
    
    return val

def inv(mat):
    mat = np.array(mat, dtype=float)
    r, c = mat.shape
    if r!=c:
        raise ValueError('matrix must be square')
    inv = np.eye(r)
    
    for i in range(r):
        pivot = mat[i, i]
        for next_row in range(i+1, r):
            below_pivot = mat[next_row, i]
            if not isclose(below_pivot, 0, abs_tol=1e-08):
                if not isclose(pivot, 0, abs_tol=1e-08):
                    inv[next_row] -= (below_pivot/pivot)*inv[i]
                    mat[next_row] -= (below_pivot/pivot)*mat[i]
                else:
                    inv[[i, next_row]] = inv[[next_row, i]]
                    mat[[i, next_row]] = mat[[next_row, i]]
                    pivot = below_pivot
        
        if isclose(pivot, 0, abs_tol=1e-08):
            return False
        inv[i] /= pivot
        mat[i] /= pivot
    
    for i in reversed(range(r)):
        for prev_row in reversed(range(i)):
            above_pivot = mat[prev_row, i]
            if not isclose(above_pivot, 0, abs_tol=1e-08):
                inv[prev_row] -= above_pivot*inv[i]
    
    return inv

def echelon_form(mat, pivots=False, reduced=False):
    mat = np.array(mat, dtype=float)
    r, c = mat.shape
    pivot_set = []
    shift = 0
    
    for i in range(min(r, c)):
        while i+shift<c and all(isclose(elem, 0, abs_tol=1e-08)
                                for elem in mat[i:, i+shift]): shift += 1
        if i+shift==c: break
        
        pivot = mat[i, i+shift]
        for next_row in range(i+1, r):
            below_pivot = mat[next_row, i+shift]
            if not isclose(below_pivot, 0, abs_tol=1e-08):
                if not isclose(pivot, 0, abs_tol=1e-08):
                    mat[next_row] -= (below_pivot/pivot)*mat[i]
                else:
                    mat[[i, next_row]] = mat[[next_row, i]]
                    pivot = below_pivot
        
        pivot_set.append(i+shift)
        mat[i, :i+shift] = 0.
    
    if reduced:
        for i, j in zip(reversed(range(len(pivot_set))), reversed(pivot_set)):
            mat[i] /= mat[i, j]
            for prev_row in reversed(range(i)):
                above_pivot = mat[prev_row, j]
                if not isclose(above_pivot, 0, abs_tol=1e-08):
                    mat[prev_row] -= above_pivot*mat[i]
                else: mat[prev_row, j] = 0.
    
    if pivots:
        return mat, pivot_set
    return mat

def rank(mat):
    mat, pivots = echelon_form(mat, pivots=True)
    
    return len(pivots)

def solve(mat):
    mat, pivots = echelon_form(mat, pivots=True)
    pivots = set(pivots)
    r, c = mat.shape
    
    if c-1 in pivots:
        return False
    
    A, b = mat[len(pivots)-1::-1, -2::-1], mat[len(pivots)-1::-1, -1]
    x = []
    
    for eq, val in zip(A, b):
        sm = 0
        for i, coeff in enumerate(eq):
            if i==len(x):
                if c-2-i in pivots:
                    x.append((val-sm)/coeff)
                    break
                x.append(Symbol(f'x{c-1-i}'))
            sm += coeff*x[i]
    
    for i in range(c-1-len(x), 0, -1):
        x.append(Symbol(f'x{i}'))
    
    return x[::-1]

def lin_comb(vec, bases):
    bases = [np.ravel(base) for base in bases]
    aug_mat = np.column_stack([*bases, vec])
    
    return solve(aug_mat)

def is_lin_indep(*vecs):
    vecs = [np.ravel(vec) for vec in vecs]
    aug_mat = np.column_stack([*vecs, [0]*len(vecs[0])])
    soln = solve(aug_mat)
    
    if all(val==0 for val in soln):
        return True
    return False


if __name__=='__main__':
    from matrices import input_mat
    A = input_mat(square=True)
    B = input_mat()
    print('Matrix A:', A)
    print('Matrix B:', B)
    
    print('Determinant A:', det(A))
    print('Inverse A:\n', inv(A))
    print('Echelon form B:\n', echelon_form(B, reduced=True))
    print('Rank B:', rank(B))
    print('Solution B:', solve(B))
