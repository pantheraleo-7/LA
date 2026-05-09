import numpy as np
from sympy import Symbol, eye, det, roots, Matrix, Poly
from gaussianelimination import solve

def get_A_LI(mat, var='lamda'):
    r, c = mat.shape
    L = Symbol(var)
    I = eye(r)
    
    return mat - L*I

def charac_poly(mat, var='lamda'):
    A_LI = get_A_LI(mat, var=var)
    
    return det(A_LI).as_poly().monic()

def eig_vals(mat):
    charac = charac_poly(mat)
    
    return roots(charac)

def eigs(mat):
    A_LI = get_A_LI(mat)
    charac = det(A_LI)
    vals = roots(charac)
    vecs = [A_LI.subs('lamda', val).nullspace() for val in vals]
    vals = {val: (alg_mcity, len(span_set))
            for (val, alg_mcity), span_set in zip(vals.items(), vecs)}
    
    return vals, vecs

def eig_vecs(mat):
    return eigs(mat)[1]

def is_diagonalisable(mat):
    A_LI = get_A_LI(mat)
    charac = det(A_LI)
    vals = roots(charac)
    
    if all(m==1 for m in vals.values()):
        return True
    
    return all(vals[val]==len(A_LI.subs('lamda', val).nullspace())
               for val in vals if vals[val]>1)

def diagonalise(mat):
    vals, vecs = eigs(mat)
    val_list = [val for val in vals for _ in range(vals[val][0])]
    vec_list = [vec for span_set in vecs for vec in span_set]
    
    if len(vec_list)<len(val_list):
        return False
    return np.diag(val_list), np.column_stack(vec_list)

def minimal_poly(mat, var='x'):
    r, c = mat.shape
    M = mat
    v = np.random.randn(c)
    
    res = np.column_stack((v, M@v, np.zeros(c)))
    for _ in range(r-1):
        soln = solve(res)
        if not all(val==0 for val in soln):
            coeffs = [val.subs(*val.free_symbols, 1).round(8)
                      for val in reversed(soln)]
            return Poly(coeffs, Symbol(var))
        
        M = M@M
        res = np.insert(res, -1, M@v, axis=1)
    
    return charac_poly(mat, var=var)

def is_derogatory(mat):
    r, c = mat.shape
    vals = eig_vals(mat)
    
    if all(m==1 for m in vals.values()):
        return False
    return minimal_poly(mat).degree()<r

def is_positive_definite(mat):
    if mat!=mat.T:
        return False
    
    vals = eig_vals(mat)
    return all(val>0 for val in vals)

def is_negative_definite(mat):
    if mat!=mat.T:
        return False
    
    vals = eig_vals(mat)
    return all(val<0 for val in vals)


if __name__=='__main__':
    from matrices import input_mat
    from sympy import S
    
    user_inp = input_mat(square=True, dtype=S)
    A = Matrix(user_inp)
    print(A)
    
    vals, vecs = eigs(A)
    print('Eigen values:', vals)
    print('Eigen vectors:')
    for span_set in vecs:
        for vec in span_set:
            print(vec.T, end=' ')
        print()
    
    print('Minimal polynomial:', minimal_poly(A))
    print('Is derogatory?:', is_derogatory(A))
    print('Is diagonalisable?:', is_diagonalisable(A))
    D, M = diagonalise(A)
    print('Diagonal matrix:\n', D)
    print('Modal matrix:\n', M)
