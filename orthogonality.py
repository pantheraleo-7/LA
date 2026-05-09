from vectors import dot as inner_product, scale, sub
from gaussianelimination import is_lin_indep
from math import isclose, sqrt, acos, degrees

def are_orthogonal(vec1, vec2):
    return isclose(inner_product(vec1, vec2), 0, abs_tol=1e-08)

def norm(vec):
    return sqrt(inner_product(vec, vec))

def normalise(vec):
    return scale(1/norm(vec), vec)

def are_orthonormal(vec1, vec2):
    return are_orthogonal(vec1, vec2)\
            and isclose(norm(vec1), 1, abs_tol=1e-08)\
            and isclose(norm(vec2), 1, abs_tol=1e-08)

def angle(vec1, vec2):
    theta_radians = acos(inner_product(vec1, vec2)/(norm(vec1)*norm(vec2)))
    return degrees(theta_radians)

def is_orthogonal_set(*vecs):
    return all(are_orthogonal(vec, next) for i, vec in enumerate(vecs) for next in vecs[i+1:])

def gram_schmidt(*vecs):
    if not is_lin_indep(vecs):
        raise ValueError('vectors must be linearly independent')
    
    ortho_set = []
    for vec in vecs:
        res = vec
        for ortho_vec in ortho_set:
            proj = scale(inner_product(vec, ortho_vec)/norm(ortho_vec)**2, ortho_vec)
            res = sub(res, proj)
        ortho_set.append([round(i, 8) for i in res])
    
    return ortho_set


if __name__=='__main__':
    from vectors import input_vec
    u = input_vec()
    v = input_vec()
    w = input_vec()
    print('Vector u:', u)
    print('Vector v:', v)
    print('Vector w:', w)

    print('Inner product <u, v>:', inner_product(u, v))
    print('Are u & v orthogonal?:', are_orthogonal(u, v))
    print('Norm ||w||:', norm(w))
    print('Are v & w orthonormal?:', are_orthonormal(v, w))
    print('Angle between u & w:', angle(u, w))
    print('Is set (u, v, w) orthogonal?:', is_orthogonal_set(u, v, w))
    print('Orthogonalised set from set (u, v, w):', gram_schmidt(u, v, w))
