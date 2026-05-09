def input_vec(dtype=float):
    dim = int(input('Enter dimension of vector: '))
    return [dtype(input(f'Enter element {i+1}: ')) for i in range(dim)]

def scale(scalar, vec):
    return [scalar*elem for elem in vec]

def add(*vecs):
    return [sum(elems) for elems in zip(*vecs, strict=True)]

def sub(vec1, vec2):
    return [elem1-elem2 for elem1, elem2 in zip(vec1, vec2, strict=True)]

def dot(vec1, vec2):
    return sum(elem1*elem2 for elem1, elem2 in zip(vec1, vec2, strict=True))


if __name__=='__main__':
    u = input_vec()
    v = input_vec()
    print('Vector u:', u)
    print('Vector v:', v)
    
    print('u+v:', add(u, v))
    print('u-v:', sub(u, v))
    print('7u:', scale(7, u))
    print('u.v:', dot(u, v))
