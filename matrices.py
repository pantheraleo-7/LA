def input_mat(square=False, dtype=float):
    if square:
        r = c = int(input('Enter the dimension of the square matrix: '))
    else:
        r = int(input('Enter the number of rows: '))
        c = int(input('Enter the number of columns: '))
    
    ret_mat = []
    for i in range(r):
        ret_row = [dtype(input(f'Enter element for position [{i+1}, {j+1}]: ')) for j in range(c)]
        ret_mat.append(ret_row)
    
    return ret_mat

def transpose(mat):
    r, c = len(mat), len(mat[0])
    
    ret_mat = []
    for j in range(c):
        ret_row = [row[j] for row in mat]
        ret_mat.append(ret_row)
    
    return ret_mat

def scale(scalar, mat):
    ret_mat = []
    for row in mat:
        ret_row = [scalar*elem for elem in row]
        ret_mat.append(ret_row)
    
    return ret_mat

def add(*mats):
    ret_mat = []
    for rows in zip(*mats, strict=True):
        ret_row = [sum(elems) for elems in zip(*rows, strict=True)]
        ret_mat.append(ret_row)
    
    return ret_mat

def sub(mat1, mat2):
    ret_mat = []
    for row1, row2 in zip(mat1, mat2, strict=True):
        ret_row = [elem1-elem2 for elem1, elem2 in zip(row1, row2, strict=True)]
        ret_mat.append(ret_row)
    
    return ret_mat

def mul(mat1, mat2):
    mat2_T = transpose(mat2)
    ret_mat = []
    for row in mat1:
        ret_row = []
        for col in mat2_T:
            ret_row.append(sum(elem1*elem2 for elem1, elem2 in zip(row, col, strict=True)))
        ret_mat.append(ret_row)
    
    return ret_mat

def det(mat):
    r, c = len(mat), len(mat[0])
    
    if r!=c:
        raise ValueError('matrix must be square')
    if r==1==c:
        return mat[0][0]
    if r==2==c:
        return (mat[0][0]*mat[1][1]) - (mat[0][1]*mat[1][0])
    
    row1, rest_rows_T = mat[0], transpose(mat[1:])
    
    val = 0
    for i, elem in enumerate(row1):
        rem_row_col = transpose([rest_rows_T[k] for k in range(r) if k!=i])
        val += (-1)**i * elem * det(rem_row_col)
    
    return val

def minors(mat):
    r, c = len(mat), len(mat[0])
    mat_T = transpose(mat)
    
    ret_mat = []
    for i in range(r):
        ret_mat.append([])
        for j in range(c):
            rem_col_T = [mat_T[k] for k in range(c) if k!=j]
            rem_col = transpose(rem_col_T)
            rem_col_row = [rem_col[k] for k in range(r) if k!=i]
            
            ret_mat[-1].append(det(rem_col_row))
    
    return ret_mat

def cofac(mat):
    r, c = len(mat), len(mat[0])
    mat_T = transpose(mat)
    
    ret_mat = []
    for i in range(r):
        ret_mat.append([])
        for j in range(c):
            rem_col_T = [mat_T[k] for k in range(c) if k!=j]
            rem_col = transpose(rem_col_T)
            rem_col_row = [rem_col[k] for k in range(r) if k!=i]
            
            ret_mat[-1].append((-1)**(i+j) * det(rem_col_row))
    
    return ret_mat

def adjoint(mat):
    return transpose(cofac(mat))

def inv(mat):
    det_val = det(mat)
    if det_val==0:
        return False
    
    return scale(1/det_val, adjoint(mat))

def cramers(A, b):
    det_A = det(A)
    if det_A==0:
        return
    
    x = []
    for i in range(len(A)):
        Ai = transpose(A)
        Ai[i] = b
        det_Ai = det(Ai)
        
        x.append(det_Ai/det_A)
    
    return x


if __name__=='__main__':
    from vectors import input_vec
    A = input_mat(square=True)
    B = input_vec()
    C = input_mat(square=True)
    print('Matrix A:', A)
    print('Vector B:', B)
    print('Matrix C:', C)
    
    print('Transpose C:', transpose(C))
    print('A+C:', add(A, C))
    print('A-C:', sub(A, C))
    print('5C:', scale(5, C))
    print('AC:', mul(A, C))
    print('Determinant C:', det(C))
    print('Inverse C:', inv(C))
    print('Solution Ax=B:', cramers(A, B))
