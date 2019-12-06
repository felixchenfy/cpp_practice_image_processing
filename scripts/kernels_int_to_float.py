
import numpy as np


def print_as_cpp_2d_vector(mat):
    r, c = len(mat), len(mat[0])
    print("{")
    for i in range(r):
        print("{", end="")
        for j in range(c):
            print(mat[i, j], end="")
            if j != c-1:
                print(", ", end="")
        print("},")
    print("}")
    print("")


''' ----------------------------- GAUSSION_5x5 ----------------------------- '''
GAUSSION_5x5 = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]

print("Gaussion 5x5 kernel:")
print_as_cpp_2d_vector(np.array(GAUSSION_5x5).astype(np.float32)/256.0)


''' ----------------------------- GAUSSION_3x3 ----------------------------- '''
GAUSSION_3x3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]

print("Gaussion 3x3 kernel:")
print_as_cpp_2d_vector(np.array(GAUSSION_3x3).astype(np.float32)/16.0)
