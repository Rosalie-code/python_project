from example import sum_of_squares as sum_of_squares_cython


def sum_of_squares(n):
    total = 0
    for i in range(1, n + 1):
        total += i * i
    return total


print(sum_of_squares_cython(1000))