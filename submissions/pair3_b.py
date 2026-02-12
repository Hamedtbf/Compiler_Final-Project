def fibonacci(k):
    x, y = 0, 1
    for _ in range(k):
        x, y = y, x + y
    return x

print(fibonacci(10))
