import numpy as np


def Lagrange(x, points):
    n = len(points)
    result = 0
    for point in points:
        f = point[1]
        for i in range(n):
            for j in range(n):
                xi = points[i][0]
                xj = points[j][0]
                if j != i:
                    f *= (x - xj)/(xi - xj)
        result += f
    return result

def integral_calc(a, b, t, sum_func, func, N, eps):
    prev_sum = np.inf
    while True:
        space = np.linspace(a, b, N)
        int_sum = 0
        for i in range(N - 1):
            int_sum += sum_func(space[i], space[i + 1], t, func)
        
        if abs(int_sum - prev_sum) < eps:
            return int_sum 
        N *= 2
        prev_sum = int_sum

def left_rectangle(a, b, t, func):
    return func(a, t) * (b - a)

def right_rectangle(a, b, t, func):
    return func(b, t) * (b - a)

def middle_rectangle(a, b, t, func):
    return func(b, t) * (b - a)

def trapezoid(a, b, t, func):
    return ((func(a, t) + func(b, t)) / 2) * (b - a)

def simpson(a, b, t, func):
    return ((func(a, t) + 4 * func((a + b, t) / 2) + func(b, t)) / 6) * (b - a)


def f(x, t):
    return np.sin(x * t)

print(integral_calc(0, np.pi, 1, middle_rectangle, f, 100, 0.1))

def main():
    print("1. -//-")
    print("2. -//-")
    print("3. -//-")
    print("4. -//-")
    print("5. -//-")
    calc_method = int(print("Выберите метод подсчета: "))
    eps = float(input("Введите эпсилон: "))

    match calc_method:
        case 1:
            pass
        case 2:
            pass
        case 3:
            pass
        case 4:
            pass
        case 5:
            pass

main()