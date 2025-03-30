import numpy as np
import matplotlib.pyplot as plt
import math

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

def integral_calc(*, a, b, t, sum_func, func, N, eps, s):
    prev_sum = np.inf
    while True:
        space = np.linspace(a, b, N)
        int_sum = 0
        for i in range(N - 1):
            int_sum += sum_func(space[i], space[i + 1], t, func)
        
        if abs(int_sum - prev_sum) / (math.pow(2, s) - 1) < eps:
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
    return ((func(a, t) + 4 * func((a + b) / 2, t) + func(b, t)) / 6) * (b - a)


def f(x, t):
    return np.sin(x * t)


def main():
    print("1. Формула правых треугольников")
    print("2. Формула левых треугольников")
    print("3. Формула средних треугольников")
    print("4. Формула трапеции")
    print("5. Формула Симпсона")
    calc_method = input("Выберите метод подсчета: ")
    eps = float(input("Введите эпсилон: "))
    alpha, beta = map(float, input("Введите альфа и бета:").split(" "))

    trange = np.arange(alpha, beta, 0.01)

    match calc_method:
        case "1":
            s = 1
            sum_func = left_rectangle
        case "2":
            s = 1
            sum_func = right_rectangle
        case "3":
            s = 2
            sum_func = middle_rectangle 
        case "4":
            s = 2
            sum_func = trapezoid 
        case "5":
            s = 4 
            sum_func = simpson
        case _:
            print("Нет такого варианта)")
            return
    
    I = []

    for t in trange:
        I.append(integral_calc(a=0, b=np.pi, t=t, sum_func=sum_func, func=f, N=10, eps=eps, s=s))

    fig, ax = plt.subplots()
    ax.plot(trange, I)

    ax.set(xlabel='t', ylabel='I(t)')
    ax.grid()

    plt.show()
main()
# print(integral_calc(a=0, b=np.pi, t=np.pi/2, sum_func=left_rectangle, func=f, N=100, eps=0.1, s=1))