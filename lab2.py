import numpy as np
import matplotlib.pyplot as plt

EPS = 0.0001
DEVISIONS_COUNT = 100000
U_DEVISION_COUNT = 100

def f(x, u):
    return 3 * x**2 + u * x - 2

def approximate_func(roots, f, u):
    for i in range(len(roots) - 1):
        first = roots[i]
        second = roots[i + 1]
        if f(first, u) * f(second, u) > 0:
            continue
        while (abs(f(first, u) - f(second, u)))>EPS:
            mid = (first + second) / 2                   
            if f(mid, u) == 0 or f(mid, u)<EPS: 
                yield mid
                break
            elif (f(first, u) * f(mid, u)) < 0:
                second = mid
            else:
                first = mid

def devide_method(a, b, A, B, f):
    roots = np.linspace(a, b, DEVISIONS_COUNT)
    us = np.linspace(A, B, U_DEVISION_COUNT)
    res_us = []
    res_roots = []
    for u in us:
        for root in approximate_func(roots, f, u):
            res_us.append(u)
            res_roots.append(root)
    return res_us, res_roots


def newton_method(a, b, A, B, f):
    print(f"a: {a} b: {b} A: {A} B: {B}")


def start():
    print("Выберете метод:")
    print("1. Метод деление отрезка пополам")
    print("2. Метод Ньютона")
    choice = input().strip()

    match choice:
        case "1":
            method = devide_method
        case "2":
            method = newton_method
        case _:
            print("Такого варианта нет")
    # try:
    a, b, A, B = map(float, input("Введите a, b, A, B: ").split())
    us, roots = method(a, b, A, B, f) 
    fig, ax = plt.subplots()
    ax.scatter(us, roots)

    ax.set(xlabel='u', ylabel='x')
    ax.grid()

    plt.show()
    # except ValueError:
        # print("Некорректный ввод")

start()
# print(list(approximate_func(np.linspace(-2, 2, 100000), f, 1)))