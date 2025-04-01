import numpy as np
import matplotlib.pyplot as plt

EPS = 0.000001
DEVISIONS_COUNT = 1000
U_DEVISION_COUNT = 50
MAX_ITER = 100

def derrivative(x, u):
    return np.cos(x)

def f(x, u):
    return np.sin(x) + u

def approximate_func(roots, f, u):
    for i in range(len(roots) - 1):
        first = roots[i]
        second = roots[i + 1]
        if f(first, u) * f(second, u) > 0:
            continue
        while (abs(f(first, u) - f(second, u)))>EPS:
            mid = (first + second) / 2                   
            if f(mid, u) == 0 or abs(f(mid, u) - f(first, u))<EPS: 
                yield mid
                break
            elif (f(first, u) * f(mid, u)) < 0:
                second = mid
            else:
                first = mid
def approximate_with_derrivative(roots, f, u, a, b):
    for i in range(len(roots) - 1):
        first = roots[i]
        second = first - f(first, u) / derrivative(first, u)
        itercount = 1
        while True:
            if (abs(second - first)) < EPS:
                yield second
                break
            elif second < a or second > b:
                break
            elif itercount > MAX_ITER:
                break
            first = second
            second = first - f(first, u) / derrivative(first, u)
            itercount += 1
        

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
    roots = np.linspace(a, b, DEVISIONS_COUNT)
    us = np.linspace(A, B, U_DEVISION_COUNT)
    res_us = []
    res_roots = []
    for u in us:
        for root in approximate_with_derrivative(roots, f, u, a, b):
            contains_root = any(abs(old_root - root) < EPS for old_root in res_roots)
            if contains_root:
                continue
            # elif root > a and root < b:
            res_us.append(u)
            res_roots.append(root)
    return res_us, res_roots


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
    try:
        a, b, A, B = map(float, input("Введите a, b, A, B: ").split())
        us, roots = method(a, b, A, B, f) 
        fig, ax = plt.subplots()
        ax.scatter(us, roots)

        ax.set(xlabel='u', ylabel='x')
        ax.grid()

        plt.show()
    except ValueError:
        print("Некорректный ввод")

start()
# print(list(approximate_func(np.linspace(-3, 3, 1000), f, 0.5)))
# test = []
# for root in approximate_with_derrivative(np.linspace(-3, 3, 1000), f, 0.5):
#     contains_root = any(abs(old_root - root) < EPS for old_root in test)
#     if contains_root:
#         continue
#     elif root > -3 and root < 3:
#         test.append(root)
# print(test)