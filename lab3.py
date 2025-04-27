import numpy as np
import matplotlib.pyplot as plt


PHI = (1 + np.sqrt(5)) / 2
MAX_ITER = 100

def f(x):
    return (x[0] - 2)**2 + (x[1])**2

def one_dim_descent(*, f, x, i, a0, b0, epsilon):
    a = a0
    b = b0
    while (b - a) >= epsilon:
        A = b - ((b - a) / PHI)
        B = a + ((b - a) / PHI)
        A_vector = np.copy(x)
        B_vector = np.copy(x)

        A_vector[i] = A
        B_vector[i] = B

        if f(A_vector) <= f(B_vector):
            # a = a 
            b = B
            xi = A
        else:
            a = A
            # b = b
            xi = B
    return xi 
    



def coord_descent(epsilon, x0, y0):
    print("coord_descent")

def grad_descent(epsilon, x0, y0):
    print("grad_descent")

def start():
    print("Выберете метод:")
    print("1. Метод покоординатного спуска")
    print("2. Метод наискорейшего градиентного спуска")
    choice = input().strip()

    match choice:
        case "1":
            method = coord_descent 
        case "2":
            method = grad_descent 
        case _:
            print("Такого варианта нет")

    try:
        epsilon, x0, y0 = map(float, input("Введите epsilon, x0, y0: ").split())
        
    except ValueError:
        print("Некорректный ввод")

