import numpy as np
import matplotlib.pyplot as plt

def coord_descent(epsilon, x0, y0):
    pass

def grad_descent(epsilon, x0, y0):
    pass

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