import numpy as np
import matplotlib.pyplot as plt
import traceback

PHI = (1 + np.sqrt(5)) / 2
MAX_ITER = 1000
RECT = [
    [-5.0, 5.0],
    [-5.0, 5.0]
]

def derivative_x(x):
    return np.cos(x[0]) * np.cos(x[1] / 2)

def derivative_y(y):
    return -1 * np.sin(y[0]) * np.sin(y[1] / 2) / 2

def f(x):
    return np.sin(x[0]) * np.cos(x[1] / 2)

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
    xk = np.array([x0, y0])
    iter = 0
    while True:
        x_prev = np.copy(xk)
        yield x_prev
        
        for i in range(len(xk)):
            xk[i] = one_dim_descent(
                f=f,
                x=xk,
                i=i,
                a0=RECT[i][0],
                b0=RECT[i][1],
                epsilon=epsilon
            )
            if np.linalg.norm(xk - x_prev) < epsilon:
                break
            
        if iter >= MAX_ITER:
            print("Превышено максимальное число итераций")
            break
        iter += 1
    

def grad_descent(epsilon, x0, y0):
    xk = np.array([x0, y0], dtype=float)
    iter = 0
    while True:
        x_prev = np.copy(xk)
        yield x_prev

        grad = np.array([
            derivative_x(xk),
            derivative_y(xk)
        ])

        def phi(alpha):
            return f(xk - alpha * grad)

        alpha_opt = one_dim_descent(f=phi, x=[0], i=0, a0=0, b0=10, epsilon=epsilon)
        xk = xk - alpha_opt * grad

        if np.linalg.norm(xk - x_prev) < epsilon:
            break

        if iter > MAX_ITER:
            print("Превышено максимальное число итераций")
            break
        iter += 1



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
            return

    try:
        epsilon, x0, y0 = map(float, input("Введите epsilon, x0, y0: ").split())
        if x0 < RECT[0][0] or x0 > RECT[0][1]:
            print(f"x должен быть в пределе от {RECT[0][0]} до {RECT[0][1]} ")
            return
        
        if y0 < RECT[1][0] or y0 > RECT[1][1]:
            print(f"y должен быть в пределе от {RECT[1][0]} до {RECT[1][1]} ")
            return

        points = list(method(epsilon, x0, y0))
        print(points)
        x_points = []        
        y_points = []
        for point in points:
            x_points.append(point[0])        
            y_points.append(point[1])
        
        extent = 5
        x = np.linspace(RECT[0][0] - extent, RECT[0][1] + extent, 400)
        y = np.linspace(RECT[1][0] - extent, RECT[1][1] + extent, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y / 2)

        plt.contour(X, Y, Z, levels=30, cmap='jet')
        plt.plot(x_points, y_points, marker='o', color='red')  
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Линия уровня')
        plt.grid(True)
        plt.show()        

    except ValueError:
        print(traceback.format_exc())
        print("Некорректный ввод")

start()