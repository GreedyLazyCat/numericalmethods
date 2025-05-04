import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def gauss_elimination(A, b):
    """
    Решение системы линейных уравнений A*x = b методом Гаусса с выбором главного элемента
    
    Args:
        A: матрица коэффициентов (numpy array размера n x n)
        b: вектор правой части (numpy array размера n)
        
    Returns:
        x: вектор решения (numpy array размера n)
    """
    n = len(b)
    # Создаем расширенную матрицу
    Ab = np.column_stack((A.copy(), b.copy()))
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Выбор главного элемента по столбцу
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Проверка на вырожденность с учетом погрешности вычислений
        if np.abs(Ab[i, i]) < 1e-8:
            # Попытка найти ненулевой элемент в строке
            for k in range(i+1, n):
                if np.abs(Ab[k, i]) > 1e-8:
                    Ab[[i, k]] = Ab[[k, i]]
                    break
            else:
                raise ValueError("Матрица близка к вырожденной, невозможно решить систему")
        
        # Нормализация строки для улучшения численной стабильности
        pivot = Ab[i, i]
        Ab[i, i:] = Ab[i, i:] / pivot
        
        # Исключение переменной из остальных строк
        for j in range(i + 1, n):
            factor = Ab[j, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Обратный ход метода Гаусса
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])
    
    return x

def p(t):
    """Коэффициент при первой производной"""
    return 2*t

def q(t):
    """Коэффициент при функции"""
    return t**2 + 1

def f(t):
    """Правая часть уравнения"""
    return np.sin(np.pi*t)

def phi(t, i):
    """Базисная функция phi_i(t)
    
    Удовлетворяет краевым условиям: phi_i(0) = 0, phi_i(1) = 0
    phi_i(t) = t^i * (1-t)
    """
    return t**i * (1-t)

def derivative_first(func, t, h=0.001):
    """Аппроксимация первой производной"""
    return (func(t + h) - func(t - h)) / (2 * h)

def derivative_second(func, t, h=0.001):
    """Аппроксимация второй производной"""
    return (func(t - h) - 2*func(t) + func(t + h)) / h**2

def L_operator(func, t, h=0.001):
    """Оператор L(x) = x'' + p(t)x' + q(t)x"""
    d2 = derivative_second(func, t, h)
    d1 = derivative_first(func, t, h)
    return d2 + p(t) * d1 + q(t) * func(t)

def inner_product(func1, func2, a=0, b=1, n=1000):
    """Скалярное произведение <func1, func2> = ∫func1(t)·func2(t)dt от a до b
    
    Используется метод Симпсона для интегрирования с хорошей точностью
    """
    t = np.linspace(a, b, n)
    y = np.array([func1(ti) * func2(ti) for ti in t])
    return simpson(y, t)

def solve_linear_system(A, b):
    """
    Решение системы линейных уравнений A*x = b с использованием SVD
    
    Args:
        A: матрица коэффициентов (numpy array размера n x n)
        b: вектор правой части (numpy array размера n)
        
    Returns:
        x: вектор решения (numpy array размера n)
    """
    # Выполняем SVD разложение
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Проверяем сингулярные значения на близость к нулю
    tol = 1e-10
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    
    # Вычисляем решение
    x = Vh.T @ (s_inv * (U.T @ b))
    
    return x

def solve_galerkin(N):
    """Решение краевой задачи методом Галеркина с N базисными функциями"""
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            A[i, j] = inner_product(lambda t: phi(t, j+1), 
                                  lambda t: L_operator(lambda t: phi(t, i+1), t))
        b[i] = inner_product(lambda t: phi(t, i+1), f)
    
    # Используем SVD вместо метода Гаусса
    C = solve_linear_system(A, b)
    
    def x_N(t):
        result = 0
        for i in range(N):
            result += C[i] * phi(t, i+1)
        return result
    
    return x_N, C

def main():
    try:
        N = int(input("Введите количество базисных функций N: "))
        if N <= 0:
            print("Ошибка: N должно быть положительным числом")
            return
    except ValueError:
        print("Ошибка: Введите целое число")
        return
    
    x_N, coefficients = solve_galerkin(N)
    
    print("\nНайденные коэффициенты C_i:")
    for i, c in enumerate(coefficients):
        print(f"C_{i+1} = {c:.6f}")
    
    t = np.linspace(0, 1, 1000)
    y = np.array([x_N(ti) for ti in t])
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'b-', linewidth=2)
    plt.title(f'Приближенное решение краевой задачи (N={N})')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    print(f"\nПроверка краевых условий:")
    print(f"x(0) = {x_N(0):.6f}")
    print(f"x(1) = {x_N(1):.6f}")
    
    plt.show()

if __name__ == "__main__":
    main()