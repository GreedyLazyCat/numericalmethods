import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

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
    # Проверка для граничных точек
    if t - h < 0:
        return (func(t + h) - func(t)) / h
    elif t + h > 1:
        return (func(t) - func(t - h)) / h
    else:
        return (func(t + h) - func(t - h)) / (2 * h)

def derivative_second(func, t, h=0.001):
    """Аппроксимация второй производной"""
    # Проверка для граничных точек
    if t - h < 0:
        return (func(t) - 2*func(t + h/2) + func(t + h)) / (h**2/4)
    elif t + h > 1:
        return (func(t - h) - 2*func(t - h/2) + func(t)) / (h**2/4)
    else:
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

def solve_galerkin(N):
    """Решение краевой задачи методом Галеркина с N базисными функциями"""
    
    # Строим матрицу A и вектор b
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            # Определяем функцию для интегрирования phi_j(t) * L(phi_i)(t)
            def integrand(t):
                return phi(t, j+1) * L_operator(lambda t: phi(t, i+1), t)
            
            # Вычисляем элемент матрицы A
            A[i, j] = inner_product(lambda t: phi(t, j+1), 
                                   lambda t: L_operator(lambda t: phi(t, i+1), t))
        
        # Определяем функцию для интегрирования phi_i(t) * f(t)
        def integrand_b(t):
            return phi(t, i+1) * f(t)
        
        # Вычисляем элемент вектора b
        b[i] = inner_product(lambda t: phi(t, i+1), f)
    
    # Решаем систему линейных уравнений A*C = b
    C = np.linalg.solve(A, b)
    
    # Функция приближенного решения
    def x_N(t):
        result = 0
        for i in range(N):
            result += C[i] * phi(t, i+1)
        return result
    
    return x_N, C

def main():
    # Ввод количества базисных функций
    try:
        N = int(input("Введите количество базисных функций N: "))
        if N <= 0:
            print("Ошибка: N должно быть положительным числом")
            return
    except ValueError:
        print("Ошибка: Введите целое число")
        return
    
    # Решаем задачу методом Галеркина
    x_N, coefficients = solve_galerkin(N)
    
    # Выводим найденные коэффициенты
    print("\nНайденные коэффициенты C_i:")
    for i, c in enumerate(coefficients):
        print(f"C_{i+1} = {c:.6f}")
    
    # Построение графика решения
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
    
    # Проверка краевых условий
    print(f"\nПроверка краевых условий:")
    print(f"x(0) = {x_N(0):.6f}")
    print(f"x(1) = {x_N(1):.6f}")
    
    plt.show()

if __name__ == "__main__":
    main()