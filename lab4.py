import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def gaussian_elimination(A, b):
    n = len(b)
    aug_matrix = np.column_stack((A, b))
    
    for i in range(n):
        max_row = i + np.argmax(np.abs(aug_matrix[i:n, i]))
        if max_row != i:
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
        
        for j in range(i + 1, n):
            factor = aug_matrix[j, i] / aug_matrix[i, i]
            aug_matrix[j, i:] -= factor * aug_matrix[i, i:]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug_matrix[i, -1] - np.sum(aug_matrix[i, i+1:n] * x[i+1:])) / aug_matrix[i, i]
    
    return x



def p(t):
    return 2*t

def q(t):
    return t**2 + 1

def f(t):
    return np.sin(np.pi*t)

def phi(t, i):
    return t**i * (1-t)

def derivative_first(func, t, h=0.001):
    return (func(t + h) - func(t - h)) / (2 * h)

def derivative_second(func, t, h=0.001):
    return (func(t - h) - 2*func(t) + func(t + h)) / h**2

def L_operator(func, t, h=0.001):
    d2 = derivative_second(func, t, h)
    d1 = derivative_first(func, t, h)
    return d2 + p(t) * d1 + q(t) * func(t)

def inner_product(func1, func2, a=0, b=1, n=1000):
    t = np.linspace(a, b, n)
    y = np.array([func1(ti) * func2(ti) for ti in t])
    return simpson(y, t)


def solve_galerkin(N):
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            A[i, j] = inner_product(
                        lambda t: phi(t, j+1), 
                        lambda t: L_operator(lambda s: phi(s, i+1), t)
                    ) 
        b[i] = inner_product(lambda t: phi(t, i+1), f)
    print(A)
    print(b)
    C = gaussian_elimination(A, b)
    # C = np.linalg.solve(A, b) 
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