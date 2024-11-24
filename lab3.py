import numpy as np
import matplotlib.pyplot as plt

# Експериментальні дані
x = np.array([0.4, 0.5, 0.8, 1, 1.5, 2, 2.5, 3])
y = np.array([1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4])

# Логарифмуємо дані для приведення рівняння y = ax^b

log_x = np.log(x)
log_y = np.log(y)

# Кількість точок кількість точок у наших даних (тут n=8).
n = len(x)

# Обчислюємо суми, потрібні для формул МНК
sum_log_x = np.sum(log_x)
sum_log_y = np.sum(log_y)
sum_log_x2 = np.sum(log_x**2)
sum_log_xy = np.sum(log_x * log_y)


# Обчислення коефіцієнтів
b = (n * sum_log_xy - sum_log_x * sum_log_y) / (n * sum_log_x2 - sum_log_x**2)
log_a = (sum_log_y - b * sum_log_x) / n
a = np.exp(log_a)

# Побудова апроксимації
y_pred = a * x**b

# Графік
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Експериментальні точки')
plt.plot(x, y_pred, color='red', label=f'Рівняння регресії: y = {a:.3f}x^{b:.3f}')
plt.title("Лінійна регресія методом найменших квадратів")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Вивід результатів
print(f"Коефіцієнт a: {a:.3f}")
print(f"Коефіцієнт b: {b:.3f}")
