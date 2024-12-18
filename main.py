import argparse

import numpy as np
from matplotlib import pyplot as plt

g = 9.81


def find_distance(v, alpha, h):
    b = (v ** 2) * np.sin(2 * np.deg2rad(alpha))
    D = (v ** 4) * np.sin(2 * np.deg2rad(alpha)) ** 2 + 8 * g * h * (v ** 2) * np.cos(np.deg2rad(alpha)) ** 2
    distance = (b + np.sqrt(D)) / (2 * g)

    return distance


def find_height(t, v, alpha, h):
    vertical_velocity = v * np.sin(np.deg2rad((alpha)))
    height = abs(h + vertical_velocity * t - 0.5 * g * (t ** 2))

    return height


def main(args):
    lever_mass = args.lever_mass  # масса рычага
    missile_arm_mass = lever_mass / 3 * 2  # масса снарядного плеча
    cargo_arm_mass = lever_mass / 3  # масса грузового плеча
    missile_mass = args.missile_mass  # масса снаряда
    cargo_mass = args.cargo_mass  # масса груза
    missile_arm_length = args.missile_arm_length  # длина снарядного плеча
    cargo_arm_length = args.cargo_arm_length  # длина грузового плеча
    alpha = args.alpha  # угол запуска
    initial_height = args.initial_height  # стартовая высота
    d = args.d

    Mk = (cargo_arm_mass * (cargo_arm_length / 2) * g) + (cargo_mass * cargo_arm_length * g)  # Момент силы для грузового плеча
    Ml = (missile_arm_mass * (missile_arm_length / 2) * g) + (missile_mass * missile_arm_length * g)  # Момент силы для снарядного плеча

    Ik = missile_mass * (missile_arm_length ** 3) * g * np.cos(np.deg2rad(alpha))  # Момент инерции для грузового плеча
    Il = cargo_mass * (cargo_arm_length ** 3) * g * np.cos(np.deg2rad(alpha))  # Момент инерции для снарядного плеча
    I = (missile_arm_mass + cargo_arm_mass) * (d ** 2) + ((missile_arm_mass + cargo_arm_mass) * (missile_arm_length + cargo_arm_length) ** 2) / 12  # Стержневая инерция

    w = 2 * missile_arm_length * np.sqrt((Mk - Ml) / (Ik + Il + I) * np.sin(np.deg2rad(alpha)))  # Угловая скорость
    distance = find_distance(w, alpha, initial_height)  # Итоговое расстояние полёта снаряда
    print(f'Расстояние: {distance:.3f} м')

    flightTime = distance / (w * np.cos(np.deg2rad(alpha)))  # Время полёта снаряда
    print(f'Время полёта: {flightTime:.3f} c')

    prec = 0.0001  # Точность отображения графика
    times = [i * prec for i in range(int(flightTime / prec) + 1)]
    heights = [find_height(t, w, alpha, initial_height) for t in times]
    distances = [i * (distance / len(times)) for i in range(len(times))]

    plt.figure(figsize=(10, 6))
    plt.plot(distances, heights, label='Траектория снаряда')
    plt.scatter([distance], 0, color='blue', label=f'Конечная точка (x={distance:.3f})')
    plt.title('График полёта снаряда')
    plt.xlabel('Расстояние (м)')
    plt.ylabel('Высота (м)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Моделирование полёта снаряда с использованием рычага.")
    parser.add_argument('--lever_mass', type=float, default=0.143, help="Масса рычага (кг)")
    parser.add_argument('--missile_mass', type=float, default=0.011, help="Масса снаряда (кг)")
    parser.add_argument('--cargo_mass', type=float, default=0.177, help="Масса груза (кг)")
    parser.add_argument('--missile_arm_length', type=float, default=0.2, help="Длина снарядного плеча (м)")
    parser.add_argument('--cargo_arm_length', type=float, default=0.1, help="Длина грузового плеча (м)")
    parser.add_argument('--alpha', type=float, default=45, help="Угол запуска (градусы)")
    parser.add_argument('--initial_height', type=float, default=0.2, help="Начальная высота (м)")
    parser.add_argument('--d', type=float, default=0.05, help="Параметр 'd' для расчёта стержневой инерции")

    args = parser.parse_args()
    main(args)