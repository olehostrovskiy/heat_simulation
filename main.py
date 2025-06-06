import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import json
from datetime import datetime
import glob


def load_config(config_file=None):
    """Завантаження конфігурації з JSON файлу"""
    if config_file is None:
        # Пошук JSON файлів у поточній директорії
        json_files = glob.glob("*.json")
        if not json_files:
            print("JSON файли не знайдено. Використовуємо параметри за замовчуванням.")
            return get_default_config()

        print("Знайдені JSON файли:")
        for i, file in enumerate(json_files):
            print(f"{i + 1}. {file}")

        try:
            choice = int(input("Оберіть номер файлу (або 0 для параметрів за замовчуванням): "))
            if choice == 0:
                return get_default_config()
            config_file = json_files[choice - 1]
        except (ValueError, IndexError):
            print("Некоректний вибір. Використовуємо параметри за замовчуванням.")
            return get_default_config()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Завантажено конфігурацію з файлу: {config_file}")
        return config
    except FileNotFoundError:
        print(f"Файл {config_file} не знайдено. Використовуємо параметри за замовчуванням.")
        return get_default_config()


def get_default_config():
    """Конфігурація за замовчуванням"""
    return {
        "material_name": "Алюміній",
        "L": 1.0,
        "Nx": 100,
        "Ny": 100,
        "alpha": 9.75e-5,
        "simulation_time": 300.0,
        "T_initial": 20.0,
        "heat_source_type": "point",  # "edge" або "point"
        "T_boundary_edge": 100.0,
        "hotspot_center": [0.4, 0.6],  # координати точкового джерела
        "hotspot_radius": 0.05,
        "hotspot_temperature": 100.0,
        "animation_frames": 20,
        "static_frames": 8
    }


def create_output_directory(material_name):
    """Створення директорії для результатів"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"results_{material_name}_{timestamp}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def apply_heat_source(T, config, Nx, Ny, dx, dy):
    """Застосування джерела тепла"""
    if config["heat_source_type"] == "edge":
        # Нагрівання лівого краю
        T[0, :] = config["T_boundary_edge"]
    elif config["heat_source_type"] == "point":
        # Точкове джерело тепла
        center_x, center_y = config["hotspot_center"]
        radius = config["hotspot_radius"]
        temp = config["hotspot_temperature"]

        # Перетворення координат в індекси сітки
        center_i = int(center_x * (Nx - 1))
        center_j = int(center_y * (Ny - 1))
        radius_i = int(radius / dx)
        radius_j = int(radius / dy)

        # Застосування кругового джерела
        for i in range(max(0, center_i - radius_i), min(Nx, center_i + radius_i + 1)):
            for j in range(max(0, center_j - radius_j), min(Ny, center_j + radius_j + 1)):
                dist = np.sqrt(((i - center_i) * dx) ** 2 + ((j - center_j) * dy) ** 2)
                if dist <= radius:
                    T[i, j] = temp


def main():
    # Завантаження конфігурації
    config = load_config()

    # Створення директорії для результатів
    output_dir = create_output_directory(config["material_name"])
    print(f"Результати будуть збережено в: {output_dir}/")

    # Збереження конфігурації в директорію результатів
    with open(f"{output_dir}/config_used.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Геометрія пластини
    L = config["L"]
    Nx = config["Nx"]
    Ny = config["Ny"]
    dx = L / (Nx - 1)
    dy = L / (Ny - 1)

    # Фізичні параметри
    alpha = config["alpha"]
    T0 = config["T_initial"]
    simulation_time = config["simulation_time"]

    # Параметри анімації
    animation_frames = config["animation_frames"]
    static_frames = config["static_frames"]

    # Часові параметри
    dt = 0.25 * min(dx, dy) ** 2 / alpha
    Nt = int(simulation_time / dt)

    # Кроки для збереження кадрів
    animation_step = max(1, Nt // animation_frames)
    static_step = max(1, Nt // static_frames)

    print(f"\n=== ПАРАМЕТРИ СИМУЛЯЦІЇ ===")
    print(f"Матеріал: {config['material_name']}")
    print(f"Коефіцієнт теплопровідності α = {alpha:.2e} м²/с")
    print(f"Розмір пластини: {L}×{L} м")
    print(f"Сітка: {Nx}×{Ny} точок")
    print(f"Джерело тепла: {config['heat_source_type']}")
    print(f"Час симуляції: {simulation_time} с")
    print(f"Кроків обчислення: {Nt}")
    print(f"Кадрів для анімації: {animation_frames}")
    print(f"Статичних кадрів: {static_frames}")

    # Початкові умови
    T = np.full((Nx, Ny), T0)

    # Зберігання даних
    animation_data = []
    animation_times = []
    static_data = []
    static_times = []

    # Симуляція
    print(f"\nЗапуск симуляції...")
    for n in range(Nt):
        Tn = T.copy()

        # Оновлення внутрішніх точок
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                T[i, j] = Tn[i, j] + alpha * dt * (
                        (Tn[i + 1, j] - 2 * Tn[i, j] + Tn[i - 1, j]) / dx ** 2 +
                        (Tn[i, j + 1] - 2 * Tn[i, j] + Tn[i, j - 1]) / dy ** 2
                )

        # Застосування джерела тепла
        apply_heat_source(T, config, Nx, Ny, dx, dy)

        # Граничні умови (ізольовані краї, крім джерела)
        if config["heat_source_type"] == "edge":
            T[-1, :] = T[-2, :]  # правий край
            T[:, 0] = T[:, 1]  # нижній край
            T[:, -1] = T[:, -2]  # верхній край

        # Збереження кадрів для анімації
        if n % animation_step == 0 and len(animation_data) < animation_frames:
            animation_data.append(T.copy())
            animation_times.append(n * dt)

        # Збереження статичних кадрів
        if n % static_step == 0 and len(static_data) < static_frames:
            static_data.append(T.copy())
            static_times.append(n * dt)

            # Збереження статичного графіка
            plt.figure(figsize=(8, 6))
            plt.imshow(T.T, cmap='hot', origin='lower', extent=[0, L, 0, L],
                       vmin=T0, vmax=max(config.get("T_boundary_edge", 100), config.get("hotspot_temperature", 100)))
            plt.colorbar(label='Температура (°C)')
            plt.title(f'{config["material_name"]} (α = {alpha:.2e} м²/с)\nЧас: {n * dt:.1f} с')
            plt.xlabel('x (м)')
            plt.ylabel('y (м)')
            plt.tight_layout()

            filename = f'{output_dir}/static_frame_{len(static_data):02d}_t_{n * dt:.1f}s.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

        # Прогрес
        if n % (Nt // 10) == 0:
            progress = (n / Nt) * 100
            print(f"Прогрес: {progress:.0f}%")

    # Збереження фінальної матриці температур
    np.save(f'{output_dir}/final_temperature_matrix.npy', T)
    np.savetxt(f'{output_dir}/final_temperature_matrix.csv', T, delimiter=',', fmt='%.2f')
    print(f"Фінальна матриця збережена: .npy та .csv формати")

    # Фінальний статичний графік
    plt.figure(figsize=(10, 6))
    plt.imshow(T.T, cmap='hot', origin='lower', extent=[0, L, 0, L],
               vmin=T0, vmax=max(config.get("T_boundary_edge", 100), config.get("hotspot_temperature", 100)))
    plt.colorbar(label='Температура (°C)')
    plt.title(f'{config["material_name"]} - Фінальний розподіл\n(α = {alpha:.2e} м²/с, t = {simulation_time:.1f} с)')
    plt.xlabel('x (м)')
    plt.ylabel('y (м)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Анімація
    if len(animation_data) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        vmax = max(config.get("T_boundary_edge", 100), config.get("hotspot_temperature", 100))
        im = ax.imshow(animation_data[0].T, cmap='hot', origin='lower',
                       extent=[0, L, 0, L], vmin=T0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Температура (°C)')
        ax.set_xlabel('x (м)')
        ax.set_ylabel('y (м)')

        def update(frame):
            im.set_data(animation_data[frame].T)
            ax.set_title(f'{config["material_name"]} (α = {alpha:.2e} м²/с)\n'
                         f'Час: {animation_times[frame]:.1f} с (кадр {frame + 1}/{len(animation_data)})')
            return [im]

        print(f"Створення анімації...")
        anim = FuncAnimation(fig, update, frames=len(animation_data),
                             interval=800, blit=False, repeat=True)

        # Збереження анімації
        anim.save(f'{output_dir}/animation.gif', writer='pillow', fps=1.5)
        print(f"Анімація збережена: {output_dir}/animation.gif")

        plt.show()

    print(f"\n=== РЕЗУЛЬТАТИ ===")
    print(f"Всі файли збережено в: {output_dir}/")
    print(f"- Конфігурація: config_used.json")
    print(f"- Статичні кадри: {len(static_data)} файлів")
    print(f"- Анімація: animation.gif")
    print(f"- Фінальна матриця: .npy та .csv")
    print(f"- Фінальний графік: final_distribution.png")


if __name__ == "__main__":
    main()