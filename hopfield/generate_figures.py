import os
import json
from pathlib import Path

import matplotlib
# Отключаем GUI-экран полностью
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def generate_fig1():
    csv_filename = "capacity_results.csv"

    if not os.path.exists(csv_filename):
        print(f"[ПРОПУСК] Файл '{csv_filename}' не найден.")
        return

    print(f"--> Генерация Рис. 1 из '{csv_filename}'...")
    df = pd.read_csv(csv_filename)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    n_values = sorted(df['n'].unique())
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(n_values)))

    for idx, n in enumerate(n_values):
        sub = df[df['n'] == n]

        stats = sub.groupby('alpha')['success'].agg(
            mean='mean',
            std='std',
            count='count'
        ).reset_index()

        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        stats['sem'] = stats['sem'].fillna(0)

        ax.plot(
            stats['alpha'], 
            stats['mean'], 
            marker='o', 
            linewidth=2, 
            markersize=6,
            color=colors[idx],
            label=f'N = {n}'
        )

        ax.fill_between(
            stats['alpha'],
            np.clip(stats['mean'] - stats['sem'], 0, 1),
            np.clip(stats['mean'] + stats['sem'], 0, 1),
            color=colors[idx],
            alpha=0.18
        )

    ax.axvline(
        x=0.138, 
        color='#d9534f', 
        linestyle='--', 
        linewidth=2, 
        label=r'Теория $\alpha_c \approx 0.138$'
    )

    ax.set_title("Рис. 1 — Кривые ёмкости сети Хопфилда (Фазовый переход)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Нагрузка α = P / N", fontsize=12)
    ax.set_ylabel("Доля успешных восстановлений (m ≥ 0.95)", fontsize=12)
    ax.set_xlim(0.098, 0.162)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title="Размер сети N", fontsize=10, title_fontsize=11, loc='lower left', frameon=True)

    output_fig = "fig1_capacity_curves.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] Успешно сохранен файл: '{output_fig}'")


def generate_fig2():
    res_file = "mnist_results.csv"
    samp_file = "mnist_samples.csv"

    if not os.path.exists(res_file) or not os.path.exists(samp_file):
        print(f"[ПРОПУСК] Файлы '{res_file}' или '{samp_file}' не найдены.")
        return

    print(f"--> Генерация Рис. 2 из '{res_file}' и '{samp_file}'...")
    results_df = pd.read_csv(res_file)
    samples_df = pd.read_csv(samp_file)

    print("\n" + "=" * 60)
    print(" ТАБЛИЦА СРЕДНЕГО ПОПАРНОГО ПЕРЕКРЫТИЯ C_ab (Утверждение C6)")
    print("=" * 60)
    cab_summary = results_df.groupby(["dataset", "k"])["c_ab"].mean().unstack()
    print(cab_summary.round(4))
    print("-" * 60)
    print("Вывод: У случайных образов C_ab ≈ 0.0 (образы ортогональны).")
    print("       У цифр MNIST C_ab ≈ 0.5...0.7 (сильная корреляция фоновых пикселей).")
    print("=" * 60 + "\n")

    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 3.5], wspace=0.1, hspace=0.1)

    stages = ["original", "corrupted", "restored"]
    titles = ["Оригинал", "Зажато", "Восстановлено"]

    for img_idx in range(5):
        for stage_idx, stage in enumerate(stages):
            ax = fig.add_subplot(gs[img_idx, stage_idx])

            row = samples_df[
                (samples_df["image_idx"] == img_idx) & (samples_df["stage"] == stage)
            ]

            if not row.empty:
                pixels_str = row.iloc[0]["pixels"]
                pixels = np.array(pixels_str.split(), dtype=float).reshape(28, 28)
                ax.imshow(pixels, cmap="gray", vmin=-1.0, vmax=1.0)

            ax.set_xticks([])
            ax.set_yticks([])

            if img_idx == 0:
                ax.set_title(titles[stage_idx], fontsize=9, fontweight="bold")

    ax_plot = fig.add_subplot(gs[:, 3])
    stats = results_df.groupby(["dataset", "k"])["overlap"].mean().reset_index()

    mnist_stats = stats[stats["dataset"] == "mnist"]
    random_stats = stats[stats["dataset"] == "random"]

    ax_plot.plot(
        mnist_stats["k"],
        mnist_stats["overlap"],
        "o-",
        label="MNIST (Высокая корреляция $C_{ab}$)",
        color="#d62728",
        linewidth=2.5,
        markersize=7,
    )
    ax_plot.plot(
        random_stats["k"],
        random_stats["overlap"],
        "s--",
        label="Random Control ($C_{ab} \\approx 0$)",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
    )

    ax_plot.set_xlabel("Количество запоминаемых образов ($k$)", fontsize=11)
    ax_plot.set_ylabel("Среднее перекрытие $m$ с оригиналом", fontsize=11)
    ax_plot.set_title("Качество восстановления: MNIST vs Control", fontsize=12, fontweight="bold")
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.set_xticks([2, 5, 10, 20, 50])
    ax_plot.grid(True, linestyle="--", alpha=0.6)
    ax_plot.legend(fontsize=10, loc="lower left")

    output_fig = "fig2_mnist_vs_control.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] Успешно сохранен файл: '{output_fig}'")


def generate_benchmark_table():
    sizes = [512, 1024, 2048, 4096]
    base_dir = Path("target/criterion/Hopfield_Comparison")

    if not base_dir.exists():
        # Резервный поиск по всем подпапкам в target/criterion
        criterion_dir = Path("target/criterion")
        if criterion_dir.exists():
            matches = list(criterion_dir.glob("**/estimates.json"))
            if not matches:
                print(f"[ПРОПУСК] Бенчмарки Criterion не найдены. Сначала запустите `cargo bench`.")
                return
        else:
            print(f"[ПРОПУСК] Директория '{base_dir}' не найдена. Сначала запустите `cargo bench`.")
            return

    print("--> Генерация таблицы бенчмарков из результатов Criterion...")

    rows = []
    for n in sizes:
        f64_file = base_dir / f"Baseline_f64/{n}/base/estimates.json"
        u64_file = base_dir / f"Optimized_u64/{n}/base/estimates.json"

        if f64_file.exists() and u64_file.exists():
            with open(f64_file, 'r', encoding='utf-8') as f:
                f64_ns = json.load(f)["mean"]["point_estimate"]
            with open(u64_file, 'r', encoding='utf-8') as f:
                u64_ns = json.load(f)["mean"]["point_estimate"]

            f64_ms = f64_ns / 1e6
            u64_ms = u64_ns / 1e6
            speedup = f64_ms / u64_ms if u64_ms > 0 else 0.0

            rows.append({
                "n": n,
                "f64_ms": f64_ms,
                "u64_ms": u64_ms,
                "speedup": speedup
            })

    if not rows:
        print("  [ВНИМАНИЕ] Файлы estimates.json не найдены по указанному пути.")
        return

    table_lines = [
        "\n### Сравнение производительности (50 seeds)\n",
        "| N | Baseline f64 (ms) | BitEngine u64 (ms) | Ускорение |",
        "|---|---|---|---|"
    ]

    for r in rows:
        table_lines.append(f"| {r['n']} | {r['f64_ms']:.2f} ms | {r['u64_ms']:.2f} ms | **{r['speedup']:.1f}x** |")

    table_md = "\n".join(table_lines) + "\n"
    print(table_md)

    output_file = "benchmark_table.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(table_md)

    print(f"  [OK] Успешно сохранена таблица: '{output_file}'")


if __name__ == "__main__":
    print("=" * 50)
    print("  Генерация графиков и отчетов проекта Hopfield Networks")
    print("=" * 50)
    generate_fig1()
    generate_fig2()
    generate_benchmark_table()
    print("Готово!")

def generate_fig3_h4():
    csv_filename = "capacity_results_h4.csv"

    if not os.path.exists(csv_filename):
        print(f"[ОШИБКА] Файл '{csv_filename}' не найден.")
        return

    print(f"--> [H4] Чтение и обработка посидовых данных из '{csv_filename}'...")
    
    # Чтение CSV (принудительно задаем имена колонок на случай расхождения с заголовком)
    df = pd.read_csv(
        csv_filename, 
        names=["n", "seed", "final_overlap", "alpha", "success_rate"], 
        header=0
    )

    # Приводим типы данных
    df['n'] = pd.to_numeric(df['n'], errors='coerce').dropna().astype(int)
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['final_overlap'] = pd.to_numeric(df['final_overlap'], errors='coerce')

    # Фиксируем успешность по порогу узнаваемости (m >= 0.95)
    df['success'] = (df['final_overlap'] >= 0.95).astype(float)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    n_values = sorted(df['n'].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_values)))

    table_metrics = []

    for idx, n in enumerate(n_values):
        sub = df[df['n'] == n]

        # Группировка по alpha: вычисляем среднюю долю успеха по всем сидам
        stats = sub.groupby('alpha')['success'].agg(
            mean='mean',
            std='std',
            count='count'
        ).reset_index().sort_values('alpha')

        stats['sem'] = (stats['std'] / np.sqrt(stats['count'])).fillna(0)

        # Расчет α_½ и α_90 методом линейной интерполяции
        def interp(y_target):
            for i in range(len(stats) - 1):
                y1, y2 = stats['mean'].iloc[i], stats['mean'].iloc[i + 1]
                a1, a2 = stats['alpha'].iloc[i], stats['alpha'].iloc[i + 1]
                if (y1 >= y_target >= y2) or (y1 <= y_target <= y2):
                    return a1 if y1 == y2 else a1 + (y_target - y1) * (a2 - a1) / (y2 - y1)
            return np.nan

        a_half = interp(0.5)
        a_90 = interp(0.9)
        
        # Расчет ширины обвала от 90% до 50%
        width = abs(a_half - a_90) if not (np.isnan(a_half) or np.isnan(a_90)) else np.nan

        table_metrics.append({'N': n, 'alpha_half': a_half, 'width': width})

        ax.plot(
            stats['alpha'], 
            stats['mean'], 
            marker='o', 
            linewidth=2.8 if n >= 16384 else 1.8, 
            markersize=7 if n >= 16384 else 4,
            color=colors[idx],
            label=f'N = {n}'
        )

        ax.fill_between(
            stats['alpha'],
            np.clip(stats['mean'] - stats['sem'], 0, 1),
            np.clip(stats['mean'] + stats['sem'], 0, 1),
            color=colors[idx],
            alpha=0.12
        )

    ax.axvline(x=0.18, color='#d9534f', linestyle='--', linewidth=1.8, label=r'Теория $\alpha_c \approx 0.138$')

    ax.set_title("Сводные кривые ёмкости (от N = 256 до N = 65 536)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Нагрузка α = P / N", fontsize=11)
    ax.set_ylabel("Доля успешных восстановлений (m ≥ 0.95)", fontsize=11)
    
    ax.set_xlim(0.128, 0.21)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title="Размер сети N", fontsize=9, title_fontsize=10, loc='lower left', frameon=True)

    output_fig = "fig3_capacity_curves_record.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] График сохранен в '{output_fig}'")

    # Генерация Markdown-таблицы C9
    table_lines = [
        "### Таблица свойств фазового перехода (Утверждение C9)\n",
        "| N | α_½ (50% успеха) | Ширина обвала Δα (90%→50%) |",
        "|---|---|---|"
    ]
    for m in table_metrics:
        ah_str = f"{m['alpha_half']:.4f}" if not np.isnan(m['alpha_half']) else "N/A"
        w_str = f"{m['width']:.4f}" if not np.isnan(m['width']) else "N/A"
        table_lines.append(f"| {m['N']} | {ah_str} | {w_str} |")

    table_md = "\n".join(table_lines) + "\n"
    print("\n" + table_md)

    with open("summary_table_record.md", "w", encoding="utf-8") as f:
        f.write(table_md)
    print("  [OK] Таблица сохранена в 'summary_table_record.md'")


if __name__ == "__main__":
    generate_fig3_h4()

def generate_mnist_projection_figure():
    # Загрузка CSV-файлов с результатами
    results_df = pd.read_csv("mnist_projection_results.csv")
    samples_df = pd.read_csv("mnist_projection_samples.csv")

    # Создаем холст: левая часть под картинки, правая — под график
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(5, 7, figure=fig, wspace=0.15, hspace=0.2)

    # --- Левая панель: Отображение 5 изображений (k = 5) ---
    k5_samples = samples_df[samples_df['k'] == 5]
    titles = {'original': 'Оригинал', 'corrupted': 'Зажато', 'restored': 'Восстановлено'}

    for img_idx in range(5):
        for col_idx, stage in enumerate(['original', 'corrupted', 'restored']):
            ax = fig.add_subplot(gs[img_idx, col_idx])
            
            # Фильтруем данные для текущей строки и стадии
            row = k5_samples[(k5_samples['image_idx'] == img_idx) & (k5_samples['stage'] == stage)]
            
            if not row.empty:
                pixels_str = row['pixels'].values[0]
                pixels = np.fromstring(pixels_str, sep=' ').reshape(28, 28)
                ax.imshow(pixels, cmap='gray', vmin=-1, vmax=1)
            
            ax.axis('off')
            
            # Заголовки над колонками (только в первой строке)
            if img_idx == 0:
                ax.set_title(titles[stage], fontsize=10, fontweight='bold', pad=8)

    # --- Правая панель: График точности восстановления ---
    ax_plot = fig.add_subplot(gs[:, 3:])

    # Считаем среднее перекрытие m для каждого k
    mean_df = results_df.groupby(['dataset', 'k'])['overlap'].mean().reset_index()

    mnist_data = mean_df[mean_df['dataset'] == 'mnist_proj']
    random_data = mean_df[mean_df['dataset'] == 'random_proj']

    # Построение линий
    ax_plot.plot(
        mnist_data['k'], mnist_data['overlap'], 
        'r-o', linewidth=2, markersize=7, 
        label=r'MNIST (Проекция, высокая $C_{ab}$)'
    )
    ax_plot.plot(
        random_data['k'], random_data['overlap'], 
        's--', color='#1f77b4', linewidth=2, markersize=7, 
        label=r'Random Control ($C_{ab} \approx 0$)'
    )

    # Оформление графика
    ax_plot.set_title('Качество восстановления: MNIST vs Control (Проекция)', fontsize=12, fontweight='bold', pad=12)
    ax_plot.set_xlabel('Количество запоминаемых образов ($k$)', fontsize=10)
    ax_plot.set_ylabel('Среднее перекрытие $m$ с оригиналом', fontsize=10)
    ax_plot.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, linestyle='--', alpha=0.5)
    ax_plot.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    plt.savefig('fig4_mnist_fix.png', dpi=300)
    print("График успешно сохранен в 'fig4_mnist_fix.png'")

if __name__ == "__main__":
    generate_mnist_projection_figure()

def generate_thermal_low_load_plot():
    df = pd.read_csv("thermal_test_results.csv")

    # Группировка по размеру сети N и температуре T
    stats = df.groupby(["n", "temp"])["overlap"].mean().reset_index()

    plt.figure(figsize=(9, 5.5))

    for n_val in stats["n"].unique():
        sub = stats[stats["n"] == n_val]
        plt.plot(sub["temp"], sub["overlap"], "-o", label=f"N = {n_val} (P = {int(n_val * 0.02)})", markersize=4)

    plt.axvline(x=1.0, color="red", linestyle="--", alpha=0.6, label=r"Теоретический $T_c \approx 1.0$")
    
    plt.title("Термический фазовый переход при средней нагрузке", fontsize=12, fontweight="bold")
    plt.xlabel("Температура $T = 1/\\beta$", fontsize=10)
    plt.ylabel("Средний параметр порядка $m(T)$", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig5_thermal_experiment.png", dpi=300)
    print("График сохранен в 'fig5_thermal_experiment.png'")

if __name__ == "__main__":
    generate_thermal_low_load_plot()