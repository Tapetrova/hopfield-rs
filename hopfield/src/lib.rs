use std::fs::OpenOptions;
use std::io::Write;

use mnist::{Mnist, MnistBuilder};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use nalgebra::DMatrix;

// ---------------------------------------------------------------------
// Дополнительно:
// Добавление температуры в сеть
// ---------------------------------------------------------------------


pub fn thermal_low_load_experiment() {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("thermal_test_results.csv")
        .expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "n,p,temp,seed,pattern_idx,overlap");

    let n_values = vec![500, 1000]; // Различные размеры сети N
    let alpha = 0.02;                     // Нормальная нагрузка P/N = 0.1
    let num_seeds = 10;                  // 1 сидов на каждую конфигурацию
    let mc_sweeps = 30;                  // Число термических обновлений (sweeps)

    println!("Запуск термического эксперимента (50 сидов, сканирование T)...");

    for &n in &n_values {
        let p = (n as f64 * alpha).max(1.0) as usize; // P ≈ 0.1 * N

        for seed in 0..num_seeds as u64 {
            let states = generate_states(p, n, seed + 1000);
            let weights = weight_matrix_calculate(&states);

            // Сканирование температуры T от 0.05 до 2.00 с шагом 0.05
            let mut temp = 0.05;
            while temp <= 2.00 {
                let beta = 1.0 / temp;

                for (pattern_idx, pattern) in states.iter().enumerate() {
                    let mut state = pattern.clone();

                    // Релаксация с тепловым шумом
                    for sweep in 0..mc_sweeps {
                        thermal_neuron_fix(
                            &mut state,
                            &weights,
                            beta,
                            seed + sweep as u64 + 2000,
                        );
                    }

                    // Подсчет итогового перекрытия m
                    let m = calculate_overlap(&state, pattern);
                    let _ = writeln!(
                        file,
                        "{},{},{:.2},{},{},{:.4}",
                        n, p, temp, seed, pattern_idx, m
                    );
                }
                temp += 0.05;
            }
        }
        println!("Эксперимент для N = {} полностью завершен!", n);
    }

    file.flush().unwrap();
    println!("Данные успешно записаны в 'thermal_test_results.csv'");
}

pub fn thermal_neuron_fix(
    state: &mut [f64],
    weights: &[Vec<f64>],
    beta: f64,
    seed: u64,
) -> usize {
    let n = state.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut flips = 0;

    for i in 0..n {
        // 1. Вычисляем локальное поле h_i для нейрона i
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weights[i][j] * state[j];
        }

        // 2. Считаем вероятность P(s_i = +1)
        let prob_plus = 1.0 / (1.0 + (-2.0 * beta * h_i).exp());

        // 3. Принимаем вероятностное решение
        let random_val: f64 = rng.gen_range(0.0..1.0); // Случайное число в диапазоне [0.0, 1.0)
        let new_state = if random_val < prob_plus { 1.0 } else { -1.0 };

        if new_state != state[i] {
            state[i] = new_state;
            flips += 1;
        }
    }

    flips
}


// ---------------------------------------------------------------------
// Дополнительно:
// MNIST фикс
// ---------------------------------------------------------------------

// Функция расчета весов по новой формуле
pub fn projection_weight_matrix(patterns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = patterns.len();       // Количество паттернов P (например, 50)
    let n = patterns[0].len();    // Количество нейронов N (например, 784)

    // Формируем матрицу Xi размера (N x P): столбцы — это образцы
    let mut xi_data = Vec::with_capacity(n * p);
    for i in 0..n {
        for mu in 0..p {
            xi_data.push(patterns[mu][i]);
        }
    }
    let xi = DMatrix::from_row_slice(n, p, &xi_data);

    // C = Xi^T * Xi (размер P x P)
    let c = xi.transpose() * &xi;

    // Находим обратную матрицу (P x P)
    let c_inv = c.try_inverse()
        .expect("Матрица Xi^T * Xi вырождена и не может быть обращена!");

    // W = Xi * C_inv * Xi^T (размер N x N)
    let w_matrix = &xi * c_inv * xi.transpose();

    // Преобразуем результат обратно в Vec<Vec<f64>>
    // Диагональ W[i][i] не обнуляется!
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = w_matrix[(i, j)];
        }
    }

    result
}

// Функция эксперимента с измененным подсчетом весов
pub fn mnist_projection_experiment() {
    use std::fs::File;
    use std::io::Write;

    let k_values = vec![2, 5, 10, 20, 50, 100, 200];
    let num_seeds = 10; // Количество запусков на каждое k

    // 1. Файл с метриками точности, C_ab и оверлапами
    let mut csv_file: File =
        File::create("mnist_projection_results.csv").expect("Не удалось создать mnist_projection_results.csv");
    writeln!(csv_file, "dataset,k,seed,image_idx,c_ab,overlap,success").unwrap();

    // 2. Файл с пикселями картинок для визуализации
    let mut samples_file =
        File::create("mnist_projection_samples.csv").expect("Не удалось создать mnist_projection_samples.csv");
    writeln!(samples_file, "k,image_idx,stage,pixels").unwrap();

    // 3. Создаем маску clamping: верхняя половина заморожена (true), нижняя свободна (false)
    let n = 784; // 28x28 пикселей
    let mut mask = vec![false; n];
    for i in 0..n / 2 {
        mask[i] = true; // Зажимаем верхние 392 нейрона
    }

    for &k in &k_values {
        for seed_idx in 0..num_seeds {
            let seed = seed_idx + k * 100;

            // --- 1. MNIST С ПРОЕКЦИОННЫМ ПРАВИЛОМ ---
            let mnist_patterns = load_mnist_binarized(k);
            
            // Расчет весов через проекционное правило вместо weight_matrix_calculate
            let weights_mnist = projection_weight_matrix(&mnist_patterns);
            let c_ab_mnist = calculate_pairwise_overlap(&mnist_patterns);

            for (img_idx, pattern) in mnist_patterns.iter().enumerate() {
                // Искажение нижней половины
                let mut state = corrupt_lower_half(pattern, seed as u64 + img_idx as u64);
                let corrupted_copy = state.clone();

                // Запуск восстановления с CLAMPING
                for iter in 0..100 {
                    let changed = neuron_fix(
                        &mut state,
                        &weights_mnist,
                        seed as u64 + iter as u64 + 500,
                        Some(&mask), // Фиксируем верхнюю половину
                    );
                    if changed == 0 {
                        break;
                    }
                }

                // Сравниваем результат с оригиналом
                let m = calculate_overlap(&state, pattern);
                let success = if m >= 0.95 { 1 } else { 0 };

                writeln!(
                    csv_file,
                    "mnist_proj,{},{},{},{:.4},{:.4},{}",
                    k, seed, img_idx, c_ab_mnist, m, success
                )
                .unwrap();

                // Сохранение состояний для k = 5 и seed_idx = 0
                if k == 5 && seed_idx == 0 {
                    let orig_str: Vec<String> = pattern.iter().map(|p| p.to_string()).collect();
                    let corr_str: Vec<String> =
                        corrupted_copy.iter().map(|p| p.to_string()).collect();
                    let rest_str: Vec<String> = state.iter().map(|p| p.to_string()).collect();

                    writeln!(
                        samples_file,
                        "{},{},original,\"{}\"",
                        k, img_idx, orig_str.join(" ")
                    )
                    .unwrap();
                    writeln!(
                        samples_file,
                        "{},{},corrupted,\"{}\"",
                        k, img_idx, corr_str.join(" ")
                    )
                    .unwrap();
                    writeln!(
                        samples_file,
                        "{},{},restored,\"{}\"",
                        k, img_idx, rest_str.join(" ")
                    )
                    .unwrap();
                }
            }

            // --- 2. RANDOM CONTROL С ПРОЕКЦИОННЫМ ПРАВИЛОМ ---
            let random_patterns = generate_states(k, 784, seed as u64);
            
            // Расчет весов случайного контроля через проекционное правило
            let weights_random = projection_weight_matrix(&random_patterns);
            let c_ab_random = calculate_pairwise_overlap(&random_patterns);

            for (img_idx, pattern) in random_patterns.iter().enumerate() {
                let mut state = corrupt_lower_half(pattern, seed as u64 + img_idx as u64);

                for iter in 0..100 {
                    let changed = neuron_fix(
                        &mut state,
                        &weights_random,
                        seed as u64 + iter as u64 + 500,
                        Some(&mask),
                    );
                    if changed == 0 {
                        break;
                    }
                }

                let m = calculate_overlap(&state, pattern);
                let success = if m >= 0.95 { 1 } else { 0 };

                writeln!(
                    csv_file,
                    "random_proj,{},{},{},{:.4},{:.4},{}",
                    k, seed, img_idx, c_ab_random, m, success
                )
                .unwrap();
            }
        }
    }

    csv_file.flush().unwrap();
    samples_file.flush().unwrap();

    println!("Проекционный эксперимент успешно завершен!");
    println!("Созданы файлы: 'mnist_projection_results.csv' и 'mnist_projection_samples.csv'.");
}

// ---------------------------------------------------------------------
// Експеримент с рекордными замерами
// ---------------------------------------------------------------------

pub fn record_capacity_experiment() {
    println!("=== Динамический расчет емкости (авто-остановка) ===");

    let experiments = [
        (256, 50),
        (1024, 50),
        (4096, 30),
        (16384, 30),
        (65536, 12),
    ];

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("capacity_results_h4.csv")
        .expect("Не удалось открыть capacity_results_h4.csv");

    // Записываем корректный 5-колоночный заголовок
    writeln!(file, "n,seed,final_overlap,alpha,success_rate").unwrap();

    for (n, num_seeds) in experiments {
        println!("\n--> Расчет N = {}, сидов = {}", n, num_seeds);

        let mut alpha = 0.1300;
        let alpha_step = 0.0025;

        loop {
            let p = (alpha * n as f64).round() as usize;
            let mut overlaps = Vec::with_capacity(num_seeds);

            for seed_idx in 0..num_seeds {
                let seed = seed_idx as u64;

                let states_f64 = generate_states(p, n, seed + 1000);
                let packed_patterns: Vec<Vec<u64>> = states_f64.iter().map(|pat| pack_bits(pat)).collect();

                let noisy_f64 = apply_noise(&states_f64[0], 0.05, seed + 2000);
                let mut state = pack_bits(&noisy_f64);

                let mut q_overlaps = calculate_initial_overlaps(&state, &packed_patterns, n);

                neuron_fix_bit(
                    &mut state,
                    &packed_patterns,
                    &mut q_overlaps,
                    n,
                    seed,
                    None,
                );

                let final_state_f64 = unpack_bits(&state, n);
                let final_overlap = calculate_overlap(&final_state_f64, &states_f64[0]);
                
                overlaps.push(final_overlap);
            }

            // Считаем общий успех для текущей alpha по всем сидам
            let success_count = overlaps.iter().filter(|&&m| m >= 0.95).count();
            let success_rate = success_count as f64 / num_seeds as f64;

            // Записываем результат каждого сида с уже посчитанным итоговым success_rate
            for (seed_idx, &overlap) in overlaps.iter().enumerate() {
                writeln!(file, "{},{},{:.6},{:.4},{:.4}", n, seed_idx, overlap, alpha, success_rate).unwrap();
            }
            file.flush().unwrap();

            println!("  N = {:5} | α = {:.4} (P = {:5}) | Доля успеха: {:.1}%", n, alpha, p, success_rate * 100.0);

            // Условие остановки: завершаем прогон для текущего N, когда успех упал ниже 10%
            if success_rate < 0.50 {
                println!("  [STOP] Доля успеха упала ниже 50%. Переход к следующему N.");
                break;
            }

            alpha += alpha_step;
        }
    }
}

// ---------------------------------------------------------------------
// Експеримент с мнистом
// ---------------------------------------------------------------------

pub fn mnist_experiment() {
    use std::fs::File;
    use std::io::Write;

    let k_values = vec![2, 5, 10, 20, 50];
    let num_seeds = 10; // Количество запусков на каждое k

    // 1. Файл с метриками точности и C_ab
    let mut csv_file: File =
        File::create("mnist_results.csv").expect("Не удалось создать mnist_results.csv");
    writeln!(csv_file, "dataset,k,seed,image_idx,c_ab,overlap,success").unwrap();

    // 2. Файл с пикселями картинок
    let mut samples_file =
        File::create("mnist_samples.csv").expect("Не удалось создать mnist_samples.csv");
    writeln!(samples_file, "k,image_idx,stage,pixels").unwrap();

    // 3. Создаем маску clamping: верхняя половина заморожена (true), нижняя свободна (false)
    let n = 784; // 28x28 пикселей
    let mut mask = vec![false; n];
    for i in 0..n / 2 {
        mask[i] = true; // Зажимаем верхние 392 нейрона
    }

    for &k in &k_values {
        for seed_idx in 0..num_seeds {
            let seed = seed_idx + k * 100;

            // --- 1. MNIST ---
            let mnist_patterns = load_mnist_binarized(k);
            let weights_mnist = weight_matrix_calculate(&mnist_patterns);
            let c_ab_mnist = calculate_pairwise_overlap(&mnist_patterns);

            for (img_idx, pattern) in mnist_patterns.iter().enumerate() {
                // Ломаем только нижнюю половину
                let mut state = corrupt_lower_half(pattern, seed as u64 + img_idx as u64);
                let corrupted_copy = state.clone();

                // Запуск восстановления с CLAMPING
                for iter in 0..100 {
                    let changed = neuron_fix(
                        &mut state,
                        &weights_mnist,
                        seed as u64 + iter as u64 + 500,
                        Some(&mask), // <-- Фиксируем верхнюю половину
                    );
                    if changed == 0 {
                        break;
                    }
                }

                // Сравниваем результат с оригиналом
                let m = calculate_overlap(&state, pattern);
                let success = if m >= 0.95 { 1 } else { 0 };

                writeln!(
                    csv_file,
                    "mnist,{},{},{},{:.4},{:.4},{}",
                    k, seed, img_idx, c_ab_mnist, m, success
                )
                .unwrap();

                // Для k = 5 на первом сиде сохраняем все 3 состояния для визуализации
                if k == 5 && seed_idx == 0 {
                    let orig_str: Vec<String> = pattern.iter().map(|p| p.to_string()).collect();
                    let corr_str: Vec<String> =
                        corrupted_copy.iter().map(|p| p.to_string()).collect();
                    let rest_str: Vec<String> = state.iter().map(|p| p.to_string()).collect();

                    writeln!(
                        samples_file,
                        "{},{},original,\"{}\"",
                        k,
                        img_idx,
                        orig_str.join(" ")
                    )
                    .unwrap();
                    writeln!(
                        samples_file,
                        "{},{},corrupted,\"{}\"",
                        k,
                        img_idx,
                        corr_str.join(" ")
                    )
                    .unwrap();
                    writeln!(
                        samples_file,
                        "{},{},restored,\"{}\"",
                        k,
                        img_idx,
                        rest_str.join(" ")
                    )
                    .unwrap();
                }
            }

            // --- 2. RANDOM CONTROL (по тому же протоколу clamping!) ---
            let random_patterns = generate_states(k, 784, seed as u64);
            let weights_random = weight_matrix_calculate(&random_patterns);
            let c_ab_random = calculate_pairwise_overlap(&random_patterns);

            for (img_idx, pattern) in random_patterns.iter().enumerate() {
                let mut state = corrupt_lower_half(pattern, seed as u64 + img_idx as u64);

                for iter in 0..100 {
                    let changed = neuron_fix(
                        &mut state,
                        &weights_random,
                        seed as u64 + iter as u64 + 500,
                        Some(&mask), // <-- Также применяем clamping
                    );
                    if changed == 0 {
                        break;
                    }
                }

                let m = calculate_overlap(&state, pattern);
                let success = if m >= 0.95 { 1 } else { 0 };

                writeln!(
                    csv_file,
                    "random,{},{},{},{:.4},{:.4},{}",
                    k, seed, img_idx, c_ab_random, m, success
                )
                .unwrap();
            }
        }
    }

    // Гарантированно выталкиваем данные из оперативной памяти на диск
    csv_file.flush().unwrap();
    samples_file.flush().unwrap();

    println!("Эксперимент успешно завершен!");
    println!("Созданы файлы: 'mnist_results.csv' и 'mnist_samples.csv'.");
}

pub fn calculate_pairwise_overlap(patterns: &[Vec<f64>]) -> f64 {
    let k = patterns.len();
    if k <= 1 {
        return 1.0;
    }

    let n = patterns[0].len() as f64;
    let mut total_overlap = 0.0;
    let mut pair_count = 0;

    for i in 0..k {
        for j in (i + 1)..k {
            let dot_product: f64 = patterns[i]
                .iter()
                .zip(patterns[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            total_overlap += dot_product / n;
            pair_count += 1;
        }
    }

    total_overlap / (pair_count as f64)
}

/// Упаковка вектора f64 (+1.0 / -1.0) в битовый массив u64
pub fn pack_bits(vec: &[f64]) -> Vec<u64> {
    let mut packed = vec![0u64; vec.len().div_ceil(64)];
    for (i, &val) in vec.iter().enumerate() {
        if val < 0.0 {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    packed
}

/// Распаковка Vec<u64> обратно в Vec<f64> для подсчета метрик
pub fn unpack_bits(packed: &[u64], n: usize) -> Vec<f64> {
    let mut vec = Vec::with_capacity(n);
    for i in 0..n {
        let bit = (packed[i / 64] >> (i % 64)) & 1;
        vec.push(if bit == 1 { -1.0 } else { 1.0 });
    }
    vec
}

/// Расчет начального массива перекрытий m_mu через XOR и popcount
pub fn calculate_initial_overlaps(state: &[u64], patterns: &[Vec<u64>], n: usize) -> Vec<i32> {
    let n_i32 = n as i32;
    patterns
        .iter()
        .map(|pat| {
            let mut popcnt = 0u32;
            for (w1, w2) in state.iter().zip(pat.iter()) {
                popcnt += (w1 ^ w2).count_ones();
            }
            // Формула: q_mu = N - 2 * popcount
            n_i32 - 2 * (popcnt as i32)
        })
        .collect()
}

// Восстановления образа без матрицы весов
pub fn neuron_fix_bit(
    state: &mut [u64],
    patterns: &[Vec<u64>],
    q_overlaps: &mut [i32], // Перевод с f64 на i32
    n: usize,
    seed: u64,
    clamped_mask: Option<&[bool]>,
) -> usize {
    let mut indices: Vec<usize> = (0..n).collect();

    if let Some(mask) = clamped_mask {
        indices.retain(|&i| !mask[i]);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let p = patterns.len() as i32;
    let mut changed_count = 0;

    for &i in &indices {
        let word_idx = i / 64;
        let bit_mask = 1u64 << (i % 64);

        // Бит 1 => s_i = -1, Бит 0 => s_i = +1
        let s_i_is_minus = (state[word_idx] & bit_mask) != 0;
        let s_i: i32 = if s_i_is_minus { -1 } else { 1 };

        // H_i = N * h_i = sum_mu(xi_i^mu * q_mu) - P * s_i
        let mut h_i_scaled: i32 = 0;
        for (mu, pat) in patterns.iter().enumerate() {
            let xi_is_minus = (pat[word_idx] & bit_mask) != 0;
            let xi_i: i32 = if xi_is_minus { -1 } else { 1 };

            h_i_scaled += xi_i * q_overlaps[mu];
        }
        h_i_scaled -= p * s_i;

        let new_s_i = if h_i_scaled > 0 {
            1
        } else if h_i_scaled < 0 {
            -1
        } else {
            s_i
        };

        // Если нейрон изменил состояние
        if new_s_i != s_i {
            let delta = new_s_i - s_i; // Ровно +2 или -2

            // Обновляем бит в векторе состояния
            if new_s_i == -1 {
                state[word_idx] |= bit_mask;
            } else {
                state[word_idx] &= !bit_mask;
            }

            // Быстрое инкрементальное обновление q_mu без пересчета всего массива
            for (mu, pat) in patterns.iter().enumerate() {
                let xi_is_minus = (pat[word_idx] & bit_mask) != 0;
                let xi_i: i32 = if xi_is_minus { -1 } else { 1 };
                q_overlaps[mu] += xi_i * delta;
            }

            changed_count += 1;
        }
    }

    changed_count
}

// Считает перекрытие с заданным образом
pub fn calculate_overlap(state: &[f64], pattern: &[f64]) -> f64 {
    let n = state.len() as f64;
    let dot_product: f64 = state.iter().zip(pattern.iter()).map(|(a, b)| a * b).sum();

    dot_product / n
}

// Функция которая перемешивает только нижнюю часть картинки MNIST
pub fn corrupt_lower_half(pattern: &[f64], seed: u64) -> Vec<f64> {
    let mut corrupted = pattern.to_vec();
    let n = corrupted.len();
    let half = n / 2; // Для N = 784 это 392

    let mut rng = StdRng::seed_from_u64(seed);

    // Стираем и зашумляем нижнюю половину (индексы 392..784)
    for i in half..n {
        corrupted[i] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    }

    corrupted
}

// ---------------------------------------------------------------------
// Експеримент емкости
// ---------------------------------------------------------------------

pub fn capacity_experiment() {
    use std::fs::File;
    use std::io::Write;

    let n_values = vec![256, 512, 1024, 2048, 4096];
    let num_seeds = 20;
    let main_seed = rand::thread_rng().gen_range(0..10000);

    let mut file = File::create("capacity_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "n,alpha,p,seed,overlap,success").unwrap();

    println!("Запуск эксперимента 1");

    for &n in &n_values {
        println!("Расчет для N = {}...", n);

        let mut alpha = 0.10;
        while alpha <= 0.1601 {
            // 0.1601 из-за погрешности float
            let p = (alpha * n as f64).round() as usize;

            for seed_idx in 0..num_seeds {
                let seed = main_seed + seed_idx;

                //Генерация состояний и расчет весов (используем твои функции)
                let states = generate_states(p, n, seed + 1000);
                let weights = weight_matrix_calculate(&states);

                //Портим 5% нейронов у первого образа
                let target_pattern = states[0].clone();
                let mut noisy_state = target_pattern.clone();

                let mut rng = StdRng::seed_from_u64(seed + 2000);
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);

                for &i in indices.iter().take(n / 20) {
                    noisy_state[i] *= -1.0;
                }

                //Динамика до остановки (максимум 100 проходов)
                for iter in 0..100 {
                    let prev_state = noisy_state.clone();
                    neuron_fix(&mut noisy_state, &weights, seed + iter as u64 + 3000, None);

                    // Если ни один нейрон не изменился за проход — останавливаемся
                    if noisy_state == prev_state {
                        break;
                    }
                }

                //Считаем перекрытие m = (1/N) * sum(x_i * y_i)
                let matches_sum: f64 = target_pattern
                    .iter()
                    .zip(noisy_state.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let overlap = matches_sum / (n as f64);
                let success = if overlap >= 0.95 { 1 } else { 0 };

                //Запись результатов в CSV
                writeln!(
                    file,
                    "{},{:.3},{},{},{:.4},{}",
                    n, alpha, p, seed, overlap, success
                )
                .unwrap();
            }

            alpha += 0.005;
        }
    }

    println!("Готово! Данные сохранены в файл 'capacity_results.csv'");
}

// Функция генерирующая случайные
pub fn generate_states(p: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut states_vector = Vec::with_capacity(p);

    for _ in 0..p {
        let mut states = Vec::with_capacity(n);
        for _ in 0..n {
            let val = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            states.push(val);
        }
        states_vector.push(states);
    }

    states_vector
}

// Создает матрицу весов
pub fn weight_matrix_calculate(states: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Если нет паттернов, возвращаем пустую матрицу
    if states.is_empty() {
        return vec![];
    }

    let n = states[0].len(); // Длина одного паттерна (количество нейронов N)
    let mut weight_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                weight_matrix[i][j] = 0.0;
            } else {
                let mut sum = 0.0;
                for state in states {
                    sum += state[i] * state[j];
                }
                weight_matrix[i][j] = sum / (n as f64);
            }
        }
    }

    weight_matrix
}

// Восстанавливает искаженный образ(1 проход)
pub fn neuron_fix(
    states: &mut [f64],
    weight_matrix: &[Vec<f64>],
    seed: u64,
    clamped_mask: Option<&[bool]>,
) -> usize {
    let n = states.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Если передана маска, отсеиваем зажатые (true) нейроны из очереди на обновление
    if let Some(mask) = clamped_mask {
        indices.retain(|&i| !mask[i]);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let mut changed_count = 0;

    for &i in &indices {
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weight_matrix[i][j] * states[j];
        }

        let new_val = if h_i > 0.0 {
            1.0
        } else if h_i < 0.0 {
            -1.0
        } else {
            states[i]
        };

        if new_val != states[i] {
            states[i] = new_val;
            changed_count += 1;
        }
    }

    changed_count
}

// Восстанавливает образ (синхронно) (Нужен был только для 1го эксперимента)
pub fn neuron_fix_sync(states: &mut [f64], weight_matrix: &[Vec<f64>]) {
    let n = states.len();
    let old_states = states.to_owned(); // Фиксируем состояние всех нейронов до обновления

    for i in 0..n {
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weight_matrix[i][j] * old_states[j];
        }

        if h_i > 0.0 {
            states[i] = 1.0;
        } else if h_i < 0.0 {
            states[i] = -1.0;
        }
    }
}

// Расчитывает енергию
pub fn calculate_energy(states: &[f64], weight_matrix: &[Vec<f64>]) -> f64 {
    let n = states.len();
    let mut energy_sum = 0.0;

    for i in 0..n {
        for j in 0..n {
            energy_sum += weight_matrix[i][j] * states[i] * states[j];
        }
    }

    -0.5 * energy_sum
}

// Функция генерирующая случайный N
pub fn generate_n(seed: u64) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(500..=1000)
}

// Случайно искажаем образ
pub fn apply_noise(pattern: &[f64], noise_ratio: f64, seed: u64) -> Vec<f64> {
    let mut noisy_state = pattern.to_vec();
    let n = noisy_state.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    // Считаем количество элементов для инвертирования (например, 0.05 = 5%)
    let flip_count = ((n as f64) * noise_ratio).round() as usize;

    for &idx in indices.iter().take(flip_count) {
        noisy_state[idx] *= -1.0;
    }

    noisy_state
}

// Загрузчик MNIST
pub fn load_mnist_binarized(count: usize) -> Vec<Vec<f64>> {
    let Mnist { trn_img, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(count as u32)
        .validation_set_length(0)
        .test_set_length(0)
        .finalize();

    let mut images = Vec::with_capacity(count);

    // Каждая картинка занимает 28x28 = 784 байт
    for chunk in trn_img.chunks(784) {
        let binarized: Vec<f64> = chunk
            .iter()
            .map(|&pixel| if pixel > 127 { 1.0 } else { -1.0 })
            .collect();

        images.push(binarized);
    }

    images
}

#[test]
pub fn test_fixed_point() {
    use std::fs::File;
    use std::io::Write;

    let mut file =
        File::create("test_fixed_point_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "n,seed,state_id").unwrap();
    for iter in 0..50 {
        let seed = iter;
        let n = generate_n(seed);
        let p = 10;

        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);

        for (i, states) in states.iter().enumerate() {
            let mut state = states.clone();
            neuron_fix(&mut state, &weights, seed + 2000, None);
            writeln!(file, "{},{},{}", n, seed, i).unwrap();
            assert_eq!(
                &state, states,
                "Паттерн {} должен оставаться неподвижной точкой. Непрошедший сид:{}",
                i, seed
            );
        }
    }
}

#[test]
pub fn test_noise_10() {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("test_noise_10_results.csv").expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "seed,state,percent");
    for seed_iter in 0..50 {
        let seed = seed_iter;
        let n = generate_n(seed);
        let p = 10;
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);
        let mut index = 0;

        for states in &states {
            let mut noisy_state = apply_noise(states, 0.1, seed + 2000);

            // Делаем несколько итераций восстановления

            for iter in 0..15 {
                let changed = neuron_fix(&mut noisy_state, &weights, seed + iter + 3000, None);
                if changed == 0 {
                    break;
                }
            }

            // Считаем % совпадения
            let matches = states
                .iter()
                .zip(noisy_state.iter())
                .filter(|(a, b)| a == b)
                .count();

            let accuracy = matches as f64 / n as f64;
            let _ = writeln!(file, "{},{},{},{:.0}", seed, index, n, accuracy * 100.00);
            assert!(
                accuracy >= 0.99,
                "Точность восстановления должна быть >= 99%, получили: {:.2}%",
                accuracy * 100.0
            );

            index += 1;
        }
    }
}

#[test]
pub fn low_load_test() {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("low_load_test_results.csv").expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "seed");
    for iter in 0..50 {
        let seed = iter;
        let n = 1000;
        let p = 20;
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);
        let _ = writeln!(file, "{}", seed);
        for _ in 0..p {
            for states in &states {
                let mut state = states.clone();
                neuron_fix(&mut state, &weights, seed + 2000, None);
                assert_eq!(
                    &state, states,
                    "При alpha <= 0.02 образ должен быть точной неподвижной точкой"
                );
            }
        }
    }
}

#[test]
pub fn basic_drop_test() {
    use std::fs::File;
    use std::io::Write;

    let mut file =
        File::create("basic_drop_test_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "seed,acc_100,acc_200").unwrap();

    let n = 1000;
    let sample_count = 3; // Сколько образов будут тестироваться

    for iter in 0..50 {
        let seed = iter as u64;

        // Функция считает среднюю точность ровно по 10 образцам
        let evaluate_10_patterns = |p: usize| -> f64 {
            let states = generate_states(p, n, seed + 1000);
            let weights = weight_matrix_calculate(&states);
            let mut total_matches = 0;

            // Берем первые 10 образцов из сгенерированных P
            for (idx, target_pattern) in states.iter().take(sample_count).enumerate() {
                let mut noisy_state = apply_noise(target_pattern, 0.05, seed + 2000 + idx as u64);

                for iter in 0..15 {
                    let changed =
                        neuron_fix(&mut noisy_state, &weights, seed + iter as u64 + 3000, None);
                    if changed == 0 {
                        break;
                    }
                }

                total_matches += target_pattern
                    .iter()
                    .zip(noisy_state.iter())
                    .filter(|(a, b)| a == b)
                    .count();
            }

            // Среднее совпадение по 10 образцам
            (total_matches as f64) / ((sample_count * n) as f64)
        };

        let acc_100 = evaluate_10_patterns(100);
        let acc_200 = evaluate_10_patterns(200);

        writeln!(file, "{}, {:.4}, {:.4}", seed, acc_100, acc_200).unwrap();

        assert!(
            acc_100 >= 0.95,
            "При P=100 средняя точность по 10 образцам должна быть >= 95%, получили: {:.2}% (сид: {})",
            acc_100 * 100.0,
            seed
        );

        assert!(
            acc_200 <= 0.85,
            "При P=200 средняя точность по 10 образцам должна упасть <= 85%, получили: {:.2}% (сид: {})",
            acc_200 * 100.0,
            seed
        );
    }
}

#[test]
pub fn test_energy() {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("test_energy_results.csv").expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "seed");
    for seed_iter in 0..50 {
        let seed: u64 = seed_iter;
        let n = 1000;
        let p = 20;
        let _ = writeln!(file, "{}", seed);
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);

        for (idx, state) in states.iter().enumerate() {
            let mut noisy_state = apply_noise(state, 0.05, seed + 2000 + idx as u64);

            for iter in 0..5 {
                // Перенесённая логика neuron_fix с проверкой энергии на каждый переворот бита
                let mut indices: Vec<usize> = (0..n).collect();
                let mut rng = StdRng::seed_from_u64(seed + iter + 3000);
                indices.shuffle(&mut rng);

                let mut changed_count = 0;
                let mut e_before = calculate_energy(&noisy_state, &weights);

                for &i in &indices {
                    let mut h_i = 0.0;
                    for j in 0..n {
                        h_i += weights[i][j] * noisy_state[j];
                    }

                    let new_val = if h_i > 0.0 {
                        1.0
                    } else if h_i < 0.0 {
                        -1.0
                    } else {
                        noisy_state[i]
                    };

                    // Проверяем энергию СРАЗУ же при изменении каждого конкретного бита
                    if new_val != noisy_state[i] {
                        noisy_state[i] = new_val;
                        changed_count += 1;

                        let e_after = calculate_energy(&noisy_state, &weights);
                        assert!(
                            e_before >= e_after,
                            "Ошибка! Энергия выросла при изменении нейрона {}: До = {:.4}, После = {:.4}",
                            i,
                            e_before,
                            e_after
                        );
                        e_before = e_after;
                    }
                }

                // Если за полный проход ни один нейрон не изменился — достигли локального минимума
                if changed_count == 0 {
                    break;
                }
            }
        }
    }
}

#[test]
pub fn sync_vs_async_test() {
    use std::fs::File;
    use std::io::Write;

    let mut found_oscillation = false;
    let mut file =
        File::create("test_async_vs_sync_results.csv").expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "seed");
    // Ищем случай, где синхронный режим зацикливается
    for seed in 0..1000 {
        let _ = writeln!(file, "{}", seed);
        let n = 500;
        let p = 60;

        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);
        let initial_noisy = apply_noise(&states[0], 0.2, seed + 2000);

        // Проверяем СИНХРОННЫЙ режим
        let mut sync_state = initial_noisy.clone();
        let mut history: Vec<Vec<f64>> = vec![sync_state.clone()];
        let mut is_oscillating = false;

        for _ in 0..30 {
            neuron_fix_sync(&mut sync_state, &weights);

            let len = history.len();
            // Cостояние совпало с шагом (t-2), но изменилось с (t-1)
            if len >= 2 && sync_state == history[len - 2] && sync_state != history[len - 1] {
                is_oscillating = true;
                break;
            }
            history.push(sync_state.clone());
        }

        // Если нашли зацикливание в синхронном режиме — проверяем асинхронный на ТЕХ ЖЕ данных
        if is_oscillating {
            // --- 2. Проверяем АСИНХРОННЫЙ режим ---
            let mut async_state = initial_noisy.clone();
            let mut async_converged = false;

            for iter in 0..50 {
                let changed =
                    neuron_fix(&mut async_state, &weights, seed + iter as u64 + 3000, None);
                if changed == 0 {
                    async_converged = true;
                    break;
                }
            }

            // Главная проверка C3: Асинхронный режим сошелся к фиксированной точке
            assert!(
                async_converged,
                "Ошибка! Асинхронный режим должен был сойтись, а не зациклиться."
            );
            found_oscillation = true;
            let _ = writeln!(file, "{},{}", seed, found_oscillation);
            break;
        }
    }

    assert!(
        found_oscillation,
        "Не удалось найти случай с осцилляцией за 1000 сидов"
    );
}

#[test]
pub fn test_bit_vs_standart_equivalence() {
    use std::fs::File;
    use std::io::Write;

    let n = 1024;
    let p = 50;
    let num_runs = 20;
    let mut file =
        File::create("test_bit_vs_standart_equivalence.csv").expect("Не удалось создать CSV файл");
    let _ = writeln!(file, "seed,iter_number");
    for run in 0..num_runs {
        let run_seed = run as u64;

        // 1. Генерируем паттерны и зашумляем 1-й образ
        let states_f64 = generate_states(p, n, run_seed + 1000);
        let noisy_f64 = apply_noise(&states_f64[0], 0.10, run_seed + 2000);

        // --- БАЗОВЫЙ ДВИЖОК ---
        let weights = weight_matrix_calculate(&states_f64);
        let mut base_state = noisy_f64.clone();

        // --- БИТОВЫЙ ДВИЖОК ---
        let packed_patterns: Vec<Vec<u64>> = states_f64.iter().map(|p| pack_bits(p)).collect();
        let mut bit_state = pack_bits(&noisy_f64);
        let mut m_overlaps = calculate_initial_overlaps(&bit_state, &packed_patterns, n);

        // 2. Шаг за шагом сравниваем траектории обоих движков
        for iter in 0..20 {
            let _ = writeln!(file, "{},{}", run_seed, iter);
            let iter_seed = run_seed + 3000 + (iter as u64);

            let base_changed = neuron_fix(&mut base_state, &weights, iter_seed, None);
            let bit_changed = neuron_fix_bit(
                &mut bit_state,
                &packed_patterns,
                &mut m_overlaps,
                n,
                iter_seed,
                None,
            );

            // Количество изменившихся нейронов должно совпадать
            assert_eq!(
                base_changed, bit_changed,
                "Прогон {run}, Итерация {iter}: Не совпадает количество измененных нейронов!"
            );

            // Состояния должны быть строго идентичны бит-в-бит
            let unpacked_bit_state = unpack_bits(&bit_state, n);
            assert_eq!(
                base_state, unpacked_bit_state,
                "Прогон {run}, Итерация {iter}: Траектории базового и битового движков разошлись!"
            );

            // Если сеть сошлась к неподвижной точке — выходим
            if base_changed == 0 {
                break;
            }
        }
    }
}
