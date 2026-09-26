use hopfield::*;

// Exact integer-valued dense matrix avoids artificial signs at zero field
// caused by summing normalized floating-point weights.
#[test]
fn packed_dynamics_matches_exact_dense_across_sizes_loads_and_masks() {
    for n in [63, 64, 65, 128, 256] {
        for alpha in [0.05_f64, 0.14, 0.20] {
            for seed in 0..5_u64 {
                let p = (alpha * n as f64).round() as usize;
                let patterns = generate_states(p, n, 1000 + seed);
                let mut w = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            w[i][j] = patterns.iter().map(|x| x[i] * x[j]).sum();
                        }
                    }
                }
                let packed: Vec<_> = patterns.iter().map(|x| pack_bits(x)).collect();
                for clamp in [false, true] {
                    let mask: Vec<bool> = (0..n).map(|i| clamp && i < n / 2).collect();
                    let mut dense = apply_noise(&patterns[0], 0.25, 2000 + seed);
                    let mut bits = pack_bits(&dense);
                    let mut q = calculate_initial_overlaps(&bits, &packed, n);
                    for sweep in 0..30 {
                        let energy = calculate_energy(&dense, &w);
                        let a = neuron_fix(&mut dense, &w, seed + sweep, Some(&mask));
                        let b = neuron_fix_bit(&mut bits, &packed, &mut q, n,
                                               seed + sweep, Some(&mask));
                        assert_eq!(a, b, "n={n} p={p} seed={seed} clamp={clamp}");
                        assert_eq!(dense, unpack_bits(&bits, n));
                        assert_eq!(q, calculate_initial_overlaps(&bits, &packed, n));
                        assert!(calculate_energy(&dense, &w) <= energy);
                        if a == 0 { break; }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_field_preserves_state() {
    let patterns = vec![pack_bits(&[1.0, 1.0]), pack_bits(&[1.0, -1.0])];
    let mut state = pack_bits(&[-1.0, 1.0]);
    let before = state.clone();
    let mut q = calculate_initial_overlaps(&state, &patterns, 2);
    assert_eq!(neuron_fix_bit(&mut state, &patterns, &mut q, 2, 0, None), 0);
    assert_eq!(state, before);
}
