//! Reproducible audit using the original packed update routine.
use hopfield::*;
use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{fs, io::{self, Write, BufWriter}, path::Path, time::Instant};

fn writer(dir: &Path, name: &str, header: &str) -> io::Result<BufWriter<fs::File>> {
    let mut f = BufWriter::new(fs::File::create(dir.join(name))?);
    writeln!(f, "{header}")?;
    Ok(f)
}

fn capacity(dir: &Path) -> io::Result<()> {
    let mut out = writer(dir, "capacity.csv", "n,p,seed,noise,alpha,one_overlap,final_overlap,sweeps,converged,seconds")?;
    let mut trace = writer(dir, "capacity_trace.csv", "n,p,seed,noise,sweep,overlap,changes")?;
    for n in [256usize, 1024, 4096] {
        for alpha in [0.10_f64, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22] {
            let p = (alpha * n as f64).round() as usize;
            for seed in 0..50_u64 {
                let patterns = generate_states(p, n, 1000 + seed);
                let target = patterns[0].clone();
                let packed: Vec<_> = patterns.iter().map(|x| pack_bits(x)).collect();
                drop(patterns);
                for noise in [0.05, 0.20] {
                    let mut state = pack_bits(&apply_noise(&target, noise, 2000 + seed));
                    let mut q = calculate_initial_overlaps(&state, &packed, n);
                    let start = Instant::now();
                    let (mut one, mut last, mut sweeps, mut converged) = (0.0, 0.0, 0, false);
                    for sweep in 0..300_u64 {
                        let changes = neuron_fix_bit(&mut state, &packed, &mut q, n, seed + sweep, None);
                        last = calculate_overlap(&unpack_bits(&state, n), &target);
                        if sweep == 0 { one = last; }
                        sweeps = sweep + 1;
                        writeln!(trace, "{n},{p},{seed},{noise},{sweeps},{last},{changes}")?;
                        if changes == 0 { converged = true; break; }
                    }
                    writeln!(out, "{n},{p},{seed},{noise},{},{one},{last},{sweeps},{converged},{}",
                             p as f64 / n as f64, start.elapsed().as_secs_f64())?;
                }
            }
            out.flush()?; trace.flush()?;
            println!("capacity N={n} alpha={alpha} complete");
        }
    }
    Ok(())
}

fn dense_recall(w: &DMatrix<f64>, initial: &[f64], mask: &[bool], seed: u64)
    -> (Vec<f64>, usize, bool) {
    let n = initial.len();
    let mut s = DVector::from_column_slice(initial);
    let mut fields = w * &s;
    let mut free: Vec<_> = (0..n).filter(|&i| !mask[i]).collect();
    for sweep in 0..100 {
        // Reset list so a seed defines the same order for every rule.
        free.sort_unstable();
        free.shuffle(&mut StdRng::seed_from_u64(seed + sweep as u64));
        let mut changed = 0;
        for &i in &free {
            let new = if fields[i] > 1e-10 { 1.0 }
                      else if fields[i] < -1e-10 { -1.0 } else { s[i] };
            let delta = new - s[i];
            if delta != 0.0 {
                s[i] = new;
                fields += w.column(i) * delta;
                changed += 1;
            }
        }
        // Limit accumulated numerical drift before checking convergence.
        fields = w * &s;
        if changed == 0 { return (s.as_slice().to_vec(), sweep + 1, true); }
    }
    (s.as_slice().to_vec(), 100, false)
}

fn mnist(dir: &Path, data_path: &str, confirm: bool) -> io::Result<()> {
    let data = fs::read(data_path)?;
    assert!(data.len() >= 16);
    let be = |i| u32::from_be_bytes(data[i..i+4].try_into().unwrap()) as usize;
    assert_eq!(be(0), 2051); assert_eq!(be(8), 28); assert_eq!(be(12), 28);
    let count = be(4); let n = 784;
    assert_eq!(data.len(), 16 + count * n);
    let mut out = writer(dir, "mnist.csv", "seed,k,rule,mask,image_idx,source_idx,rank,overlap,hidden_accuracy,exact_hidden,sweeps,converged")?;
    let seeds = if confirm { 10..30_u64 } else { 0..10_u64 };
    let sizes = if confirm { vec![100, 200] } else { vec![10, 50, 100] };
    for seed in seeds {
        let mut indices: Vec<_> = (0..count).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(10_000 + seed));
        for &k in &sizes {
            let x = DMatrix::from_fn(n, k, |i, mu|
                if data[16 + indices[mu] * n + i] > 127 { 1.0 } else { -1.0 });
            let gram = x.transpose() * &x;
            let svd = gram.svd(true, true);
            let tol = svd.singular_values.max() * 1e-12;
            let rank = svd.singular_values.iter().filter(|&&v| v > tol).count();
            let inv = svd.pseudo_inverse(tol).expect("SVD factors available");
            let projection = &x * inv * x.transpose();
            let mut proj_zero = projection.clone();
            let mut proj_half = projection.clone();
            let mut hebb = &x * x.transpose() / n as f64;
            for i in 0..n {
                proj_zero[(i,i)] = 0.0; proj_half[(i,i)] *= 0.5; hebb[(i,i)] = 0.0;
            }
            let rules = if confirm {
                vec![("projection_zero", &proj_zero), ("projection_half", &proj_half), ("projection_diag", &projection)]
            } else {
                vec![("hebb", &hebb), ("projection_zero", &proj_zero), ("projection_diag", &projection)]
            };
            for (rule, w) in rules {
                for mask_kind in ["lower_half", "random_half"] {
                    for mu in 0..k {
                        let trial_seed = 100_000 + seed * 1000 + mu as u64;
                        let mut rng = StdRng::seed_from_u64(trial_seed);
                        let mut order: Vec<_> = (0..n).collect();
                        if mask_kind == "random_half" { order.shuffle(&mut rng); }
                        let mut mask = vec![false; n];
                        for &i in &order[..n/2] { mask[i] = true; }
                        let target: Vec<_> = x.column(mu).iter().copied().collect();
                        let mut initial = target.clone();
                        for i in 0..n {
                            if !mask[i] { initial[i] = if rng.gen_bool(0.5) {1.0} else {-1.0}; }
                        }
                        let (restored, sweeps, converged) = dense_recall(w, &initial, &mask, trial_seed);
                        let errors = (0..n).filter(|&i| !mask[i] && restored[i] != target[i]).count();
                        let overlap = calculate_overlap(&restored, &target);
                        let accuracy = 1.0 - errors as f64 / (n/2) as f64;
                        writeln!(out, "{seed},{k},{rule},{mask_kind},{mu},{},{rank},{overlap},{accuracy},{},{sweeps},{converged}",
                                 indices[mu], errors == 0)?;
                    }
                }
            }
            out.flush()?;
            println!("MNIST seed={seed} k={k} complete");
        }
    }
    Ok(())
}

fn benchmark(dir: &Path) -> io::Result<()> {
    let mut out = writer(dir, "benchmark.csv", "n,p,seed,mode,engine,seconds,sweeps,converged,overlap")?;
    for n in [512usize, 1024, 2048] {
        for (mode, p) in [("fixed_p", 100), ("fixed_alpha", (0.14*n as f64).round() as usize)] {
            for seed in 0..10_u64 {
                let patterns = generate_states(p, n, 1000 + seed);
                let noisy = apply_noise(&patterns[0], 0.05, 2000 + seed);
                let weights = weight_matrix_calculate(&patterns);
                let packed: Vec<_> = patterns.iter().map(|x| pack_bits(x)).collect();
                // Alternate order to reduce systematic warmup/order bias.
                for engine in if seed % 2 == 0 { ["dense", "packed"] } else { ["packed", "dense"] } {
                    let start = Instant::now();
                    let mut dense = noisy.clone();
                    let mut bits = pack_bits(&noisy);
                    let mut q = if engine == "packed" { calculate_initial_overlaps(&bits, &packed, n) } else { vec![] };
                    let (mut sweeps, mut converged) = (0, false);
                    for sweep in 0..300_u64 {
                        let changed = if engine == "dense" { neuron_fix(&mut dense, &weights, seed+sweep, None) }
                            else { neuron_fix_bit(&mut bits, &packed, &mut q, n, seed+sweep, None) };
                        sweeps = sweep + 1;
                        if changed == 0 { converged = true; break; }
                    }
                    let seconds = start.elapsed().as_secs_f64();
                    if engine == "packed" { dense = unpack_bits(&bits, n); }
                    let overlap = calculate_overlap(&dense, &patterns[0]);
                    writeln!(out, "{n},{p},{seed},{mode},{engine},{seconds},{sweeps},{converged},{overlap}")?;
                }
            }
            out.flush()?;
            println!("benchmark N={n} {mode} complete");
        }
    }
    Ok(())
}

fn nearest_baseline(dir: &Path, data_path: &str) -> io::Result<()> {
    let data = fs::read(data_path)?;
    let be = |i| u32::from_be_bytes(data[i..i+4].try_into().unwrap()) as usize;
    assert_eq!(be(0), 2051); assert_eq!(be(8),28); assert_eq!(be(12),28);
    let count = be(4); let n = 784;
    assert_eq!(data.len(),16+count*n);
    let mut out = writer(dir,"nearest.csv","seed,k,mask,image_idx,source_idx,cue_matches,overlap,hidden_accuracy,exact_hidden")?;
    for seed in 10..30_u64 {
        let mut indices: Vec<_> = (0..count).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(10_000+seed));
        for k in [100,200] {
            let patterns: Vec<Vec<f64>> = indices[..k].iter().map(|&idx|
                data[16+idx*n..16+(idx+1)*n].iter().map(|&v|if v>127 {1.0}else{-1.0}).collect()).collect();
            for mask_kind in ["lower_half","random_half"] {
                for mu in 0..k {
                    let mut rng = StdRng::seed_from_u64(100_000+seed*1000+mu as u64);
                    let mut order: Vec<_> = (0..n).collect();
                    if mask_kind=="random_half" { order.shuffle(&mut rng); }
                    // A noiseless visible cue contains the stored target, so
                    // nearest visible Hamming distance is zero. Break ties by
                    // storage order, without access to hidden target pixels.
                    let matches: Vec<_> = (0..k).filter(|&j|
                        order[..n/2].iter().all(|&i|patterns[j][i]==patterns[mu][i])).collect();
                    let restored = &patterns[matches[0]];
                    let errors = order[n/2..].iter().filter(|&&i|restored[i]!=patterns[mu][i]).count();
                    let overlap = calculate_overlap(restored,&patterns[mu]);
                    let accuracy = 1.0-errors as f64/(n/2) as f64;
                    writeln!(out,"{seed},{k},{mask_kind},{mu},{},{},{overlap},{accuracy},{}",indices[mu],matches.len(),errors==0)?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 { panic!("usage: research <capacity|mnist|mnist-confirm|mnist-baseline|benchmark> <output-directory> [MNIST IDX file]"); }
    let dir = Path::new(&args[2]); fs::create_dir_all(dir)?;
    match args[1].as_str() {
        "capacity" => capacity(dir),
        "mnist" => mnist(dir, args.get(3).expect("MNIST IDX path required"), false),
        "mnist-confirm" => mnist(dir, args.get(3).expect("MNIST IDX path required"), true),
        "mnist-baseline" => nearest_baseline(dir, args.get(3).expect("MNIST IDX path required")),
        "benchmark" => benchmark(dir),
        _ => panic!("unknown experiment"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_recall_preserves_clamped_pixels_and_zero_field() {
        let w = DMatrix::zeros(4,4);
        let initial = vec![1.0,-1.0,1.0,-1.0];
        let (state, steps, converged) = dense_recall(&w, &initial, &[true,false,true,false], 0);
        assert_eq!(state,initial); assert_eq!(steps,1); assert!(converged);
    }
    #[test]
    fn duplicate_patterns_use_finite_projection() {
        let x = DMatrix::from_row_slice(3,2,&[1.0,1.0,-1.0,-1.0,1.0,1.0]);
        let gram = x.transpose()*&x;
        let inv = gram.svd(true,true).pseudo_inverse(1e-10).unwrap();
        let w = &x*inv*x.transpose();
        assert!((&w*&x-&x).norm()<1e-10);
        assert!((&w*&w-&w).norm()<1e-10);
    }
    #[test]
    fn incremental_dense_matches_direct_fields() {
        let patterns = generate_states(20, 128, 123);
        let weights = weight_matrix_calculate(&patterns);
        let w = DMatrix::from_fn(128,128,|i,j|weights[i][j]);
        let initial = apply_noise(&patterns[0],0.2,456);
        let mask: Vec<_> = (0..128).map(|i|i<64).collect();
        let (fast,_,_) = dense_recall(&w,&initial,&mask,789);
        let mut direct = initial.clone();
        for step in 0..100 {
            if neuron_fix(&mut direct,&weights,789+step,Some(&mask)) == 0 { break; }
        }
        assert_eq!(fast,direct);
        for i in 0..64 { assert_eq!(fast[i],initial[i]); }
    }
}
