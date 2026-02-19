# benchmark_shear

Benchmark comparing JAX (GPU, brute-force) vs TreeCorr (CPU, kd-tree) for tangential shear estimation from a galaxy lensing catalog.

Lenses: `z ∈ [0.4, 0.6]` — Sources: `z > 0.6` — Shear: `γ1, γ2` (true shear from simulation)  
Hardware: NVIDIA A40 (48 GB) vs 32 CPU cores.

## Install

```bash
pip install -U "jax[cuda12]" treecorr astropy scipy matplotlib pandas pyarrow
```

## Run

```bash
python benchmark_shear.py --catalog path/to/catalog.parquet --plot
python benchmark_shear.py --catalog path/to/catalog.parquet --plot --quick   # fast test
```

## Results

Catalog: 100 deg² patch from the [Euclid Flagship Mock Galaxy Catalogue](https://cosmohub.pic.es/catalogs/353), tiled 3×3 to 900 deg² (~37M sources, ~10M lenses).  
Separation range: 0.5–100 arcmin (15 log-spaced bins).

### Quick run (r_max = 100 arcmin)

| N sources | N lenses |  TreeCorr (s) | JAX GPU (s) |
|----------:|---------:|--------------:|------------:|
|    10,000 |    2,768 |         0.035 |       0.284 |
|    50,000 |   13,840 |         0.078 |       0.459 |
|   100,000 |   27,680 |         0.131 |       0.640 |
|   300,000 |   83,041 |         0.381 |       2.796 |

### Full run (r_max = 100 arcmin)

| N sources  |  N lenses | TreeCorr (s) | JAX GPU (s) |
|-----------:|----------:|-------------:|------------:|
|    100,000 |    27,680 |        0.127 |       0.638 |
|    500,000 |   138,401 |        0.662 |       7.619 |
|  1,000,000 |   276,803 |        1.365 |     **OOM** |
|  3,000,000 |   830,410 |        4.480 |     **OOM** |
|  5,000,000 | 1,384,017 |        7.863 |     **OOM** |

Empirical scaling: TreeCorr **α ≈ 1.06** (O(N log N)). JAX **α ≈ 1.54** over the valid range, approaching O(N²) as N grows. JAX OOMs above ~500k sources on a 48 GB A40 due to the full N_lens × N_source distance matrix.

## Conclusion

TreeCorr wins for this workload. The kd-tree's O(N log N) scaling and aggressive pair pruning outperform brute-force GPU at all tested scales. JAX's perfectly vectorized arithmetic is negated by the O(N²) memory footprint.

JAX brute force is competitive only for small catalogs (N ≲ 50k) or small apertures (r_max ≲ 5 arcmin) where the candidate set per lens is small enough that GPU parallelism offsets the worse asymptotic scaling. A cell-list implementation would recover GPU competitiveness for larger N.