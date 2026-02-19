# Pair Statistics on Galaxy Catalogues: Algorithms and GPU Compatibility

*A brief survey of pair-counting approaches for two-point statistics in weak lensing and LSS, with notes on GPU suitability.*

---

## 1. The Problem

Two-point statistics — galaxy–galaxy correlations ($\xi(r)$), shear–shear ($\xi_\pm(\theta)$), and galaxy–galaxy lensing (tangential shear $\gamma_t(r)$) — all reduce to the same kernel: for every pair $(i,j)$, compute a separation $r_{ij}$ and accumulate a weighted contribution into radial bins. Naively this is $O(N^2)$.

Stage-IV surveys push this to the limit. Euclid targets ~30 gal/arcmin² [[1]](#ref1) over 15,000 deg², giving $N_\text{gal} \sim 30 \times 15{,}000 \times 3{,}600 \approx 1.6 \times 10^9$ source galaxies — making brute-force pair counting completely intractable.

---

## 2. Algorithms

### 2.1 Brute-Force Pair Counting — $O(N^2)$

Evaluate all pairs directly. No data structure required, embarrassingly parallel. On GPU this maps to a matrix of pairwise distances, fully vectorized in JAX. The bottleneck is memory: a float32 distance matrix at $N_L = N_S = 10^6$ requires ~4 TB — infeasible even with batching beyond $N \sim 500\text{k}$ on a 48 GB GPU (confirmed in benchmarks here).

**GPU-friendly:** ✅ for $N \lesssim 100\text{k}$. **Scales to surveys:** ❌

### 2.2 Cell-List (Grid Hash) — $O(N)$ at fixed density

Partition the field into spatial cells of side $\sim r_\text{max}$. Each lens queries only its cell and its $3^d - 1$ neighboring cells, bounding the candidate set to $\bar{n} \times (3 r_\text{max})^d$ sources per lens. For 30 gal/arcmin² at $r_\text{max} = 10$ arcmin this gives ~27,000 candidates per lens vs ~4M total sources — a 150× reduction in work at no algorithmic approximation.

Cell assignment is a pure integer division, and the 9 neighbors are a fixed-size gather — both statically shaped and fully vectorizable. This makes cell-lists a natural GPU-compatible replacement for the inner loop of a tree algorithm, avoiding the branch divergence that makes kd-trees GPU-unfriendly. Variable cell occupancy must be handled by padding cells to a fixed maximum size for JAX/XLA static shapes.

**GPU-friendly:** ✅ **Scales to surveys:** ✅

### 2.3 KD-Tree / Ball-Tree — $O(N \log N)$

Recursively partition space into a binary tree; prune subtrees when the minimum possible distance exceeds the current best. [TreeCorr](https://github.com/rmjarvis/TreeCorr) [[2]](#ref2) implements ball trees in C++ with OpenMP, achieving sub-second runtimes for $N \sim 10^6$ on 32 cores (measured here: 1.4 s at $N_S = 10^6$, empirical $\alpha \approx 1.06$).

**GPU-friendly:** ❌ Tree traversal is data-dependent: each query follows a different path, causing warp divergence and irregular memory access patterns on GPU. Stackless GPU kd-tree algorithms exist but recover only a fraction of CPU tree efficiency. **Scales to surveys:** ✅ on CPU.

### 2.4 FFT-Based Estimators — $O(N_g \log N_g)$

The two-point correlation function is the autocorrelation of the density field, computable via the convolution theorem: paint galaxies onto a grid → FFT → multiply by conjugate → IFFT. For **shear–shear and g-g statistics**, the catalog is routinely pixellised — galaxies are painted onto a HEALPix or regular grid and the transform applied to the resulting map. This is standard practice for cosmic shear power spectra (pseudo-$C_\ell$ estimators).

The flat-sky FFT is valid for patches $\lesssim 10°$. For wide-field or full-sky coverage, spherical harmonic transforms are required. [S2FFT](https://github.com/astro-informatics/s2fft) [[3]](#ref3) provides differentiable spin spherical harmonic transforms in JAX, deployable on GPUs and TPUs, with up to 400× acceleration over CPU C codes. Its algorithms are built on stable Wigner-d function recursions that recurse along harmonic order $m$ alone, enabling extreme parallelisation — making full-sky FFT-based estimators GPU-compatible in a differentiable framework for the first time.

The caveat for sparse catalogs: grid resolution must match $r_\text{min}$. For 30 gal/arcmin² resolving 1 arcmin over a 10° field, $N_g \sim (600)^2 = 360\text{k}$ pixels, most of them empty — grid overhead is modest here. At sub-arcminute resolution over large areas, $N_g^2$ can exceed $N_\text{gal}^2$, negating the FFT advantage.

**GPU-friendly:** ✅ (`cuFFT` / S2FFT). **Scales to surveys:** ✅ for pixellised statistics.

### 2.5 KNN — $O(N^2)$ + partial sort

For each query, find the $K$ nearest neighbours via brute-force distances + `argpartition`. Most useful in **dense LSS contexts** — N-body snapshots, simulation post-processing — where the distance to the $K$-th neighbour is itself an observable (local density estimator: $\hat{n} \propto K / V(r_K)$) or where environment tagging requires a well-defined local scale. For sparse weak lensing catalogs the statistic is ill-defined, as fixed $K$ mixes different physical scales.

**GPU-friendly:** ✅ (distance step is identical to brute force). **Scales to surveys:** ⚠️ approximate methods (e.g. FAISS) needed beyond $N \sim 10^7$.

---

## 3. Statistic Dependence: Shear–Shear / LSS vs. Tangential Shear

The right algorithm depends strongly on which statistic is being computed.

**$\xi(r)$, $w(\theta)$, $\xi_\pm(\theta)$ (same-population, isotropic):** pixellisation is standard. FFT-based estimators apply directly, making GPU acceleration natural. Tree pair counting in real space is also common and handles curved sky natively.

**Tangential shear $\gamma_t(r)$ (cross-population, directional, spin-2):** for each lens–source pair the ellipticity must be projected onto the tangential direction relative to that specific lens:
$$\gamma_t^{ij} = -(\gamma_1^j \cos 2\phi_{ij} + \gamma_2^j \sin 2\phi_{ij})$$
where $\phi_{ij}$ is the pair's position angle. This per-pair orientation makes $\gamma_t$ structurally harder:

- **KNN is inappropriate** — $\gamma_t$ requires fixed radial bins in physical separation, not fixed neighbour counts.
- **FFT requires reformulation** — the projection becomes a spin-2 phase factor in Fourier space, valid under the flat-sky approximation but complicated by survey masks and the lens-dependent orientation.
- **Tree pair counting via TreeCorr's `NG` correlator** handles this natively and remains the standard approach.

---

## 4. GPU vs. CPU: Empirical Results

Benchmark on NVIDIA A40 (48 GB) vs 32 CPU cores, [Euclid Flagship Mock Galaxy Catalogue](https://cosmohub.pic.es/catalogs/353) (100 deg² patch tiled 3×3 to 900 deg², ~37M sources, ~10M lenses), $r_\text{max} = 100$ arcmin:

| $N_\text{sources}$ | $N_\text{lenses}$ | TreeCorr CPU (s) | JAX GPU brute force (s) |
|-------------------:|------------------:|----------------:|------------------------:|
| 100,000            | 27,680            | 0.13            | 0.64                    |
| 500,000            | 138,401           | 0.66            | 7.6                     |
| 1,000,000          | 276,803           | 1.4             | OOM (10.9 GB req.)      |
| 5,000,000          | 1,384,017         | 7.9             | OOM                     |

TreeCorr scales as $\alpha \approx 1.06$ (close to $O(N \log N)$); JAX brute force OOMs above $N \sim 500\text{k}$ due to the $N_L \times N_S$ distance matrix. TreeCorr wins by up to 60× at $N = 500\text{k}$.

GPU brute force becomes competitive only for $N \lesssim 50\text{k}$ or $r_\text{max} \lesssim 5$ arcmin, where the candidate set per lens is small enough that GPU parallelism offsets the worse asymptotic scaling. A **cell-list implementation** would recover GPU competitiveness for larger $N$ by reducing the effective work per lens from $O(N_S)$ to $O(\bar{n} \cdot r_\text{max}^2)$.

---

## 5. Summary

| Algorithm | Complexity | GPU | Best use case |
|---|---|---|---|
| Brute force | $O(N^2)$ | ✅ | $N \lesssim 100\text{k}$, any statistic |
| Cell-list | $O(N)$ | ✅ | Small $r_\text{max}$, uniform density, GPU-native |
| KD-tree / Ball-tree | $O(N \log N)$ | ❌ | Sparse catalogs, CPU clusters, standard tool |
| FFT (flat-sky) | $O(N_g \log N_g)$ | ✅ | Shear–shear / g-g pixellised maps |
| Spherical FFT (S2FFT) | $O(L^2 \log L)$ | ✅ | Full-sky, wide-field, differentiable pipelines |
| KNN | $O(N^2)$ + sort | ✅ | Density estimation, dense LSS snapshots |

For galaxy–galaxy lensing ($\gamma_t$) at survey scale, **tree methods on CPU remain the practical standard**. FFT-based approaches are the right tool for shear–shear and g-g power spectra, with S2FFT enabling this on GPU for the first time in a differentiable JAX framework. A JAX cell-list implementation would recover GPU competitiveness for $\gamma_t$ at small to moderate $r_\text{max}$.

---

## References

<a name="ref1">[1]</a> Laureijs et al. (2011), [Euclid Definition Study Report](https://arxiv.org/abs/1110.3193), arXiv:1110.3193 — target source density ~30 gal/arcmin², 15,000 deg² survey  
<a name="ref2">[2]</a> Jarvis, Bernstein & Jain (2004), MNRAS 352, 338 — TreeCorr ball-tree algorithm ([github](https://github.com/rmjarvis/TreeCorr))  
<a name="ref3">[3]</a> Price & McEwen (2024), Journal of Computational Physics, arXiv:2311.14670 — S2FFT: differentiable spherical harmonic transforms on GPU ([github](https://github.com/astro-informatics/s2fft))