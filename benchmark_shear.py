"""
benchmark_shear.py
==================
Compare JAX brute-force tangential shear (batched) against TreeCorr's
NGCorrelation for subsamples of a real simulation catalog.

Lenses : true_redshift_gal in [0.4, 0.6]
Sources : true_redshift_gal > 0.6
Shear   : gamma1, gamma2 (true shear from simulation)

Usage
-----
    python benchmark_shear.py --catalog path/to/24307.parquet [--device cpu|gpu] [--plot] [--quick] [--r_max 100]
"""

import argparse
import time
import warnings
import gc

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import jax
import jax.numpy as jnp

try:
    import treecorr
    HAS_TREECORR = True
except ImportError:
    HAS_TREECORR = False
    warnings.warn("TreeCorr not found; skipping TreeCorr benchmark.")


# ── catalog loader ────────────────────────────────────────────────────────────

def load_catalog(path, z_lens_min=0.4, z_lens_max=0.6, z_src_min=0.6):
    print(f"Loading catalog from {path} ...")
    cat = Table.read(path)
    z   = np.array(cat['true_redshift_gal'], dtype=np.float32)

    lens_mask = (z >= z_lens_min) & (z <= z_lens_max)
    src_mask  = z > z_src_min

    def pos(mask):
        # flat-sky projection: correct ra for cos(dec)
        dec_rad = np.deg2rad(np.array(cat['dec_gal'][mask], dtype=np.float32))
        ra  = np.array(cat['ra_gal'][mask],  dtype=np.float32) * np.cos(dec_rad) * 60.0
        dec = np.array(cat['dec_gal'][mask], dtype=np.float32) * 60.0
        return np.stack([ra, dec], axis=1)

    full = dict(
        lens_pos   = pos(lens_mask),
        source_pos = pos(src_mask),
        gamma1     = np.array(cat['gamma1'][src_mask], dtype=np.float32),
        gamma2     = np.array(cat['gamma2'][src_mask], dtype=np.float32),
        z_lens     = z[lens_mask],
        z_source   = z[src_mask],
    )
    print(f"  N_lens={full['lens_pos'].shape[0]:,}  "
          f"N_source={full['source_pos'].shape[0]:,}")
    return full


def tile_catalog(full_cat, n_tiles_per_side=3):
    """Tile the catalog on a grid to increase total area."""
    field_size = float(np.ptp(full_cat['lens_pos'][:, 0]))

    tiles_lens, tiles_src, tiles_g1, tiles_g2 = [], [], [], []
    for ix in range(n_tiles_per_side):
        for iy in range(n_tiles_per_side):
            shift = np.array([ix * field_size, iy * field_size], dtype=np.float32)
            tiles_lens.append(full_cat['lens_pos']   + shift)
            tiles_src.append( full_cat['source_pos'] + shift)
            tiles_g1.append(  full_cat['gamma1'])
            tiles_g2.append(  full_cat['gamma2'])

    tiled = dict(
        lens_pos   = np.concatenate(tiles_lens, axis=0),
        source_pos = np.concatenate(tiles_src,  axis=0),
        gamma1     = np.concatenate(tiles_g1),
        gamma2     = np.concatenate(tiles_g2),
    )
    n = n_tiles_per_side
    print(f"Tiled {n}x{n}: N_lens={tiled['lens_pos'].shape[0]:,}  "
          f"N_source={tiled['source_pos'].shape[0]:,}  "
          f"field={n*field_size/60:.1f}x{n*field_size/60:.1f} deg²")
    return tiled


def subsample(full_cat, n_sources, rng=None):
    """Subsample sources (and lenses proportionally) from full_cat."""
    if rng is None:
        rng = np.random.default_rng(0)

    NS_full = full_cat['source_pos'].shape[0]
    NL_full = full_cat['lens_pos'].shape[0]
    frac    = min(n_sources / NS_full, 1.0)
    NS      = min(n_sources, NS_full)
    NL      = max(1, int(NL_full * frac))

    src_idx  = rng.choice(NS_full, size=NS,  replace=False)
    lens_idx = rng.choice(NL_full, size=NL,  replace=False)
    side     = float(np.ptp(full_cat['lens_pos'][:, 0]))

    return dict(
        lens_pos    = full_cat['lens_pos'][lens_idx],
        source_pos  = full_cat['source_pos'][src_idx],
        gamma1      = full_cat['gamma1'][src_idx],
        gamma2      = full_cat['gamma2'][src_idx],
        side_arcmin = side,
    )


# ── JAX implementation ────────────────────────────────────────────────────────

def make_jax_estimator(r_edges, batch_size=512):
    r_lo   = jnp.array(r_edges[:-1], dtype=jnp.float32)
    r_hi   = jnp.array(r_edges[1:],  dtype=jnp.float32)
    N_bins = len(r_edges) - 1

    @jax.jit
    def _single_lens(lens_xy, src_xy, g1, g2):
        d      = src_xy - lens_xy[None, :]
        r      = jnp.linalg.norm(d, axis=-1)
        phi    = jnp.arctan2(d[:, 1], d[:, 0])
        gt     = -(g1 * jnp.cos(2*phi) + g2 * jnp.sin(2*phi))
        in_bin = (r[:, None] >= r_lo[None, :]) & (r[:, None] < r_hi[None, :])
        return jnp.sum(in_bin * gt[:, None], axis=0), jnp.sum(in_bin, axis=0)

    def _run_batch(lens_batch, src_xy, g1, g2):
        return jax.vmap(_single_lens, in_axes=(0, None, None, None))(
            lens_batch, src_xy, g1, g2)

    def gamma_t_jax(cat):
        lens_xy = jnp.array(cat['lens_pos'])
        src_xy  = jnp.array(cat['source_pos'])
        g1      = jnp.array(cat['gamma1'])
        g2      = jnp.array(cat['gamma2'])
        NL      = lens_xy.shape[0]

        pad          = (-NL) % batch_size
        lens_pad     = jnp.concatenate(
            [lens_xy, jnp.zeros((pad, 2), dtype=jnp.float32)], axis=0)
        lens_batches = lens_pad.reshape(-1, batch_size, 2)

        gt_b, w_b = jax.lax.map(
            lambda lb: _run_batch(lb, src_xy, g1, g2),
            lens_batches
        )
        gt_all  = gt_b.reshape(-1, N_bins)[:NL]
        w_all   = w_b.reshape(-1,  N_bins)[:NL]
        w_total = w_all.sum(axis=0)
        gt_out  = jnp.where(w_total > 0, gt_all.sum(axis=0) / w_total, jnp.nan)
        return gt_out, w_total

    return gamma_t_jax


# ── TreeCorr implementation ───────────────────────────────────────────────────

def gamma_t_treecorr(cat, r_edges, n_patches=16):
    if not HAS_TREECORR:
        return None, None, None

    lens_cat = treecorr.Catalog(
        x=cat['lens_pos'][:, 0], y=cat['lens_pos'][:, 1],
        x_units='arcmin', y_units='arcmin',
        npatch=n_patches,
    )
    src_cat = treecorr.Catalog(
        x=cat['source_pos'][:, 0], y=cat['source_pos'][:, 1],
        g1=cat['gamma1'], g2=cat['gamma2'],
        x_units='arcmin', y_units='arcmin',
        patch_centers=lens_cat.patch_centers,
    )
    ng = treecorr.NGCorrelation(
        min_sep=r_edges[0], max_sep=r_edges[-1], nbins=len(r_edges)-1,
        sep_units='arcmin', var_method='jackknife',
    )
    ng.process(lens_cat, src_cat)
    return ng.xi, ng.rnom, ng.estimate_cov('jackknife')


# ── benchmarking ──────────────────────────────────────────────────────────────

def benchmark(full_cat, n_source_list, r_edges, n_repeats=3,
              batch_size=512, n_patches=16, warmup=True):

    results = {
        'N': [], 'N_lens': [],
        'jax_mean': [], 'jax_std': [], 'jax_gt': [], 'jax_r': [],
        'tc_mean':  [], 'tc_std':  [], 'tc_gt':  [], 'tc_r':  [],
    }

    estimator  = make_jax_estimator(r_edges, batch_size=batch_size)
    r_centers  = np.sqrt(r_edges[:-1] * r_edges[1:])
    nan_profile = np.full(len(r_edges)-1, np.nan)

    for N in n_source_list:
        print(f"\n── N_sources = {N:,} ──────────────────────")
        cat = subsample(full_cat, N, rng=np.random.default_rng(0))
        NL  = cat['lens_pos'].shape[0]
        print(f"   N_lens={NL:,}  N_sources={N:,}")

        # ── TreeCorr first (always attempted) ────────────────────────────────
        tc_mean = tc_std = np.nan
        tc_gt   = nan_profile.copy()
        tc_r    = r_centers
        if HAS_TREECORR:
            tc_times = []
            try:
                for i in range(n_repeats):
                    t0 = time.perf_counter()
                    tc_gt, tc_r, _ = gamma_t_treecorr(cat, r_edges, n_patches)
                    tc_times.append(time.perf_counter() - t0)
                tc_mean = np.mean(tc_times)
                tc_std  = np.std(tc_times)
                print(f"   TreeCorr : {tc_mean:.3f} ± {tc_std:.3f} s")
            except Exception as e:
                print(f"   TreeCorr FAILED: {e}")

        # ── JAX (may OOM at large N) ──────────────────────────────────────────
        jax_mean = jax_std = np.nan
        jax_gt   = nan_profile.copy()
        try:
            if warmup:
                print("   JAX warmup (JIT compile)...", end=' ', flush=True)
                _ = estimator(cat)
                jax.block_until_ready(_[0])
                print("done")

            jax_times = []
            for i in range(n_repeats):
                t0  = time.perf_counter()
                out = estimator(cat)
                jax.block_until_ready(out[0])
                jax_times.append(time.perf_counter() - t0)
            jax_mean = np.mean(jax_times)
            jax_std  = np.std(jax_times)
            jax_gt   = np.array(out[0])
            print(f"   JAX      : {jax_mean:.3f} ± {jax_std:.3f} s")
        except Exception as e:
            print(f"   JAX FAILED (likely OOM): {e}")

        results['N'].append(N)
        results['N_lens'].append(NL)
        results['jax_mean'].append(jax_mean)
        results['jax_std'].append(jax_std)
        results['jax_gt'].append(jax_gt)
        results['jax_r'].append(r_centers)
        results['tc_mean'].append(tc_mean)
        results['tc_std'].append(tc_std)
        results['tc_gt'].append(tc_gt)
        results['tc_r'].append(tc_r)

        # free GPU memory between steps
        jax.clear_caches()
        gc.collect()

    for k in ['N', 'N_lens', 'jax_mean', 'jax_std', 'tc_mean', 'tc_std']:
        results[k] = np.array(results[k])
    return results


def fit_power_law(N, t):
    mask = np.isfinite(t) & (t > 0)
    if mask.sum() < 2:
        return np.nan, np.nan
    coeffs = np.polyfit(np.log10(N[mask]), np.log10(t[mask]), 1)
    return coeffs[0], 10**coeffs[1]


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, r_edges, save_path='benchmark_results.png'):
    N      = results['N']
    N_fine = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── panel 1: timing ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title('Timing: JAX GPU vs TreeCorr (32 CPU cores)', fontsize=12)

    jax_valid = np.isfinite(results['jax_mean'])
    tc_valid  = np.isfinite(results['tc_mean'])

    if jax_valid.any():
        alpha_j, a_j = fit_power_law(N, results['jax_mean'])
        ax.errorbar(N[jax_valid], results['jax_mean'][jax_valid],
                    yerr=results['jax_std'][jax_valid],
                    fmt='o-', label=f'JAX GPU (α={alpha_j:.2f})', color='steelblue')
        ax.plot(N_fine, a_j * N_fine**alpha_j, '--', color='steelblue', alpha=0.4)
        # reference lines
        t0, n0 = results['jax_mean'][jax_valid][0], N[jax_valid][0]
        ax.plot(N_fine, t0*(N_fine/n0)**2, ':',  color='gray',      label='O(N²)')
        ax.plot(N_fine, t0*(N_fine/n0)**1, '-.', color='lightgray', label='O(N)')

    if tc_valid.any():
        alpha_t, a_t = fit_power_law(N, results['tc_mean'])
        ax.errorbar(N[tc_valid], results['tc_mean'][tc_valid],
                    yerr=results['tc_std'][tc_valid],
                    fmt='s-', label=f'TreeCorr (α={alpha_t:.2f})', color='tomato')
        ax.plot(N_fine, a_t * N_fine**alpha_t, '--', color='tomato', alpha=0.4)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N sources'); ax.set_ylabel('Time (s)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── panel 2: gamma_t(r) for largest successful N ──────────────────────────
    ax2 = axes[1]
    ax2.set_title('Tangential shear profile γ_t(r)', fontsize=12)
    ax2.axhline(0, color='k', lw=0.5, ls='--')

    # TreeCorr: largest N with valid signal
    for i in range(len(N)-1, -1, -1):
        gt = results['tc_gt'][i]
        r  = results['tc_r'][i]
        if gt is not None and np.any(np.isfinite(gt)):
            ax2.plot(r, gt, 's-', color='tomato',
                     label=f'TreeCorr  N={N[i]:,}', markersize=4)
            break

    # JAX: largest N with valid signal
    for i in range(len(N)-1, -1, -1):
        gt = results['jax_gt'][i]
        r  = results['jax_r'][i]
        if gt is not None and np.any(np.isfinite(gt)):
            ax2.plot(r, gt, 'o-', color='steelblue',
                     label=f'JAX GPU   N={N[i]:,}', markersize=4, alpha=0.8)
            break

    ax2.set_xscale('log')
    ax2.set_xlabel('r [arcmin]'); ax2.set_ylabel('γ_t')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True,
                        help='Path to parquet catalog (e.g. 24307.parquet)')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--plot',   action='store_true')
    parser.add_argument('--quick',  action='store_true',
                        help='Small N list for quick test')
    parser.add_argument('--r_max',  type=float, default=100.0,
                        help='Max separation in arcmin (default: 100)')
    args = parser.parse_args()

    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    import multiprocessing
    print(f"JAX devices     : {jax.devices()}")
    print(f"CPU cores       : {multiprocessing.cpu_count()}")
    print(f"TreeCorr        : {HAS_TREECORR}")
    print(f"r_max           : {args.r_max} arcmin")

    full_cat = load_catalog(args.catalog)
    full_cat = tile_catalog(full_cat, n_tiles_per_side=3)

    r_edges = np.logspace(np.log10(0.5), np.log10(args.r_max), 16)

    if args.quick:
        n_source_list = [10_000, 50_000, 100_000, 300_000]
        n_repeats = 2
    else:
        n_source_list = [100_000, 500_000, 1_000_000, 3_000_000, 5_000_000]
        n_repeats = 3

    results = benchmark(
        full_cat, n_source_list, r_edges,
        n_repeats=n_repeats, batch_size=512, n_patches=16,
    )

    print("\n── Summary ─────────────────────────────────────────────────────")
    alpha_j, _ = fit_power_law(results['N'], results['jax_mean'])
    print(f"JAX scaling exponent     : {alpha_j:.3f}  (expected 2.0 for O(N²))")
    if not np.all(np.isnan(results['tc_mean'])):
        alpha_t, _ = fit_power_law(results['N'], results['tc_mean'])
        print(f"TreeCorr scaling exponent: {alpha_t:.3f}  (expected ~1.0 for tree)")

    if args.plot:
        plot_results(results, r_edges)


if __name__ == '__main__':
    main()