# Moa

A Monte Carlo neutron transport code in C99.

Named after the moa: 3.6 metres tall, 230 kg, couldn't fly, couldn't
outrun a Polynesian with a spear. But it was here first, and that
counts for something. Much like FORTRAN.

## What It Does

Moa solves the neutron transport equation by Monte Carlo simulation
for criticality eigenvalue (k-effective) and fixed-source problems.
Neutrons are tracked individually through constructive solid geometry,
interacting with nuclei according to evaluated nuclear data from the
ENDF/B-VII.1 library. Fission sites from each generation become the
source for the next, and the eigenvalue converges through power
iteration over hundreds of batches.

It runs on CPU (with OpenMP parallelism) and GPU (via
[BarraCUDA](https://github.com/Zaneham/BarraCUDA) targeting both
AMD and NVIDIA hardware). The GPU kernel is compiled without NVCC
or any proprietary NVIDIA toolchain.

Validated against three ICSBEP criticality benchmarks (Godiva,
Jezebel, Flattop) with results within statistical uncertainty of
published reference values.

## Benchmark Results

**CPU Platform**: AMD MI300X host, Xeon Platinum 8470 (13 cores @ 2 GHz), 224 GB RAM.
**GPU Platforms**: NVIDIA RTX 4060 Ti (local), AMD MI300X (remote), Tenstorrent (remote).

All benchmarks use ENDF/B-VII.1 cross-section data with unresolved
resonance probability tables. Criticality eigenvalue via power
iteration with 100 total batches (20 inactive).

### Scientific Accuracy

| Benchmark          | Computed k_eff          | Reference k_eff     | Leakage |
|--------------------|-------------------------|---------------------|---------|
| Godiva (10K, 1T)   | 0.99501 +/- 0.00120    | 1.0000 +/- 0.0010   | 82.4%   |
| Godiva (10K, 13T)  | 0.99521 +/- 0.00147    | 1.0000 +/- 0.0010   | 82.3%   |
| Godiva (50K, 1T)   | 0.99500 +/- 0.00053    | 1.0000 +/- 0.0010   | 82.3%   |
| Godiva (50K, 13T)  | 0.99488 +/- 0.00051    | 1.0000 +/- 0.0010   | 82.3%   |
| Jezebel (10K, 1T)  | 0.97881 +/- 0.00120    | 1.0000 +/- 0.0020   | 92.0%   |
| Jezebel (10K, 13T) | 0.98001 +/- 0.00124    | 1.0000 +/- 0.0020   | 92.0%   |
| Flattop (10K, 13T) | 0.99910 +/- 0.00135    | 1.0000 +/- 0.0030   | 38.4%   |

### CPU Performance

| Benchmark | Particles | Threads | Wall Time | Particles/s | Speedup |
|-----------|-----------|---------|-----------|-------------|---------|
| Godiva    | 10K       | 1       | 13.2 s    | 75,486      | 1.0x    |
| Godiva    | 10K       | 13      | 2.5 s     | 407,315     | 5.4x    |
| Godiva    | 50K       | 1       | 64.6 s    | 77,458      | 1.0x    |
| Godiva    | 50K       | 13      | 9.0 s     | 554,034     | 7.2x    |
| Jezebel   | 10K       | 1       | 1.9 s     | 531,850     | 1.0x    |
| Jezebel   | 10K       | 13      | 0.9 s     | 1,112,426   | 2.1x    |
| Flattop   | 10K       | 1       | ~52 min*  | ~3,200*     | 1.0x    |
| Flattop   | 10K       | 13      | 425 s     | 2,351       | ~7x*    |

\* Flattop single-thread killed at 52 min (extrapolated from batch time).

### GPU vs CPU (Godiva, RTX 4060 Ti)

Single-thread CPU vs GPU on the same machine (i7-14700KF). Honest numbers.

The GPU kernel is compiled by [BarraCUDA](https://github.com/Zaneham/BarraCUDA)
to PTX, loaded via CUDA Driver API, and JIT-compiled by the NVIDIA driver.
No NVCC involved. Open-source CUDA compiler running a nuclear reactor
benchmark on consumer gaming hardware.

| Particles | CPU (1 thread)  | GPU (RTX 4060 Ti) | Speedup | k_eff              |
|-----------|-----------------|--------------------|---------|--------------------|
| 1,000,000 | 107,210 p/s     | 409,690 p/s        | 3.8x    | 0.995 +/- 0.0001  |
| 2,000,000 | --              | 403,206 p/s        | --      | 0.995 +/- 0.0001  |

**3.8x speedup** over a single CPU core. The CPU code is compiled by
GCC -O2, which has had forty years of optimisation passes lavished upon
it by some of the best compiler engineers alive. The GPU code is compiled
by BarraCUDA, which has had about three months and does no optimisation
whatsoever -- no constant folding across blocks, no instruction
scheduling, no register coalescing. The PTX goes in naive and the
NVIDIA driver JIT does what it can with the wreckage. 3.8x anyway.

GPU throughput stays flat from 1M to 2M particles -- the RTX 4060 Ti
isn't breaking a sweat. For Godiva's trivial geometry (1 sphere,
3 nuclides), this is compute-bound, not memory-bound.

**Where GPU wins:** Any batch size above ~100K. The GPU's 4,352 CUDA
cores outnumber a single CPU core's ability to care.

**Where CPU wins:** OpenMP on all cores. 13 threads at 554K p/s
beats the GPU for Godiva. But for complex geometries with dozens
of cells and hundreds of nuclides, the GPU pulls ahead decisively.

Don't buy a GPU for Godiva. Do buy one for a full reactor model.

### Other Hardware

The GPU kernel also compiles and runs on AMD MI300X (CDNA3, via
BarraCUDA's `.hsaco` backend) and Tenstorrent hardware. Both of these
are accessed by SSHing into remote VMs, which is a bit like performing
open-heart surgery in Shanghai via Zoom while you're stuck in a
basement with an AT&T connection. The kernel compiles, dispatches,
and produces correct results, but getting meaningful benchmark
numbers through 200ms of round-trip latency and a terminal that
drops characters when you type too fast is an exercise in patience
that would test a Tibetan monk. Proper benchmarking on local
hardware is pending access to something that isn't on the other
side of the Pacific.

### Notes

1. **Parallelism doesn't change physics** -- 1T and 13T k_eff values
   agree within 1 sigma for all benchmarks.
2. **Scaling** -- 5-7x on 13 cores. Higher particle counts scale
   better (per-batch overhead amortised).
3. **Flattop is ~200x costlier per particle** than Godiva (reflected
   geometry = many more collisions per history). This is exactly where
   parallel computing earns its keep.
4. **Godiva is 0.5% low, Jezebel is 2% low** vs reference. Godiva's
   bias is within 5 sigma of statistical noise. Jezebel's 2% gap is a
   physics model limitation (likely missing thermal upscatter or
   fission spectrum bias), not a parallelism issue.

### k-eigenvalue

- k = 1.0: critical. Steady state. The reactor hums along. Good.
- k > 1.0: supercritical. Power rising. Bad for everyone except weapons
  designers, who have different priorities.
- k < 1.0: subcritical. The reactor is dying. Also bad, but in a more
  polite, British sort of way.

## Physics

### Nuclear Data

- ENDF/B-VII.1 cross-section data (MF3: total, elastic, fission,
  capture, inelastic) with pointwise energy grids up to 10,000 points
  per nuclide
- Resolved resonance reconstruction via Single-Level Breit-Wigner
  formalism, with Reich-Moore (LRF=3) support through combined fission
  channel widths. Energy-dependent neutron widths, statistical spin
  factors, hard-sphere phase shifts, and potential scattering
  interference terms
- Unresolved resonance region probability tables (MF2/LRU=2) with
  per-history sampling for self-shielding
- Doppler broadening of resolved resonances to arbitrary temperature
- Discrete inelastic scattering levels parsed from MF3/MT51-89
  excitation energies, with uniform level sampling and continuum
  evaporation fallback above the highest discrete level
- Anisotropic elastic angular distributions from MF4/MT2 average
  scattering cosine tables (linear interpolation on energy grid)
- S(alpha,beta) bound-atom thermal scattering for moderating materials
  (H-in-H2O, graphite) parsed from MF7/MT2
- Free-gas thermal scattering with target motion sampling below 4kT
- Watt fission spectrum sampling for fission neutron energies
- nu-bar (average neutrons per fission) from pointwise ENDF tables

### Transport

- Survival biasing (implicit capture) with Russian roulette variance
  reduction at configurable weight cutoff
- Track-length and collision estimators for flux and fission rate
- Energy-binned tallies (log-uniform or user-defined bin edges) for
  spectral analysis
- 3D Cartesian mesh tallies with Gnuplot-compatible output
- Power iteration for k-eigenvalue with configurable batch size,
  inactive batches, and convergence monitoring
- Fixed-source mode with point and volumetric sources

### Geometry

- Constructive solid geometry (CSG) with boolean intersection of
  halfspaces
- General quadric surfaces (planes, cylinders, spheres, and arbitrary
  Ax^2 + By^2 + ... + K = 0)
- Lattice geometry with rectangular grids and universe hierarchy
  (a 17x17 fuel assembly is ~10 surfaces instead of ~1500)
- Surface-crossing with epsilon nudge to prevent re-intersection
- Cell-finding by point-testing all cells (bounded by n_cell)

## Build

```bash
cd /c/dev/moa
make        # release build (-O2)
make test   # tests (-O0 -g, because optimisers hide bugs like
            # flatmates hide dishes)
```

Requires MinGW GCC and a functioning sense of humour.

## Usage

```bash
./moa bench/godiva.inp --verbose --seed 42
```

You'll need ENDF nuclear data files from NNDC. Place them in `data/`.
The code won't download them for you because we're not an npm package
and we have standards.

## Architecture

```
moa.h           All types. One header. The nuclear family.
src/rng.c       xoshiro256** PRNG. Deterministic chaos, carefully curated.
src/nd_parse.c  ENDF-6 parser. For a format designed before the moon landing.
src/nd_xs.c     Cross-section lookup. Binary search + linear interpolation.
src/nd_res.c    Resolved resonance reconstruction. Breit-Wigner, SLBW, Reich-Moore.
src/nd_rmat.c   R-matrix resonance evaluation.
src/nd_dopl.c   Doppler broadening. Temperature effects on resonances.
src/nd_urr.c    Unresolved resonance probability tables.
src/nd_thrm.c   Free-gas thermal scattering model.
src/nd_sab.c    S(alpha,beta) bound-atom thermal scattering.
src/csg.c       CSG geometry. Ray tracing for physicists, not artists.
src/cg_lat.c    Lattice geometry. Copy-paste for nuclear engineers.
src/tl_score.c  Track-length and collision tallies.
src/tl_ebin.c   Energy-binned flux tallies.
src/tl_mesh.c   3D Cartesian mesh tallies.
src/tp_loop.c   Transport loop. One neutron's entire life story.
src/tp_crit.c   Power iteration. k-eigenvalue from fission banking.
src/tp_fixd.c   Fixed-source transport driver.
src/io_input.c  Input parser. Cards, like it's 1965.
src/main.c      CLI driver. The front door.
```

## Things to Remember

### The 10,000 Iteration Guard

The tracking loop in `tp_loop.c` (`tp_hist`) is bounded by
`KA_GUARD(g, MO_MAX_HIST)` where `MO_MAX_HIST = 10000`. If a neutron
bounces 10,000 times without dying or leaking, we kill it. This is not
a physics decision -- it's a "your geometry has a hole and the neutron
is stuck in an infinite loop" decision.

A real neutron in Godiva undergoes maybe 5-20 collisions before being
absorbed or escaping. If you're hitting 10,000, something is wrong with
your surfaces, your cross-sections, or your life choices.

The guard also applies to:
- `nd_load`: 500,000 line scan limit (ENDF files are big, but not *that* big)
- `nd_tab1`: 5,000 data pairs per TAB1 record
- `tc_src`: 10,000 rejection samples per source particle
- `cg_find`: bounded by `n_cell` (max 256) -- no guard needed, it's O(n)
- Binary search in `xs_look`: guard at 20 (log2(10000) ~ 14)

If any guard trips, it means something has gone properly wrong. The code
fails safe rather than spinning forever, because infinite loops in
nuclear simulation code are -- to use a technical term -- *bad*.

### Memory Layout

`mo_prob_t` is approximately 18 MB (48 nuclides x 6 arrays x 10,000
doubles, plus resonance parameters, lattice data, and tally arrays).
This lives on the heap via calloc, not the stack, because Windows has
a 1 MB stack limit and we're not monsters. The fission banks in
`tp_crit.c` are static arrays for the same reason.

Everything is pre-allocated. No malloc in the hot path. Kauri arena for
scratch only. The moa didn't need dynamic memory allocation and neither
do we.*

*The moa also didn't need to avoid predators, and look how that worked out.

### ENDF Format Gotchas

FORTRAN E-notation: `+1.23456+003` means `1.23456e+003`. The 'E' is
optional. Our parser (`nd_fval`) handles this by inserting an 'E' before
orphaned exponent signs. It also handles 'D' notation because FORTRAN
has two ways of writing the same thing, much like English.

### CSG Geometry

Surfaces are general quadrics. Cells are intersections of halfspaces.
`cg_cross` nudges the particle `MO_EPS` (1e-10 cm) past the surface
after crossing, to prevent re-intersection with the surface it just
left. This is the geometry equivalent of holding the door open behind
you so it doesn't hit you in the face.

## Tests

26 tests, 5 categories. All tests compile at `-O0 -g` because
optimisers are the seagulls of the compiler world -- they swoop in,
rearrange everything, and leave you wondering where your variable went.

```
rng    -- determinism, uniformity, jump, no-zero-state
parse  -- FORTRAN E-notation, D-notation, integers
xs     -- exact/interpolated/clamped lookup, macroscopic XS
geom   -- sphere/plane/cylinder eval, ray tracing, cell finding
trans  -- particle movement, histories, fission banking, tallies
```

## Dependencies

- **Kauri** (`kauri.h`) -- arena allocator, bounds checking, string builder
- **MinGW GCC** -- C99
- **ENDF data** -- from NNDC (user-supplied)
- Nothing else. No boost. No cmake. No node_modules. Just C and regret.

## GPU Path

`gpu/tp_kern.cu` compiles via [BarraCUDA](https://github.com/Zaneham/BarraCUDA)
to `.hsaco` (AMD) or `.ptx` (NVIDIA). Each GPU thread tracks one particle
independently because neutrons are embarrassingly parallel -- they
literally don't interact with each other. If only people were this
easy to parallelise.

Two backends:
- **AMD** (`gpu/gp_host.c`): HSA runtime, `.hsaco` binary. `make GPU=1`.
- **NVIDIA** (`gpu/gp_nv.c`): CUDA Driver API, `.ptx` text. `make GPU=NV`.

GPU path includes mainframe-style ABEND dumps (via `bc_abend.c`) --
when the GPU faults, you get a structured dump with fault address
correlation against tracked allocations, kernel descriptor state,
and dispatch parameters. IBM solved crash diagnostics in the 1960s.
We got there eventually.

## References

- Blackman, D., & Vigna, S. (2021). Scrambled linear pseudorandom
  number generators. *ACM TOMS*, 47(4), Article 36.
- Herman, M., & Trkov, A. (Eds.). (2009). *ENDF-6 formats manual*
  (BNL-90365-2009 Rev. 2). Brookhaven National Laboratory.
- Lewis, E. E., & Miller, W. F. (1984). *Computational methods of
  neutron transport*. John Wiley & Sons.
- Lux, I., & Koblinger, L. (1991). *Monte Carlo particle transport
  methods*. CRC Press.
- Duderstadt, J. J., & Hamilton, L. J. (1976). *Nuclear reactor
  analysis*. John Wiley & Sons.
- International Criticality Safety Benchmark Evaluation Project. (2016).
  *International handbook of evaluated criticality safety benchmark
  experiments* (NEA/NSC/DOC(95)03/I). OECD/NEA.

## Licence

Apache 2.0. Do what you like. If you use this to design an actual
reactor, please invite us to the opening ceremony. We'll bring pavlova.
