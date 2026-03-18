/* moa.h -- Monte Carlo neutron transport
 *
 * Named after the moa: a large, extinct New Zealand bird that didn't
 * see it coming. Much like the neutrons we're about to simulate.
 *
 * All types, constants, pool sizes. One header to rule them all,
 * and in the darkness bind them (to pre-allocated memory).
 *
 * (c) 2026 Zane. Because "what would be cool to build" is a
 * dangerous question when you know C99.
 */

#ifndef MOA_H
#define MOA_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "C:/dev/kauri/kauri.h"

/* ---- Pool Sizes ----
 * Fixed at compile time. Like reactor containment: if you need
 * bigger, you're building a different reactor. */

#define MO_MAX_SURF   512      /* max surfaces in problem              */
#define MO_MAX_CELL   256      /* max cells (bumped for lattice univ)  */
#define MO_MAX_NUC    48       /* max nuclides                         */
#define MO_MAX_EPTS   10000    /* max energy grid points per nuclide   */
#define MO_MAX_PART   10000000 /* max particles per batch              */
#define MO_MAX_TBIN   1000     /* max tally bins                       */
#define MO_MAX_REG    64       /* max tally regions                    */
#define MO_MAX_MAT    16       /* max materials                        */
#define MO_MAX_HALF   16       /* max halfspaces per cell              */
#define MO_MAX_MNUC   8        /* max nuclides per material            */
#define MO_MAX_HIST   10000    /* max tracking steps per history       */
#define MO_MAX_RES    4096     /* max resonances per nuclide           */
#define MO_MAX_DLEV   40       /* max discrete inelastic levels        */
#define MO_MAX_MUBAR  200      /* max μ̄(E) angular distribution pts    */

/* Phase 3: energy-binned + mesh tallies + fixed-source */
#define MO_MAX_EBIN  256       /* max energy tally bins                */
#define MO_MESH_NX   64        /* mesh tally voxels X                  */
#define MO_MESH_NY   64        /* mesh tally voxels Y                  */
#define MO_MESH_NZ   64        /* mesh tally voxels Z                  */
#define MO_MESH_BIN  (MO_MESH_NX * MO_MESH_NY * MO_MESH_NZ)

/* Phase 4: thermal scattering */
#define MO_MAX_SAB   4         /* S(α,β) tables (H2O, graphite, etc.)  */
#define MO_MAX_SINC  200       /* SAB incident energy grid pts         */
#define MO_MAX_BETA  200       /* SAB β grid pts                       */
#define MO_MAX_SDAT  40000     /* SAB data points total                */

/* Phase 5: lattice geometry */
#define MO_MAX_LAT   8         /* lattice definitions                  */
#define MO_MAX_UNIV  32        /* universes                            */
#define MO_MAX_LELEM 1024      /* lattice elements (32×32 max)         */

/* Phase 6: nuclear data improvements */
#define MO_MAX_UBIN  20        /* URR energy bins                      */
#define MO_MAX_UPROB 20        /* URR probability entries per bin      */

/* Phase 7: OpenMP parallelism */
#define MO_MAX_THRD  16        /* max OpenMP threads                   */
#define MO_PT_FBSZ   (MO_MAX_PART / MO_MAX_THRD)  /* per-thread fbank */

/* ---- Physical Constants ---- */

#define MO_PI     3.14159265358979323846
#define MO_EPS    1.0e-10    /* nudge distance past surfaces (cm)     */
#define MO_KB     8.6173e-5  /* Boltzmann constant (eV/K)             */
#define MO_EMIN   1.0e-5     /* min energy cutoff (eV) — below this, absorb */
#define MO_EMAX   2.0e7      /* max energy (eV) — 20 MeV, fission spectrum  */
#define MO_WMIN   0.25       /* weight cutoff for Russian roulette    */
#define MO_WSURV  1.0        /* survival weight after roulette        */

/* ---- Surface Types ---- */

enum {
    MO_PX, MO_PY, MO_PZ,          /* axis-aligned planes       */
    MO_CX, MO_CY, MO_CZ,          /* axis-aligned cylinders    */
    MO_SPH,                        /* sphere                    */
    MO_GQ                          /* general quadric           */
};

/* ---- Particle ---- */

typedef struct {
    double x, y, z;       /* position (cm)                         */
    double u, v, w;       /* direction cosines (unit vector)       */
    double E;             /* energy (eV)                           */
    double wt;            /* statistical weight                    */
    int    cell;          /* current cell index                    */
    int    alive;         /* 0 = dead, 1 = alive                   */
    int    lat;           /* lattice index (-1=none)               */
    int    li, lj, lk;    /* lattice element indices               */
} mo_part_t;

/* ---- Surface ----
 * General quadric: Ax² + By² + Cz² + Dxy + Eyz + Fxz + Gx + Hy + Jz + K = 0
 * Most surfaces only use a few of these, but memory is cheap and
 * special-casing is not. */

typedef struct {
    int    type;
    double A, B, C, D, E, F, G, H, J, K;
} mo_surf_t;

/* ---- Cell (boolean CSG) ----
 * Intersection of halfspaces. If you need union, make more cells.
 * This isn't a CAD system, it's a physics engine with delusions. */

typedef struct {
    int    n_half;                   /* number of halfspaces     */
    int    surf[MO_MAX_HALF];        /* surface indices          */
    int    sense[MO_MAX_HALF];       /* +1 outside, -1 inside    */
    int    mat;                      /* material index (-1=void) */
    double rho;                      /* atom density (at/b-cm)   */
    double temp;                     /* temperature (K), 0=cold  */
    int    lat_id;                   /* lattice index (-1=none)  */
    int    univ;                     /* universe id (0=root)     */
} mo_cell_t;

/* ---- Resonance Parameters ----
 * One per resolved resonance. Parsed from ENDF MF2/MT151.
 * For SLBW/MLBW (LRF=1,2): Gfa = GF (fission width), Gfb = 0.
 * For Reich-Moore (LRF=3):  Gfa, Gfb = partial fission channel
 * widths (signed — the signs encode interference between fission
 * channels, which matters for R-matrix but not for our SLBW
 * approximation where we just compute √(Gfa² + Gfb²)). */

typedef struct {
    double Er;     /* resonance energy (eV)                     */
    double AJ;     /* spin J (absolute value used)              */
    double Gn;     /* neutron width Γ_n at E_r (eV)             */
    double Gg;     /* capture width Γ_γ (eV)                    */
    double Gfa;    /* fission width a (eV), or GF for SLBW      */
    double Gfb;    /* fission width b (eV), 0 for SLBW          */
} mo_res_t;

/* ---- Nuclide (ENDF data) ----
 * One of these per isotope. Cross-section data that someone
 * measured in a reactor so we could simulate it on a laptop.
 * Progress. Now with resonance parameters, because the smooth
 * MF3 background below the resolved region boundary is about
 * as useful as a chocolate teapot. */

typedef struct {
    int    ZA;                       /* Z*1000 + A               */
    double awr;                      /* atomic weight ratio       */
    int    n_pts;                    /* energy grid points        */
    double E[MO_MAX_EPTS];          /* energy grid (eV)          */
    double xs_t[MO_MAX_EPTS];       /* total xs (barns)          */
    double xs_a[MO_MAX_EPTS];       /* absorption (cap+fiss)     */
    double xs_f[MO_MAX_EPTS];       /* fission xs                */
    double xs_el[MO_MAX_EPTS];      /* elastic scatter xs        */
    double nu[MO_MAX_EPTS];         /* nu-bar (n/fission)        */

    /* MF2 resonance data — resolved region only.
     * Below EH, MF3 gives only the smooth background (~0 for
     * U-235). The actual cross-section requires adding the
     * resonance contribution computed from these parameters. */
    int    lrf;                      /* formalism: 0=none, 1=SLBW,
                                        2=MLBW, 3=Reich-Moore     */
    double SPI;                      /* target spin I             */
    double AP;                       /* scattering radius (1e-12 cm) */
    double EL, EH;                  /* resolved range (eV)       */
    int    n_res;                    /* number of resonances      */
    mo_res_t res[MO_MAX_RES];       /* resonance parameters      */

    /* MF3/MT51-MT89 discrete inelastic levels.
     * Excitation energies parsed from QI values. Used to
     * determine outgoing neutron energy after inelastic scatter
     * instead of guessing with an evaporation model. */
    int    n_lev;                    /* discrete inelastic levels */
    double lev_Q[MO_MAX_DLEV];     /* excitation energies (eV)  */

    /* MF4/MT2 elastic angular distribution — average scattering
     * cosine μ̄(E) in centre-of-mass frame. Without this, every
     * elastic scatter is isotropic, which is like assuming every
     * billiard ball ricochets at a random angle regardless of how
     * you hit it. At 1-2 MeV on uranium, half the scatters barely
     * change direction. Pretending otherwise adds ~4% to k_eff
     * because neutrons that should be streaming out of the geometry
     * keep getting randomly redirected back in. */
    int    n_mu;                     /* μ̄ table points            */
    double mu_E[MO_MAX_MUBAR];      /* energy grid (eV)          */
    double mu_bar[MO_MAX_MUBAR];    /* avg cos(θ_CM) at each E   */

    /* Unresolved resonance region (LRU=2) — probability tables.
     * Between EH_ur and the resolved boundary, xs is not deterministic
     * but follows a probability distribution. Each energy bin has a
     * discrete CDF of xs values. We sample one per history for
     * self-shielding. */
    int    n_ubin;                  /* URR energy bins            */
    int    n_uprob;                 /* probability entries/bin    */
    double EL_ur, EH_ur;           /* URR energy bounds (eV)     */
    double u_E[MO_MAX_UBIN];       /* URR energy grid            */
    double u_prob[MO_MAX_UBIN][MO_MAX_UPROB];
    double u_xs_t[MO_MAX_UBIN][MO_MAX_UPROB];
    double u_xs_f[MO_MAX_UBIN][MO_MAX_UPROB];
    double u_xs_a[MO_MAX_UBIN][MO_MAX_UPROB];
    double u_xs_el[MO_MAX_UBIN][MO_MAX_UPROB];
} mo_nucl_t;

/* ---- Material ---- */

typedef struct {
    int    n_nuc;                    /* number of nuclides        */
    int    nuc_id[MO_MAX_MNUC];     /* nuclide pool indices      */
    double frac[MO_MAX_MNUC];       /* atom fractions            */
    int    sab_id[MO_MAX_MNUC];     /* S(α,β) table per nuc, -1=none */
} mo_mat_t;

/* ---- Tally ----
 * Track-length and collision estimators. Accumulates sum and
 * sum-of-squares because statistics is just applied pessimism. */

typedef struct {
    int    n_bat;                    /* batches scored            */
    double sum[MO_MAX_REG];         /* running sum of batch avgs */
    double sq[MO_MAX_REG];          /* running sum-of-squares    */
    double bat[MO_MAX_REG];         /* current batch accumulator */
} mo_taly_t;

/* ---- RNG State ----
 * xoshiro256**. Fast, well-distributed, and pronounceable if
 * you've had enough coffee. */

typedef struct {
    uint64_t s[4];
} mo_rng_t;

/* ---- Cross-Section Result ----
 * All partial xs at a single energy. Avoids repeated lookups
 * when the transport loop needs everything at once. */

typedef struct {
    double sig_t;     /* total macroscopic xs (1/cm)     */
    double sig_a;     /* absorption (1/cm)               */
    double sig_f;     /* fission (1/cm)                  */
    double sig_el;    /* elastic scatter (1/cm)           */
    double sig_in;    /* inelastic scatter (1/cm)         */
    double nu_bar;    /* avg neutrons per fission         */
    int    fiss_za;   /* ZA of dominant fissile nuclide   */
} mo_xs_t;

/* ---- Fission Bank Site ---- */

typedef struct {
    double x, y, z;   /* position                        */
    double E;          /* energy (sampled from Watt)       */
    double wt;         /* weight                          */
} mo_site_t;

/* ---- Energy-Binned Tally ----
 * Score flux in energy bins. Log-uniform default grid,
 * or user-defined edges for the discerning physicist.
 * Three 2D arrays (bat/sum/sq) at ~128 KB each. Not small,
 * but mo_prob_t is already heap-allocated so who's counting. */

typedef struct {
    int    n_bin;
    int    n_reg;
    double edge[MO_MAX_EBIN + 1];
    int    reg[MO_MAX_REG];
    double bat[MO_MAX_REG][MO_MAX_EBIN];
    double sum[MO_MAX_REG][MO_MAX_EBIN];
    double sq[MO_MAX_REG][MO_MAX_EBIN];
    int    n_bat;
    int    active;
} mo_etly_t;

/* ---- Mesh Tally ----
 * Overlay Cartesian grid, score track-length flux per voxel.
 * Heap-allocated separately (~6 MB). Gnuplot-friendly dump. */

typedef struct {
    double lo[3], hi[3];
    int    nx, ny, nz;
    double bat[MO_MESH_BIN];
    double sum[MO_MESH_BIN];
    double sq[MO_MESH_BIN];
    int    n_bat;
    int    active;
} mo_mtly_t;

/* ---- Fixed Source ---- */

enum { MO_MODE_CRIT, MO_MODE_FIXD };
enum { MO_SRC_PT, MO_SRC_VOL };

typedef struct {
    int    type;               /* MO_SRC_PT or MO_SRC_VOL     */
    double pos[3];
    double E;
    int    cell;
} mo_src_t;

/* ---- S(α,β) Thermal Scattering ----
 * Bound-atom scattering kernels for water, graphite, etc.
 * Parsed from ENDF MF7/MT2. Without this, H-in-H2O is
 * about as realistic as a free hydrogen atom in vacuum,
 * which is to say, not very realistic in a reactor. */

typedef struct {
    int    ZA;
    int    n_inc;
    double E_inc[MO_MAX_SINC];
    int    n_beta;
    double beta[MO_MAX_BETA];
    double sab[MO_MAX_SDAT];   /* flattened [n_inc × n_beta]   */
    double sig_b;               /* bound scattering xs (barns)  */
    double temp;
} mo_sab_t;

/* ---- Lattice ----
 * Repeat geometry in a regular grid. Without this, a 17×17
 * fuel assembly needs ~1500 surfaces. With it: ~10. The
 * nuclear engineer's copy-paste. */

typedef struct {
    int    type;               /* 0=rect, 1=hex                */
    int    nx, ny, nz;
    double pitch[3];
    double orig[3];
    int    fill[MO_MAX_LELEM]; /* universe IDs per element     */
} mo_lat_t;

typedef struct {
    int    cell_lo, cell_hi;   /* range in cell pool           */
} mo_univ_t;

/* ---- Problem Context ----
 * Everything. The whole reactor in a struct. Heap-allocated because
 * ~18 MB on the stack would make Windows weep. */

typedef struct {
    /* geometry */
    mo_surf_t  surf[MO_MAX_SURF];
    int        n_surf;
    mo_cell_t  cell[MO_MAX_CELL];
    int        n_cell;

    /* nuclear data */
    mo_nucl_t  nucl[MO_MAX_NUC];
    int        n_nucl;
    mo_mat_t   mat[MO_MAX_MAT];
    int        n_mat;

    /* tallies */
    mo_taly_t  tl_flux;              /* flux tally               */
    mo_taly_t  tl_fiss;              /* fission rate tally       */
    mo_etly_t  etl_flux;             /* energy-binned flux tally */
    mo_mtly_t  *mesh;                /* mesh tally (heap, ~6 MB) */

    /* S(α,β) thermal scattering */
    mo_sab_t   sab[MO_MAX_SAB];
    int        n_sab;

    /* lattice geometry */
    mo_lat_t   lat[MO_MAX_LAT];
    int        n_lat;
    mo_univ_t  univ[MO_MAX_UNIV];
    int        n_univ;

    /* criticality params */
    int        n_batch;              /* total batches            */
    int        n_skip;               /* inactive batches         */
    int        n_part;               /* particles per batch      */
    double     k_eff;                /* current k estimate       */
    int        verbose;              /* verbosity level          */
    int        mode;                 /* MO_MODE_CRIT / MO_MODE_FIXD */
    int        n_thrd;               /* OpenMP thread count      */
    mo_src_t   src;                  /* fixed-source definition  */
    int        use_gpu;              /* GPU transport (0=off)    */
    char       hsaco_path[256];      /* path to .hsaco kernel    */

    /* RNG */
    mo_rng_t   rng;

    /* Kauri arena for scratch */
    ka_arena_t arena;
} mo_prob_t;


/* ---- Function Declarations ---- */

/* rng.c */
void     rn_init(mo_rng_t *r, uint64_t seed);
uint64_t rn_next(mo_rng_t *r);
double   rn_dbl (mo_rng_t *r);
void     rn_jump(mo_rng_t *r);

/* nd_parse.c */
double   nd_fval(const char *s, int n);
int      nd_ival(const char *s);
int      nd_tab1(FILE *f, double *x, double *y, int max_pts);
int      nd_load(mo_nucl_t *nuc, const char *path, int za);

/* nd_xs.c */
void     xs_look(const mo_nucl_t *nuc, double E,
                 double *st, double *sa, double *sf, double *sel, double *nu);
mo_xs_t  xs_macro(const mo_prob_t *prob, int cell, double E);

/* nd_res.c */
void     rs_eval(const mo_nucl_t *nuc, double E,
                 double *st, double *sf, double *sg, double *sel);

/* csg.c */
double   cg_eval(const mo_surf_t *s, double x, double y, double z);
double   cg_dist(const mo_surf_t *s, double x, double y, double z,
                 double u, double v, double w);
int      cg_find(const mo_prob_t *prob, double x, double y, double z);
double   cg_next(const mo_prob_t *prob, const mo_part_t *p, int *s_hit);
int      cg_cross(const mo_prob_t *prob, mo_part_t *p, int s_hit);
void     cg_plane(mo_surf_t *s, int axis, double d);
void     cg_cyl (mo_surf_t *s, int axis, double r);
void     cg_sph (mo_surf_t *s, double x0, double y0, double z0, double r);

/* tl_score.c */
void     tl_init(mo_taly_t *t);
void     tl_coll(mo_taly_t *t, int reg, double wt, double sig_t);
void     tl_trak(mo_taly_t *t, int reg, double wt, double dist);
void     tl_fiss(mo_taly_t *t, int reg, double wt, double nu,
                 double sig_f, double sig_t);
void     tl_stat(const mo_taly_t *t, int reg, int n_bat,
                 double *mean, double *sdev);
void     tl_norm(mo_taly_t *t, int n_part);
void     tl_zero(mo_taly_t *t);

/* tp_loop.c */
int      tp_hist(mo_prob_t *prob, mo_part_t *p,
                 mo_site_t *fbank, int *n_fbank);
int      tp_fhst(mo_prob_t *prob, mo_part_t *p,
                 mo_site_t *fbank, int *n_fbank);
void     tp_move(mo_part_t *p, double d);
void     tp_oset(mo_prob_t *prob, int n_thrd);
void     tp_ored(mo_prob_t *prob, int n_thrd);
mo_rng_t *tp_trng(int tid);

/* tp_crit.c */
void     tc_run (mo_prob_t *prob);
void     tc_src (mo_prob_t *prob, mo_site_t *bank, int n);

/* tp_fixd.c */
void     tf_run (mo_prob_t *prob);
void     tf_src (mo_prob_t *prob, mo_src_t *src, mo_part_t *p,
                 mo_rng_t *rng);

/* tl_ebin.c */
void     te_init(mo_etly_t *t);
void     te_set (mo_etly_t *t, int n_bin, double emin, double emax);
void     te_scor(mo_etly_t *t, int reg, double wt, double dist, double E);
void     te_norm(mo_etly_t *t, int n_part);
void     te_stat(const mo_etly_t *t, int reg, int bin, int n_bat,
                 double *mean, double *sdev);
void     te_prnt(const mo_etly_t *t, int n_bat);

/* tl_mesh.c */
void     tm_init(mo_mtly_t *t, const double lo[3], const double hi[3]);
void     tm_scor(mo_mtly_t *t, double x, double y, double z,
                 double wt, double dist);
void     tm_norm(mo_mtly_t *t, int n_part);
void     tm_stat(const mo_mtly_t *t, int ix, int iy, int iz, int n_bat,
                 double *mean, double *sdev);
void     tm_dump(const mo_mtly_t *t, int n_bat, const char *path);

/* nd_thrm.c */
void     th_fgas(mo_rng_t *rng, double E, double A, double kT,
                 double *E_out, double *mu_lab);
double   th_maxw(mo_rng_t *rng, double kT, double A);

/* nd_sab.c */
int      sb_load(mo_sab_t *sab, const char *path, int za);
void     sb_samp(const mo_sab_t *sab, mo_rng_t *rng, double E_in,
                 double *E_out, double *mu);

/* nd_rmat.c */
void     rm_eval(const mo_nucl_t *nuc, double E,
                 double *sig_t, double *sig_f, double *sig_g, double *sig_el);

/* nd_urr.c */
int      ur_load(mo_nucl_t *nuc, FILE *f, const char *head);
void     ur_samp(const mo_nucl_t *nuc, double E, mo_rng_t *rng,
                 double *st, double *sf, double *sa, double *sel);

/* nd_dopl.c */
void     dp_brod(mo_nucl_t *nuc, double T);

/* cg_lat.c */
int      cl_find(const mo_prob_t *prob, const mo_lat_t *lat,
                 double x, double y, double z, int *li, int *lj, int *lk);
void     cl_locl(const mo_lat_t *lat, int i, int j, int k,
                 double x, double y, double z,
                 double *lx, double *ly, double *lz);
void     cl_glob(const mo_lat_t *lat, int i, int j, int k,
                 double lx, double ly, double lz,
                 double *x, double *y, double *z);
double   cl_next(const mo_prob_t *prob, const mo_lat_t *lat,
                 const mo_part_t *p, int *s_hit);

/* io_input.c */
int      io_load(mo_prob_t *prob, const char *path);

/* gpu/gp_host.c (conditional via GPU=1 build) */
#ifdef MOA_GPU
#ifdef MOA_GPU_NV
#include "gpu/gp_nv.h"
#else
#include "gpu/gp_host.h"
#endif
#endif

#endif /* MOA_H */
