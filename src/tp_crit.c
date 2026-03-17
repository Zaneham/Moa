/* tp_crit.c -- Criticality power iteration
 *
 * The k-eigenvalue solver. We iterate fission generations: each
 * batch of neutrons produces fission sites that become the source
 * for the next batch. The ratio of new sites to old is k_eff.
 *
 * k = 1.0: critical (steady state, good for reactors)
 * k > 1.0: supercritical (bad for everyone except weapons designers)
 * k < 1.0: subcritical (reactor is dying, also bad)
 *
 * The first n_skip batches are "inactive" — thrown away to let the
 * fission source converge. Like warming up an engine before driving.
 * Except the engine is a nuclear reactor. So maybe don't rush it.
 *
 * Shannon entropy of the fission source distribution tells us when
 * the source has actually converged. k_eff can look converged while
 * the source is still wandering around like a tourist in Wellington.
 *
 * References:
 *   Brown, F. B. (2009). Fundamentals of Monte Carlo particle
 *   transport (LA-UR-05-4983 Rev.). Los Alamos National Laboratory.
 *
 *   Brown, F. B. (2006). On the use of Shannon entropy of the
 *   fission distribution for assessing convergence of Monte Carlo
 *   criticality calculations. Proc. PHYSOR-2006.
 *   LA-UR-06-3954. Brookhaven National Laboratory.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include "../moa.h"
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- Shannon entropy mesh ----
 * Overlay a 3D Cartesian mesh on the geometry. Each fission site
 * is binned; the Shannon entropy H = -Σ(pᵢ·log₂(pᵢ)) measures
 * how uniformly the source is distributed.
 *
 * When H stabilises (fluctuates around a constant), the source
 * has converged. When it's still trending, you need more inactive
 * batches. MCNP5 rule of thumb: ~20 particles per bin.
 *
 * 8³ = 512 bins with 10K particles ≈ 20 per non-empty bin. */

#define SE_NX  8
#define SE_NY  8
#define SE_NZ  8
#define SE_LO  -25.0   /* bounding box lower (cm) */
#define SE_HI   25.0   /* bounding box upper (cm) */
#define SE_NBIN (SE_NX * SE_NY * SE_NZ)

static double
tc_ent(const mo_site_t *bank, int n_bank)
{
    int bins[SE_NBIN];
    int i;
    double H = 0.0;

    if (n_bank <= 0) return 0.0;

    memset(bins, 0, sizeof(bins));

    double dx = (SE_HI - SE_LO) / SE_NX;
    double dy = (SE_HI - SE_LO) / SE_NY;
    double dz = (SE_HI - SE_LO) / SE_NZ;

    for (i = 0; i < n_bank; i++) {
        int ix = (int)((bank[i].x - SE_LO) / dx);
        int iy = (int)((bank[i].y - SE_LO) / dy);
        int iz = (int)((bank[i].z - SE_LO) / dz);

        /* Clamp to mesh bounds. Particles outside the mesh
         * get stuffed into the nearest edge bin, which is fine
         * because they shouldn't be outside anyway. */
        if (ix < 0) ix = 0;
        if (ix >= SE_NX) ix = SE_NX - 1;
        if (iy < 0) iy = 0;
        if (iy >= SE_NY) iy = SE_NY - 1;
        if (iz < 0) iz = 0;
        if (iz >= SE_NZ) iz = SE_NZ - 1;

        bins[ix * SE_NY * SE_NZ + iy * SE_NZ + iz]++;
    }

    for (i = 0; i < SE_NBIN; i++) {
        if (bins[i] > 0) {
            double Si = (double)bins[i] / (double)n_bank;
            H -= Si * log2(Si);
        }
    }
    return H;
}

/* ---- Initial source ----
 * Distribute n particles uniformly in fissile cells.
 * For Godiva this means uniformly in the U-235 sphere. */

void
tc_src(mo_prob_t *prob, mo_site_t *bank, int n)
{
    int i, attempts;

    /* Find the bounding box of all fissile cells by checking
     * spherical surfaces. For a simple sphere this is easy. */
    double xmin = -20.0, xmax = 20.0;
    double ymin = -20.0, ymax = 20.0;
    double zmin = -20.0, zmax = 20.0;

    for (i = 0; i < n; i++) {
        bank[i].wt = 1.0;
        bank[i].E  = 2.0e6;  /* born at 2 MeV, fission spectrum peak */

        /* Rejection sample: pick random point in bounding box,
         * accept if it's in a fissile cell */
        attempts = 0;
        KA_GUARD(g, 10000);
        while (g--) {
            double x = xmin + rn_dbl(&prob->rng) * (xmax - xmin);
            double y = ymin + rn_dbl(&prob->rng) * (ymax - ymin);
            double z = zmin + rn_dbl(&prob->rng) * (zmax - zmin);

            int c = cg_find(prob, x, y, z);
            if (c >= 0 && prob->cell[c].mat >= 0) {
                /* Check if material has any fissile nuclides */
                const mo_mat_t *m = &prob->mat[prob->cell[c].mat];
                int j, fissile = 0;
                for (j = 0; j < m->n_nuc; j++) {
                    int nid = m->nuc_id[j];
                    if (!KA_CHK(nid, prob->n_nucl)) {
                        /* Has fission data? Check if xs_f has any nonzero */
                        if (prob->nucl[nid].xs_f[0] > 0.0 ||
                            prob->nucl[nid].xs_f[prob->nucl[nid].n_pts/2] > 0.0)
                            fissile = 1;
                    }
                }
                if (fissile) {
                    bank[i].x = x;
                    bank[i].y = y;
                    bank[i].z = z;
                    break;
                }
            }
            attempts++;
        }

        /* If we couldn't find a fissile cell, just put it at origin.
         * This will cause problems, but at least it won't segfault. */
        if (attempts >= 9999) {
            bank[i].x = 0.0;
            bank[i].y = 0.0;
            bank[i].z = 0.0;
        }
    }
}

/* ---- Normalise fission bank ----
 * If we have more or fewer sites than n_part, resample to
 * get exactly n_part sites. This keeps the population stable. */

static void
tc_norm(mo_prob_t *prob, mo_site_t *bank, int n_bank, int n_want)
{
    if (n_bank <= 0 || n_want <= 0) return;

    if (n_bank == n_want) return;  /* Goldilocks zone */

    /* Simple resampling: pick n_want sites uniformly from the bank.
     * This is unbiased but loses some information. Good enough. */
    static mo_site_t tmp[MO_MAX_PART];
    int i;

    int n_copy = (n_bank < MO_MAX_PART) ? n_bank : MO_MAX_PART;
    memcpy(tmp, bank, n_copy * sizeof(mo_site_t));

    for (i = 0; i < n_want; i++) {
        int idx = (int)(rn_dbl(&prob->rng) * n_copy);
        if (idx >= n_copy) idx = n_copy - 1;
        bank[i] = tmp[idx];
        bank[i].wt = 1.0;
    }
}

/* ---- Run criticality calculation ---- */

void
tc_run(mo_prob_t *prob)
{
    static mo_site_t bank_a[MO_MAX_PART];
    static mo_site_t bank_b[MO_MAX_PART];
    mo_site_t *src_bank = bank_a;
    mo_site_t *fis_bank = bank_b;

    int n_src, n_fis;
    int batch;
    double k_sum = 0.0, k_sq = 0.0;
    int k_cnt = 0;

    /* Running average over ALL batches (inactive + active) for
     * normalization. Using the raw batch k causes yo-yo oscillation
     * in reflected systems where k_batch alternates between 0.5 and
     * 2.0 like a seismograph during an earthquake. */
    double k_all = 0.0;
    int k_all_n = 0;

    /* Timing — wall clock via POSIX monotonic */
    struct timespec ts0, ts1, tb0, tb1;
    double t_wall = 0.0;
    (void)tb0; (void)tb1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    /* Leakage tracking — fraction of neutrons that escape the
     * geometry. High for bare assemblies (Godiva ~67%), low for
     * reflected ones (Flattop ~3%). Useful for sanity checking. */
    double lk_sum = 0.0, lk_sq = 0.0;
    int lk_cnt = 0;

    int np = prob->n_part;
    if (np > MO_MAX_PART) np = MO_MAX_PART;

    prob->k_eff = 1.0;

    /* Initialise tallies — the real init, not the per-batch zero */
    tl_init(&prob->tl_flux);
    tl_init(&prob->tl_fiss);

    /* Generate initial source */
    tc_src(prob, src_bank, np);
    n_src = np;

    printf("moa: starting criticality calculation\n");
    printf("     %d particles/batch, %d batches (%d inactive)\n",
           np, prob->n_batch, prob->n_skip);
    printf("     %-6s  %-12s  %-20s  %-6s  %s\n",
           "batch", "k_eff", "k_avg +/- std", "H_src", "time");

    /* ---- Power iteration ---- */
    for (batch = 1; batch <= prob->n_batch; batch++) {
        int i;
        int n_leak = 0;
        n_fis = 0;

        clock_gettime(CLOCK_MONOTONIC, &tb0);
        /* Reset per-batch accumulators */
        tl_zero(&prob->tl_flux);
        tl_zero(&prob->tl_fiss);

        /* Zero energy tally batch accumulators */
        if (prob->etl_flux.active) {
            int r, b;
            for (r = 0; r < prob->etl_flux.n_reg; r++)
                for (b = 0; b < prob->etl_flux.n_bin; b++)
                    prob->etl_flux.bat[r][b] = 0.0;
        }
        /* Zero mesh tally batch accumulator */
        if (prob->mesh && prob->mesh->active) {
            int j;
            for (j = 0; j < MO_MESH_BIN; j++)
                prob->mesh->bat[j] = 0.0;
        }

        /* Transport all particles in this batch */
        {
            int nn = (n_src < np) ? n_src : np;
#ifdef MOA_GPU
            if (prob->use_gpu) {
                /* GPU transport — one kernel dispatch per batch.
                 * The neutrons don't know they've been offloaded
                 * to an AMD GPU via a hand-rolled CUDA compiler.
                 * Ignorance is bliss. */
                extern void *g_gpu_ctx;
                if (g_gpu_ctx) {
                    gp_ctx_t *gc = (gp_ctx_t *)g_gpu_ctx;
                    int grc = gp_batch(gc, prob,
                                       src_bank, nn,
                                       fis_bank, &n_fis);
                    if (grc < 0) {
                        fprintf(stderr, "moa: GPU batch failed, "
                                        "falling back to CPU\n");
                        prob->use_gpu = 0;
                    }
                }
            }
            if (!prob->use_gpu)
#endif
#ifdef _OPENMP
            if (prob->n_thrd > 1) {
                /* ---- Parallel transport ----
                 * Each thread gets its own RNG, tallies, and fission
                 * bank segment. Like separate lanes in a pool, except
                 * the swimmers are radioactive. */
                int nt = prob->n_thrd;
                int t;
                static mo_site_t pt_fb[MO_MAX_THRD][MO_PT_FBSZ];
                static int pt_nf[MO_MAX_THRD];

                if (nt > MO_MAX_THRD) nt = MO_MAX_THRD;
                for (t = 0; t < nt; t++) pt_nf[t] = 0;

                tp_oset(prob, nt);

                #pragma omp parallel num_threads(nt)
                {
                    int tid = omp_get_thread_num();
                    mo_rng_t *rng = tp_trng(tid);
                    int my_nf = 0, my_lk = 0;

                    #pragma omp for schedule(static)
                    for (i = 0; i < nn; i++) {
                        mo_part_t p;
                        memset(&p, 0, sizeof(p));
                        p.x     = src_bank[i].x;
                        p.y     = src_bank[i].y;
                        p.z     = src_bank[i].z;
                        p.E     = src_bank[i].E;
                        p.wt    = src_bank[i].wt;
                        p.alive = 1;
                        p.lat   = -1;

                        p.cell = cg_find(prob, p.x, p.y, p.z);
                        if (p.cell < 0 || prob->cell[p.cell].mat < 0)
                            continue;
                        {
                            double mu  = 2.0 * rn_dbl(rng) - 1.0;
                            double phi = 2.0 * MO_PI * rn_dbl(rng);
                            double st  = sqrt(1.0 - mu * mu);
                            p.u = st * cos(phi);
                            p.v = st * sin(phi);
                            p.w = mu;
                        }
                        my_lk += tp_hist(prob, &p,
                                         pt_fb[tid], &my_nf);
                    }

                    pt_nf[tid] = my_nf;
                    #pragma omp atomic
                    n_leak += my_lk;
                }

                tp_ored(prob, nt);

                /* Merge per-thread fission banks */
                n_fis = 0;
                for (t = 0; t < nt; t++) {
                    int tn = pt_nf[t];
                    if (tn > MO_PT_FBSZ) tn = MO_PT_FBSZ;
                    if (tn > 0 && n_fis + tn <= MO_MAX_PART) {
                        memcpy(&fis_bank[n_fis], pt_fb[t],
                               (size_t)tn * sizeof(mo_site_t));
                        n_fis += tn;
                    }
                }
            } else
#endif
            {
                for (i = 0; i < nn; i++) {
                    mo_part_t p;
                    memset(&p, 0, sizeof(p));
                    p.x     = src_bank[i].x;
                    p.y     = src_bank[i].y;
                    p.z     = src_bank[i].z;
                    p.E     = src_bank[i].E;
                    p.wt    = src_bank[i].wt;
                    p.alive = 1;
                    p.lat   = -1;

                    p.cell = cg_find(prob, p.x, p.y, p.z);
                    if (p.cell < 0 || prob->cell[p.cell].mat < 0)
                        continue;
                    {
                        double mu  = 2.0 * rn_dbl(&prob->rng) - 1.0;
                        double phi = 2.0 * MO_PI * rn_dbl(&prob->rng);
                        double st  = sqrt(1.0 - mu * mu);
                        p.u = st * cos(phi);
                        p.v = st * sin(phi);
                        p.w = mu;
                    }
                    n_leak += tp_hist(prob, &p, fis_bank, &n_fis);
                }
            }
        }

        /* Estimate k for this batch */
        double k_batch = (double)n_fis / (double)np;

        /* Update running average for normalization.
         * Using the smoothed estimate prevents the yo-yo effect
         * in reflected geometries (Flattop, etc.) */
        k_all += k_batch;
        k_all_n++;
        prob->k_eff = k_all / k_all_n;

        /* Shannon entropy of the fission source.
         * Computed BEFORE resampling so we see the raw distribution. */
        double H = tc_ent(fis_bank, n_fis);

        /* Accumulate active-batch statistics */
        if (batch > prob->n_skip) {
            k_sum += k_batch;
            k_sq  += k_batch * k_batch;
            k_cnt++;

            /* Fold tallies into running statistics */
            tl_norm(&prob->tl_flux, np);
            tl_norm(&prob->tl_fiss, np);
            if (prob->etl_flux.active)
                te_norm(&prob->etl_flux, np);
            if (prob->mesh && prob->mesh->active)
                tm_norm(prob->mesh, np);

            /* Leakage fraction for this batch */
            double lk_frac = (double)n_leak / (double)np;
            lk_sum += lk_frac;
            lk_sq  += lk_frac * lk_frac;
            lk_cnt++;

            double k_avg = k_sum / (double)k_cnt;
            double k_std = 0.0;
            if (k_cnt > 1) {
                double var = (k_sq - k_sum * k_sum / (double)k_cnt)
                           / (double)(k_cnt - 1);
                k_std = (var > 0.0) ? sqrt(var / (double)k_cnt) : 0.0;
            }

            clock_gettime(CLOCK_MONOTONIC, &tb1);
            {
                double bms = (tb1.tv_sec - tb0.tv_sec) * 1e3
                           + (tb1.tv_nsec - tb0.tv_nsec) * 1e-6;
                if (prob->verbose || batch == prob->n_batch
                    || (batch % 10) == 0) {
                    printf("     %4d    %10.5f    %10.5f +/- %.5f"
                           "  %.3f  %7.0f ms\n",
                           batch, k_batch, k_avg, k_std, H, bms);
                }
            }

            prob->k_eff = k_avg;
        } else {
            clock_gettime(CLOCK_MONOTONIC, &tb1);
            {
                double bms = (tb1.tv_sec - tb0.tv_sec) * 1e3
                           + (tb1.tv_nsec - tb0.tv_nsec) * 1e-6;
                if (prob->verbose || (batch % 10) == 0) {
                    printf("     %4d    %10.5f    (inactive)"
                           "           %.3f  %7.0f ms\n",
                           batch, k_batch, H, bms);
                }
            }
        }

        /* Swap banks: fission sites become next source */
        tc_norm(prob, fis_bank, n_fis, np);
        {
            mo_site_t *tmp = src_bank;
            src_bank = fis_bank;
            fis_bank = tmp;
            n_src = np;
        }
    }

    /* ---- Final results ---- */
    if (k_cnt > 0) {
        double k_avg = k_sum / (double)k_cnt;
        double k_std = 0.0;
        if (k_cnt > 1) {
            double var = (k_sq - k_sum * k_sum / (double)k_cnt)
                       / (double)(k_cnt - 1);
            k_std = (var > 0.0) ? sqrt(var / (double)k_cnt) : 0.0;
        }
        printf("\n     Final k_eff = %.5f +/- %.5f  (%d active batches)\n",
               k_avg, k_std, k_cnt);
        prob->k_eff = k_avg;

    }

    /* ---- Timing summary ---- */
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    t_wall = (ts1.tv_sec - ts0.tv_sec)
           + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    printf("\n     Wall time: %.3f s  (%.1f particles/s)\n",
           t_wall,
           (double)np * (double)prob->n_batch / t_wall);

    /* ---- Cell tallies ---- */
    if (k_cnt > 0) {
        int c;
        printf("\n     ---- Cell Tallies (%d active batches) ----\n", k_cnt);
        printf("     %4s  %3s  %14s  %12s  %14s  %12s\n",
               "cell", "mat", "flux", "+/- sigma",
               "fiss_rate", "+/- sigma");

        for (c = 0; c < prob->n_cell; c++) {
            if (prob->cell[c].mat < 0) continue; /* skip void */

            double f_mean, f_sdev, r_mean, r_sdev;
            tl_stat(&prob->tl_flux, c, k_cnt, &f_mean, &f_sdev);
            tl_stat(&prob->tl_fiss, c, k_cnt, &r_mean, &r_sdev);

            printf("     %4d  %3d  %14.5e  %12.3e  %14.5e  %12.3e\n",
                   c, prob->cell[c].mat,
                   f_mean, f_sdev, r_mean, r_sdev);
        }
    }

    /* ---- Leakage fraction ---- */
    if (lk_cnt > 0) {
        double lk_avg = lk_sum / (double)lk_cnt;
        double lk_std = 0.0;
        if (lk_cnt > 1) {
            double var = (lk_sq - lk_sum * lk_sum / (double)lk_cnt)
                       / (double)(lk_cnt - 1);
            lk_std = (var > 0.0) ? sqrt(var / (double)lk_cnt) : 0.0;
        }
        printf("\n     Leakage fraction = %.4f +/- %.4f\n", lk_avg, lk_std);
    }

    /* ---- Energy-binned tally ---- */
    if (prob->etl_flux.active && k_cnt > 0)
        te_prnt(&prob->etl_flux, k_cnt);

    /* ---- Mesh tally dump ---- */
    if (prob->mesh && prob->mesh->active && k_cnt > 0)
        tm_dump(prob->mesh, k_cnt, "mesh_out.dat");
}
