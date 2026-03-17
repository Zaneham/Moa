/* tp_fixd.c -- Fixed-source transport mode
 *
 * The gentler sibling of criticality: no fission banking, no
 * k-eigenvalue, no existential questions about whether the reactor
 * is critical. Just emit neutrons from a source, transport them
 * until death, and score tallies.
 *
 * Useful for shielding, dosimetry, and answering the question
 * "if I put a neutron source here, what happens over there?"
 * which is surprisingly relevant if you work near reactors.
 *
 * Source types:
 *   MO_SRC_PT  — isotropic point source at (x,y,z)
 *   MO_SRC_VOL — uniform volumetric source in a cell
 *
 * Reference:
 *   Brown, F. B. (2009). Fundamentals of Monte Carlo particle
 *   transport (LA-UR-05-4983 Rev.). Los Alamos National Laboratory.
 *   Section 3: fixed-source problems. */

#include "../moa.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- Sample source particle ---- */

void
tf_src(mo_prob_t *prob, mo_src_t *src, mo_part_t *p,
       mo_rng_t *rng)
{
    memset(p, 0, sizeof(*p));

    if (src->type == MO_SRC_PT) {
        /* Point source: all particles born at same location */
        p->x = src->pos[0];
        p->y = src->pos[1];
        p->z = src->pos[2];
        p->cell = cg_find(prob, p->x, p->y, p->z);
    } else {
        /* Volume source: rejection sample within the cell.
         * Bounding box is estimated from surfaces. For spheres
         * this is generous; for irregular cells, shrug emoji. */
        double lo = -50.0, hi = 50.0;

        KA_GUARD(g, 10000);
        while (g--) {
            p->x = lo + rn_dbl(rng) * (hi - lo);
            p->y = lo + rn_dbl(rng) * (hi - lo);
            p->z = lo + rn_dbl(rng) * (hi - lo);

            int c = cg_find(prob, p->x, p->y, p->z);
            if (c == src->cell) {
                p->cell = c;
                break;
            }
        }
        if (p->cell < 0) {
            /* Couldn't find the cell. Stuff it at the point. */
            p->x = src->pos[0];
            p->y = src->pos[1];
            p->z = src->pos[2];
            p->cell = cg_find(prob, p->x, p->y, p->z);
        }
    }

    p->E  = src->E;
    p->wt = 1.0;
    p->alive = 1;
    p->lat = -1;

    /* Isotropic direction */
    {
        double mu  = 2.0 * rn_dbl(rng) - 1.0;
        double phi = 2.0 * MO_PI * rn_dbl(rng);
        double st  = sqrt(1.0 - mu * mu);
        p->u = st * cos(phi);
        p->v = st * sin(phi);
        p->w = mu;
    }
}

/* ---- Run fixed-source calculation ----
 * Much simpler than tc_run: no fission bank, no Shannon entropy,
 * no k-eigenvalue. Just N batches of M particles from the source.
 * The kind of Monte Carlo problem that lets you sleep at night. */

void
tf_run(mo_prob_t *prob)
{
    int batch, i;
    int np = prob->n_part;
    int n_active;

    if (np > MO_MAX_PART) np = MO_MAX_PART;

    /* Tallies */
    tl_init(&prob->tl_flux);
    tl_init(&prob->tl_fiss);
    if (prob->etl_flux.active) {
        int r;
        for (r = 0; r < prob->etl_flux.n_reg; r++) {
            int b;
            for (b = 0; b < prob->etl_flux.n_bin; b++) {
                prob->etl_flux.bat[r][b] = 0.0;
            }
        }
    }

    printf("moa: starting fixed-source calculation\n");
    printf("     %d particles/batch, %d batches (%d inactive)\n",
           np, prob->n_batch, prob->n_skip);

    n_active = 0;

    for (batch = 1; batch <= prob->n_batch; batch++) {
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

        /* Transport particles */
#ifdef _OPENMP
        if (prob->n_thrd > 1) {
            int nt = prob->n_thrd;
            if (nt > MO_MAX_THRD) nt = MO_MAX_THRD;
            tp_oset(prob, nt);

            #pragma omp parallel num_threads(nt)
            {
                int tid = omp_get_thread_num();
                mo_rng_t *rng = tp_trng(tid);

                #pragma omp for schedule(static)
                for (i = 0; i < np; i++) {
                    mo_part_t p;
                    tf_src(prob, &prob->src, &p, rng);
                    if (p.cell < 0 || prob->cell[p.cell].mat < 0)
                        continue;
                    tp_hist(prob, &p, NULL, NULL);
                }
            }

            tp_ored(prob, nt);
        } else
#endif
        {
            for (i = 0; i < np; i++) {
                mo_part_t p;
                tf_src(prob, &prob->src, &p, &prob->rng);
                if (p.cell < 0 || prob->cell[p.cell].mat < 0)
                    continue;
                tp_hist(prob, &p, NULL, NULL);
            }
        }

        if (batch > prob->n_skip) {
            tl_norm(&prob->tl_flux, np);
            tl_norm(&prob->tl_fiss, np);
            if (prob->etl_flux.active)
                te_norm(&prob->etl_flux, np);
            if (prob->mesh && prob->mesh->active)
                tm_norm(prob->mesh, np);
            n_active++;

            if (prob->verbose || batch == prob->n_batch
                || (batch % 10) == 0) {
                printf("     batch %4d  (active)\n", batch);
            }
        } else {
            if (prob->verbose || (batch % 10) == 0) {
                printf("     batch %4d  (inactive)\n", batch);
            }
        }
    }

    /* ---- Results ---- */
    if (n_active > 0) {
        int c;
        printf("\n     ---- Cell Tallies (%d active batches) ----\n",
               n_active);
        printf("     %4s  %3s  %14s  %12s\n",
               "cell", "mat", "flux", "+/- sigma");

        for (c = 0; c < prob->n_cell; c++) {
            if (prob->cell[c].mat < 0) continue;
            double f_mean, f_sdev;
            tl_stat(&prob->tl_flux, c, n_active, &f_mean, &f_sdev);

            if (f_mean < 1.0e-30) continue;
            printf("     %4d  %3d  %14.5e  %12.3e\n",
                   c, prob->cell[c].mat, f_mean, f_sdev);
        }

        if (prob->etl_flux.active)
            te_prnt(&prob->etl_flux, n_active);

        if (prob->mesh && prob->mesh->active)
            tm_dump(prob->mesh, n_active, "mesh_out.dat");
    }
}
