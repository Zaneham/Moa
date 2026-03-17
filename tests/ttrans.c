/* ttrans.c -- Transport loop tests
 * Testing whether we can follow a neutron through a sphere without
 * losing it, breaking physics, or dividing by zero. A surprisingly
 * high bar for 400 lines of C. */

#include "../moa.h"
#include "tharns.h"

/* Build a minimal Godiva-like problem for testing:
 * Single sphere of U-235, radius 8.7407 cm */
static void
mk_godiva(mo_prob_t *prob)
{
    memset(prob, 0, sizeof(*prob));

    /* Surface 0: sphere at origin, r=8.7407 cm */
    cg_sph(&prob->surf[0], 0.0, 0.0, 0.0, 8.7407);
    prob->n_surf = 1;

    /* Cell 0: inside sphere */
    prob->cell[0].n_half = 1;
    prob->cell[0].surf[0] = 0;
    prob->cell[0].sense[0] = -1;
    prob->cell[0].mat = 0;
    prob->cell[0].rho = 0.04794;

    /* Cell 1: outside (void) */
    prob->cell[1].n_half = 1;
    prob->cell[1].surf[0] = 0;
    prob->cell[1].sense[0] = 1;
    prob->cell[1].mat = -1;
    prob->n_cell = 2;

    /* Synthetic nuclide: simplified U-235 */
    prob->nucl[0].ZA  = 92235;
    prob->nucl[0].awr = 235.0;
    prob->nucl[0].n_pts = 3;
    prob->nucl[0].E[0] = 1.0e-5;
    prob->nucl[0].E[1] = 1.0e3;
    prob->nucl[0].E[2] = 2.0e7;
    /* Total ~ 10 barns (fast spectrum) */
    prob->nucl[0].xs_t[0]  = 50.0;
    prob->nucl[0].xs_t[1]  = 15.0;
    prob->nucl[0].xs_t[2]  = 7.0;
    /* Elastic */
    prob->nucl[0].xs_el[0] = 10.0;
    prob->nucl[0].xs_el[1] = 5.0;
    prob->nucl[0].xs_el[2] = 3.0;
    /* Fission */
    prob->nucl[0].xs_f[0]  = 30.0;
    prob->nucl[0].xs_f[1]  = 5.0;
    prob->nucl[0].xs_f[2]  = 2.0;
    /* Absorption */
    prob->nucl[0].xs_a[0]  = 40.0;
    prob->nucl[0].xs_a[1]  = 10.0;
    prob->nucl[0].xs_a[2]  = 4.0;
    /* Nu-bar */
    prob->nucl[0].nu[0] = 2.43;
    prob->nucl[0].nu[1] = 2.43;
    prob->nucl[0].nu[2] = 2.60;
    prob->n_nucl = 1;

    /* Material 0: pure U-235 */
    prob->mat[0].n_nuc = 1;
    prob->mat[0].nuc_id[0] = 0;
    prob->mat[0].frac[0] = 1.0;
    prob->n_mat = 1;

    /* Defaults */
    prob->k_eff = 1.0;
    prob->n_part = 100;
    prob->n_batch = 5;
    prob->n_skip = 2;

    rn_init(&prob->rng, 42);
    tl_init(&prob->tl_flux);
    tl_init(&prob->tl_fiss);
}

/* ---- Particle movement ---- */

static void t_tp_move(void)
{
    mo_part_t p;
    memset(&p, 0, sizeof(p));
    p.x = 1.0; p.y = 2.0; p.z = 3.0;
    p.u = 1.0; p.v = 0.0; p.w = 0.0;

    tp_move(&p, 5.0);

    CHFLT(p.x, 6.0, 1e-10);
    CHFLT(p.y, 2.0, 1e-10);
    CHFLT(p.z, 3.0, 1e-10);
    PASS();
}
TH_REG("trans", t_tp_move)

/* ---- Single history: particle should eventually die or leak ---- */

static void t_tp_one(void)
{
    static mo_prob_t prob;
    mo_part_t p;
    static mo_site_t fbank[MO_MAX_PART];
    int n_fbank = 0;

    mk_godiva(&prob);

    /* Start at origin, heading +x, 1 MeV */
    memset(&p, 0, sizeof(p));
    p.x = 0.0; p.y = 0.0; p.z = 0.0;
    p.u = 1.0; p.v = 0.0; p.w = 0.0;
    p.E = 1.0e6;
    p.wt = 1.0;
    p.cell = 0;
    p.alive = 1;

    tp_hist(&prob, &p, fbank, &n_fbank);

    /* Particle should be dead after history */
    CHEQ(p.alive, 0);
    PASS();
}
TH_REG("trans", t_tp_one)

/* ---- Multiple histories: check fission bank gets populated ---- */

static void t_tp_fbank(void)
{
    static mo_prob_t prob;
    static mo_site_t fbank[MO_MAX_PART];
    int n_fbank = 0;
    int i;

    mk_godiva(&prob);

    /* Run 100 histories */
    for (i = 0; i < 100; i++) {
        mo_part_t p;
        memset(&p, 0, sizeof(p));
        p.x = 0.0; p.y = 0.0; p.z = 0.0;
        p.u = 2.0 * rn_dbl(&prob.rng) - 1.0;
        p.v = 2.0 * rn_dbl(&prob.rng) - 1.0;
        p.w = 2.0 * rn_dbl(&prob.rng) - 1.0;
        /* Normalise direction */
        double mag = sqrt(p.u * p.u + p.v * p.v + p.w * p.w);
        if (mag > 1e-10) { p.u /= mag; p.v /= mag; p.w /= mag; }
        else { p.u = 1.0; p.v = 0.0; p.w = 0.0; }
        p.E = 1.0e6;
        p.wt = 1.0;
        p.cell = 0;
        p.alive = 1;

        tp_hist(&prob, &p, fbank, &n_fbank);
    }

    /* Should have some fission sites (U-235 is quite fissile) */
    CHECK(n_fbank > 0);

    /* Fission sites should be inside the sphere */
    for (i = 0; i < n_fbank && i < 10; i++) {
        double r2 = fbank[i].x * fbank[i].x
                   + fbank[i].y * fbank[i].y
                   + fbank[i].z * fbank[i].z;
        CHECK(r2 < 8.7407 * 8.7407 + 1.0);  /* inside sphere + tolerance */
    }
    PASS();
}
TH_REG("trans", t_tp_fbank)

/* ---- Tally accumulation ---- */

static void t_tl_basic(void)
{
    mo_taly_t t;
    double mean, sdev;

    tl_init(&t);

    /* Score some track-length contributions into bat[] */
    tl_trak(&t, 0, 1.0, 5.0);   /* wt=1, dist=5 → score 5 */
    tl_trak(&t, 0, 1.0, 3.0);   /* wt=1, dist=3 → score 3 */

    /* Batch accumulator should hold 8 */
    CHFLT(t.bat[0], 8.0, 1e-10);

    /* Region 1 should be zero */
    CHFLT(t.bat[1], 0.0, 1e-10);

    /* After normalisation with n_part=2, mean score = 4 */
    tl_norm(&t, 2);
    tl_stat(&t, 0, 1, &mean, &sdev);
    CHFLT(mean, 4.0, 1e-10);
    PASS();
}
TH_REG("trans", t_tl_basic)
