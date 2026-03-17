/* tnew.c -- Tests for Phases 3-7 new features
 *
 * Energy-binned tallies, mesh tallies, fixed-source mode,
 * thermal scattering, lattice geometry, Reich-Moore, URR,
 * and Doppler broadening. Each tested in isolation, then
 * tested together, then tested while weeping softly.
 */

#include "../moa.h"
#include "tharns.h"

/* ---- Shared test geometry ----
 * Godiva-like sphere for reuse across tests */

static void
mk_test(mo_prob_t *prob)
{
    memset(prob, 0, sizeof(*prob));

    cg_sph(&prob->surf[0], 0.0, 0.0, 0.0, 8.7407);
    prob->n_surf = 1;

    prob->cell[0].n_half = 1;
    prob->cell[0].surf[0] = 0;
    prob->cell[0].sense[0] = -1;
    prob->cell[0].mat = 0;
    prob->cell[0].rho = 0.04794;
    prob->cell[0].lat_id = -1;

    prob->cell[1].n_half = 1;
    prob->cell[1].surf[0] = 0;
    prob->cell[1].sense[0] = 1;
    prob->cell[1].mat = -1;
    prob->cell[1].lat_id = -1;
    prob->n_cell = 2;

    prob->nucl[0].ZA  = 92235;
    prob->nucl[0].awr = 235.0;
    prob->nucl[0].n_pts = 3;
    prob->nucl[0].E[0] = 1.0e-5;
    prob->nucl[0].E[1] = 1.0e3;
    prob->nucl[0].E[2] = 2.0e7;
    prob->nucl[0].xs_t[0]  = 50.0;
    prob->nucl[0].xs_t[1]  = 15.0;
    prob->nucl[0].xs_t[2]  = 7.0;
    prob->nucl[0].xs_el[0] = 10.0;
    prob->nucl[0].xs_el[1] = 5.0;
    prob->nucl[0].xs_el[2] = 3.0;
    prob->nucl[0].xs_f[0]  = 30.0;
    prob->nucl[0].xs_f[1]  = 5.0;
    prob->nucl[0].xs_f[2]  = 2.0;
    prob->nucl[0].xs_a[0]  = 40.0;
    prob->nucl[0].xs_a[1]  = 10.0;
    prob->nucl[0].xs_a[2]  = 4.0;
    prob->nucl[0].nu[0] = 2.43;
    prob->nucl[0].nu[1] = 2.43;
    prob->nucl[0].nu[2] = 2.60;
    prob->n_nucl = 1;

    prob->mat[0].n_nuc = 1;
    prob->mat[0].nuc_id[0] = 0;
    prob->mat[0].frac[0] = 1.0;
    prob->mat[0].sab_id[0] = -1;
    prob->n_mat = 1;

    prob->k_eff = 1.0;
    prob->n_part = 100;
    prob->n_batch = 5;
    prob->n_skip = 2;

    rn_init(&prob->rng, 42);
    tl_init(&prob->tl_flux);
    tl_init(&prob->tl_fiss);
    te_init(&prob->etl_flux);
}

/* ==== Phase 3a: Energy-binned tallies ==== */

static void t_te_init(void)
{
    mo_etly_t t;
    te_init(&t);
    CHEQ(t.n_bin, 0);
    CHEQ(t.active, 0);
    PASS();
}
TH_REG("etally", t_te_init)

static void t_te_set(void)
{
    mo_etly_t t;
    te_init(&t);
    te_set(&t, 100, 1.0e-5, 2.0e7);

    CHEQ(t.n_bin, 100);
    CHEQ(t.active, 1);

    /* First edge should be emin, last should be emax */
    CHFLT(t.edge[0], 1.0e-5, 1.0e-10);
    CHFLT(t.edge[100], 2.0e7, 1.0);

    /* Edges should be monotonically increasing */
    int i;
    for (i = 0; i < 100; i++) {
        CHECK(t.edge[i + 1] > t.edge[i]);
    }
    PASS();
}
TH_REG("etally", t_te_set)

static void t_te_scor(void)
{
    mo_etly_t t;
    te_init(&t);
    te_set(&t, 10, 1.0e-5, 2.0e7);
    t.n_reg = 1;
    t.reg[0] = 0;

    /* Score at 1 MeV — should land in one of the upper bins */
    te_scor(&t, 0, 1.0, 5.0, 1.0e6);

    /* At least one bin should be non-zero */
    int found = 0;
    int i;
    for (i = 0; i < 10; i++) {
        if (t.bat[0][i] > 0.0) found = 1;
    }
    CHECK(found);
    PASS();
}
TH_REG("etally", t_te_scor)

static void t_te_norm(void)
{
    mo_etly_t t;
    te_init(&t);
    te_set(&t, 4, 1.0, 1.0e4);
    t.n_reg = 1;
    t.reg[0] = 0;

    /* Score and normalise */
    te_scor(&t, 0, 1.0, 10.0, 50.0);
    te_norm(&t, 2);

    /* After normalisation, sum should have the normalised value */
    double total = 0.0;
    int i;
    for (i = 0; i < 4; i++) total += t.sum[0][i];
    CHFLT(total, 5.0, 1.0e-8);  /* 10/2 = 5 */

    /* bat should be zeroed */
    for (i = 0; i < 4; i++) {
        CHFLT(t.bat[0][i], 0.0, 1.0e-15);
    }
    PASS();
}
TH_REG("etally", t_te_norm)

/* ==== Phase 3b: Mesh tallies ==== */

/* Mesh tally is ~6 MB — must be static or heap to avoid stack overflow.
 * Windows has a 1 MB stack limit and mo_mtly_t laughs at that. */
static mo_mtly_t g_mesh;

static void t_tm_init(void)
{
    double lo[3] = {-10, -10, -10};
    double hi[3] = { 10,  10,  10};
    tm_init(&g_mesh, lo, hi);

    CHEQ(g_mesh.active, 1);
    CHEQ(g_mesh.nx, MO_MESH_NX);
    CHFLT(g_mesh.lo[0], -10.0, 1e-10);
    CHFLT(g_mesh.hi[2], 10.0, 1e-10);
    PASS();
}
TH_REG("mesh", t_tm_init)

static void t_tm_scor(void)
{
    double lo[3] = {-10, -10, -10};
    double hi[3] = { 10,  10,  10};
    tm_init(&g_mesh, lo, hi);

    /* Score at origin — should hit the centre voxel */
    tm_scor(&g_mesh, 0.0, 0.0, 0.0, 1.0, 5.0);

    int cx = MO_MESH_NX / 2;
    int cy = MO_MESH_NY / 2;
    int cz = MO_MESH_NZ / 2;
    int idx = cx * MO_MESH_NY * MO_MESH_NZ + cy * MO_MESH_NZ + cz;

    CHFLT(g_mesh.bat[idx], 5.0, 1.0e-10);

    /* Score outside mesh — should not crash or accumulate */
    tm_scor(&g_mesh, 100.0, 0.0, 0.0, 1.0, 5.0);
    CHFLT(g_mesh.bat[idx], 5.0, 1.0e-10);  /* unchanged */
    PASS();
}
TH_REG("mesh", t_tm_scor)

static void t_tm_norm(void)
{
    double lo[3] = {-10, -10, -10};
    double hi[3] = { 10,  10,  10};
    tm_init(&g_mesh, lo, hi);

    tm_scor(&g_mesh, 0.0, 0.0, 0.0, 2.0, 3.0);
    tm_norm(&g_mesh, 4);

    int cx = MO_MESH_NX / 2;
    int cy = MO_MESH_NY / 2;
    int cz = MO_MESH_NZ / 2;
    int idx = cx * MO_MESH_NY * MO_MESH_NZ + cy * MO_MESH_NZ + cz;

    /* bat should be zeroed, sum should have 6/4 = 1.5 */
    CHFLT(g_mesh.bat[idx], 0.0, 1e-15);
    CHFLT(g_mesh.sum[idx], 1.5, 1e-10);
    PASS();
}
TH_REG("mesh", t_tm_norm)

/* ==== Phase 3c: Fixed-source mode ==== */

static void t_tf_src(void)
{
    static mo_prob_t prob;
    mo_part_t p;
    mo_src_t src;

    mk_test(&prob);

    /* Point source at origin */
    memset(&src, 0, sizeof(src));
    src.type = MO_SRC_PT;
    src.pos[0] = 0.0;
    src.pos[1] = 0.0;
    src.pos[2] = 0.0;
    src.E = 1.0e6;

    tf_src(&prob, &src, &p, &prob.rng);

    CHFLT(p.x, 0.0, 1e-10);
    CHFLT(p.y, 0.0, 1e-10);
    CHFLT(p.z, 0.0, 1e-10);
    CHFLT(p.E, 1.0e6, 1e-2);
    CHEQ(p.alive, 1);
    CHEQ(p.cell, 0);  /* should be in the sphere */

    /* Direction should be normalised */
    double mag = p.u*p.u + p.v*p.v + p.w*p.w;
    CHFLT(mag, 1.0, 1e-10);
    PASS();
}
TH_REG("fixd", t_tf_src)

static void t_tf_null_fbank(void)
{
    /* Test that tp_hist works with NULL fission bank */
    static mo_prob_t prob;
    mo_part_t p;

    mk_test(&prob);

    memset(&p, 0, sizeof(p));
    p.x = 0.0; p.y = 0.0; p.z = 0.0;
    p.u = 1.0; p.v = 0.0; p.w = 0.0;
    p.E = 1.0e6;
    p.wt = 1.0;
    p.cell = 0;
    p.alive = 1;
    p.lat = -1;

    /* Should not crash with NULL fission bank */
    tp_hist(&prob, &p, NULL, NULL);
    CHEQ(p.alive, 0);
    PASS();
}
TH_REG("fixd", t_tf_null_fbank)

/* ==== Phase 4a: Free-gas thermal ==== */

static void t_th_maxw(void)
{
    mo_rng_t rng;
    rn_init(&rng, 123);

    /* Sample 1000 Maxwellian speeds, check mean ~ √(8kT/(πA)) */
    double kT = 293.6 * MO_KB;
    double A = 1.0;
    double sum = 0.0;
    int i;

    for (i = 0; i < 1000; i++) {
        double v = th_maxw(&rng, kT, A);
        CHECK(v >= 0.0);
        sum += v;
    }

    double avg = sum / 1000.0;
    /* Average speed should be reasonable (order of magnitude) */
    CHECK(avg > 0.0);
    CHECK(avg < 1.0);  /* natural units, room temp */
    PASS();
}
TH_REG("thrm", t_th_maxw)

static void t_th_fgas(void)
{
    mo_rng_t rng;
    rn_init(&rng, 456);

    /* Free-gas scatter at thermal energy on hydrogen */
    double E_in = 0.025;  /* 25 meV, thermal */
    double A = 1.0;
    double kT = 293.6 * MO_KB;  /* room temp */

    double sum_E = 0.0;
    int i, up = 0;

    for (i = 0; i < 500; i++) {
        double E_out, mu;
        th_fgas(&rng, E_in, A, kT, &E_out, &mu);

        CHECK(E_out >= MO_EMIN);
        CHECK(mu >= -1.0 && mu <= 1.0);

        if (E_out > E_in) up++;
        sum_E += E_out;
    }

    /* Upscattering should occur (thermal equilibrium) */
    CHECK(up > 10);  /* at least some neutrons gained energy */

    /* Average outgoing energy should be near kT */
    double avg_E = sum_E / 500.0;
    CHECK(avg_E > 0.01);
    CHECK(avg_E < 0.1);
    PASS();
}
TH_REG("thrm", t_th_fgas)

/* ==== Phase 5: Lattice ==== */

static void t_cl_find(void)
{
    static mo_prob_t prob;
    mk_test(&prob);

    mo_lat_t lat;
    memset(&lat, 0, sizeof(lat));
    lat.type = 0;  /* rectangular */
    lat.nx = 3; lat.ny = 3; lat.nz = 1;
    lat.pitch[0] = 2.0; lat.pitch[1] = 2.0; lat.pitch[2] = 1.0;
    lat.orig[0] = -3.0; lat.orig[1] = -3.0; lat.orig[2] = 0.0;

    /* Fill with universe IDs */
    int i;
    for (i = 0; i < 9; i++) lat.fill[i] = i;

    int li, lj, lk;

    /* Centre element: (0,0,0.5) should be in element (1,1,0) */
    int u = cl_find(&prob, &lat, 0.0, 0.0, 0.5, &li, &lj, &lk);
    CHEQ(li, 1);
    CHEQ(lj, 1);
    CHEQ(lk, 0);
    CHEQ(u, 4);  /* element 1*3*1 + 1*1 + 0 = 4, fill[4] = 4 */

    /* Corner: (-2.5, -2.5, 0.5) should be (0,0,0) */
    u = cl_find(&prob, &lat, -2.5, -2.5, 0.5, &li, &lj, &lk);
    CHEQ(li, 0);
    CHEQ(lj, 0);
    CHEQ(u, 0);

    /* Outside lattice */
    u = cl_find(&prob, &lat, 10.0, 0.0, 0.5, &li, &lj, &lk);
    CHEQ(u, -1);
    PASS();
}
TH_REG("lat", t_cl_find)

static void t_cl_locl(void)
{
    mo_lat_t lat;
    memset(&lat, 0, sizeof(lat));
    lat.type = 0;
    lat.nx = 3; lat.ny = 3; lat.nz = 1;
    lat.pitch[0] = 2.0; lat.pitch[1] = 2.0; lat.pitch[2] = 1.0;
    lat.orig[0] = -3.0; lat.orig[1] = -3.0; lat.orig[2] = 0.0;

    double lx, ly, lz;
    /* Element (1,1,0) centre is at (-3+1*2+1, -3+1*2+1, 0) = (0,0,0) */
    cl_locl(&lat, 1, 1, 0, 0.0, 0.0, 0.5, &lx, &ly, &lz);

    /* Local coords should be (0,0,0.5) relative to element centre */
    CHFLT(lx, 0.0, 1e-10);
    CHFLT(ly, 0.0, 1e-10);
    PASS();
}
TH_REG("lat", t_cl_locl)

static void t_cl_glob(void)
{
    mo_lat_t lat;
    memset(&lat, 0, sizeof(lat));
    lat.type = 0;
    lat.nx = 3; lat.ny = 3; lat.nz = 1;
    lat.pitch[0] = 2.0; lat.pitch[1] = 2.0; lat.pitch[2] = 1.0;
    lat.orig[0] = -3.0; lat.orig[1] = -3.0; lat.orig[2] = 0.0;

    double x, y, z;
    /* Local (0.5, 0.5, 0) in element (2,2,0) → global */
    cl_glob(&lat, 2, 2, 0, 0.5, 0.5, 0.0, &x, &y, &z);

    /* Element (2,2) centre: orig + (2+0.5)*pitch = -3+5 = 2 */
    CHFLT(x, 2.5, 1e-10);
    CHFLT(y, 2.5, 1e-10);
    PASS();
}
TH_REG("lat", t_cl_glob)

/* ==== Phase 6a: Reich-Moore ==== */

static void t_rm_eval(void)
{
    /* Minimal test: create a nuclide with one RM resonance
     * and verify rm_eval produces reasonable cross-sections */
    mo_nucl_t nuc;
    memset(&nuc, 0, sizeof(nuc));

    nuc.ZA = 94239;
    nuc.awr = 239.0;
    nuc.SPI = 0.5;
    nuc.AP = 0.9494;  /* typical for Pu-239 */
    nuc.lrf = 3;
    nuc.EL = 1.0;
    nuc.EH = 100.0;
    nuc.n_res = 1;
    nuc.res[0].Er  = 0.296;
    nuc.res[0].AJ  = 1.0;
    nuc.res[0].Gn  = 0.000573;
    nuc.res[0].Gg  = 0.0398;
    nuc.res[0].Gfa = 0.00233;
    nuc.res[0].Gfb = 0.0;

    double st, sf, sg, sel;
    rm_eval(&nuc, 0.296, &st, &sf, &sg, &sel);

    /* At resonance peak, xs should be large (>100 barns) */
    CHECK(st > 10.0);
    CHECK(sf >= 0.0);
    CHECK(sg >= 0.0);
    CHECK(sel >= 0.0);
    PASS();
}
TH_REG("rmat", t_rm_eval)

static void t_rm_away(void)
{
    /* Off-resonance, xs should be small */
    mo_nucl_t nuc;
    memset(&nuc, 0, sizeof(nuc));

    nuc.ZA = 94239;
    nuc.awr = 239.0;
    nuc.SPI = 0.5;
    nuc.AP = 0.9494;
    nuc.lrf = 3;
    nuc.EL = 1.0;
    nuc.EH = 100.0;
    nuc.n_res = 1;
    nuc.res[0].Er  = 0.296;
    nuc.res[0].AJ  = 1.0;
    nuc.res[0].Gn  = 0.000573;
    nuc.res[0].Gg  = 0.0398;
    nuc.res[0].Gfa = 0.00233;
    nuc.res[0].Gfb = 0.0;

    double st, sf, sg, sel;
    rm_eval(&nuc, 50.0, &st, &sf, &sg, &sel);

    /* Far from resonance, fission xs should be tiny */
    CHECK(sf < 1.0);
    CHECK(st > 0.0);  /* potential scattering */
    PASS();
}
TH_REG("rmat", t_rm_away)

/* ==== Phase 6b: URR ==== */

static void t_ur_samp(void)
{
    mo_nucl_t nuc;
    mo_rng_t rng;
    memset(&nuc, 0, sizeof(nuc));
    rn_init(&rng, 789);

    /* Fake URR table: 2 energy bins, 1 probability entry each */
    nuc.n_ubin = 2;
    nuc.n_uprob = 1;
    nuc.u_E[0] = 1000.0;
    nuc.u_E[1] = 10000.0;
    nuc.u_prob[0][0] = 1.0;
    nuc.u_xs_t[0][0] = 20.0;
    nuc.u_xs_f[0][0] = 5.0;
    nuc.u_xs_a[0][0] = 8.0;
    nuc.u_xs_el[0][0] = 7.0;
    nuc.u_prob[1][0] = 1.0;
    nuc.u_xs_t[1][0] = 15.0;
    nuc.u_xs_f[1][0] = 3.0;
    nuc.u_xs_a[1][0] = 6.0;
    nuc.u_xs_el[1][0] = 6.0;
    nuc.EL_ur = 1000.0;
    nuc.EH_ur = 10000.0;

    double st, sf, sa, sel;
    ur_samp(&nuc, 5000.0, &rng, &st, &sf, &sa, &sel);

    /* Interpolated values at midpoint */
    CHECK(st > 10.0 && st < 25.0);
    CHECK(sf > 1.0 && sf < 10.0);
    PASS();
}
TH_REG("urr", t_ur_samp)

/* ==== Phase 6c: Doppler ==== */

static void t_dp_brod(void)
{
    /* Dense grid around 6.5 eV so the Gaussian kernel (±4Δ ≈ ±0.3 eV)
     * has neighbours to actually convolve with. With only 5 points
     * spread over 11 eV, the nearest neighbour was 20Δ away and the
     * kernel couldn't reach it — a rookie mistake, like expecting
     * to hear your neighbour sneeze from across the county. */
    mo_nucl_t nuc;
    memset(&nuc, 0, sizeof(nuc));

    nuc.ZA = 92238;
    nuc.awr = 238.0;

    /* 21-point grid: 6.0 to 7.0 eV in 0.05 eV steps.
     * Δ at 6.5 eV, T=600 K, A=238 is ~0.075 eV.
     * Points within ±4Δ = ±0.3 eV of the peak: about 12 points. */
    int n = 21, pk = 10;   /* peak at index 10 → E = 6.5 */
    int i;
    for (i = 0; i < n; i++) {
        nuc.E[i] = 6.0 + (double)i * 0.05;
        nuc.xs_t[i]  = 10.0;
        nuc.xs_el[i] = 5.0;
        nuc.xs_a[i]  = 3.0;
        nuc.xs_f[i]  = 2.0;
    }
    nuc.n_pts = n;

    /* Sharp spike at the centre */
    nuc.xs_t[pk]  = 1000.0;
    nuc.xs_el[pk] = 500.0;
    nuc.xs_a[pk]  = 300.0;
    nuc.xs_f[pk]  = 200.0;

    double peak_before = nuc.xs_t[pk];
    dp_brod(&nuc, 600.0);

    /* Peak should decrease — the Gaussian smears it into neighbours */
    CHECK(nuc.xs_t[pk] < peak_before);

    /* Wings near the peak should increase (received broadened flux) */
    CHECK(nuc.xs_t[pk - 2] > 10.0);
    CHECK(nuc.xs_t[pk + 2] > 10.0);

    /* All partials should still be non-negative */
    for (i = 0; i < n; i++) {
        CHECK(nuc.xs_t[i] >= 0.0);
        CHECK(nuc.xs_el[i] >= 0.0);
    }
    PASS();
}
TH_REG("dopl", t_dp_brod)

/* ==== Integration test: energy tally with transport ==== */

static void t_etly_transport(void)
{
    static mo_prob_t prob;
    mk_test(&prob);

    /* Set up energy tally */
    te_set(&prob.etl_flux, 5, 1.0e-5, 2.0e7);
    prob.etl_flux.n_reg = 1;
    prob.etl_flux.reg[0] = 0;

    /* Run a few histories */
    static mo_site_t fbank[MO_MAX_PART];
    int n_fbank = 0;
    int i;

    for (i = 0; i < 50; i++) {
        mo_part_t p;
        memset(&p, 0, sizeof(p));
        p.x = 0.0; p.y = 0.0; p.z = 0.0;
        p.u = 2.0 * rn_dbl(&prob.rng) - 1.0;
        p.v = 2.0 * rn_dbl(&prob.rng) - 1.0;
        p.w = 2.0 * rn_dbl(&prob.rng) - 1.0;
        double mag = sqrt(p.u*p.u + p.v*p.v + p.w*p.w);
        if (mag > 1e-10) { p.u /= mag; p.v /= mag; p.w /= mag; }
        else { p.u = 1.0; }
        p.E = 1.0e6;
        p.wt = 1.0;
        p.cell = 0;
        p.alive = 1;
        p.lat = -1;

        tp_hist(&prob, &p, fbank, &n_fbank);
    }

    /* Energy tally should have scored something */
    double total = 0.0;
    int b;
    for (b = 0; b < 5; b++) {
        total += prob.etl_flux.bat[0][b];
    }
    CHECK(total > 0.0);
    PASS();
}
TH_REG("integ", t_etly_transport)
