/* txs.c -- Cross-section lookup tests
 * Making sure binary search finds what it's looking for.
 * Unlike most of us. */

#include "../moa.h"
#include "tharns.h"

/* Build a simple test nuclide with known data */
static void
mk_nuc(mo_nucl_t *nuc)
{
    memset(nuc, 0, sizeof(*nuc));
    nuc->ZA  = 92235;
    nuc->awr = 235.0;
    nuc->n_pts = 5;

    /* Energy grid (eV) */
    nuc->E[0] = 1.0e-5;  nuc->E[1] = 1.0;
    nuc->E[2] = 100.0;   nuc->E[3] = 1.0e4;
    nuc->E[4] = 1.0e7;

    /* Total xs */
    nuc->xs_t[0] = 100.0;  nuc->xs_t[1] = 50.0;
    nuc->xs_t[2] = 20.0;   nuc->xs_t[3] = 10.0;
    nuc->xs_t[4] = 5.0;

    /* Elastic */
    nuc->xs_el[0] = 10.0;  nuc->xs_el[1] = 10.0;
    nuc->xs_el[2] = 10.0;  nuc->xs_el[3] = 5.0;
    nuc->xs_el[4] = 3.0;

    /* Fission */
    nuc->xs_f[0] = 80.0;   nuc->xs_f[1] = 30.0;
    nuc->xs_f[2] = 5.0;    nuc->xs_f[3] = 2.0;
    nuc->xs_f[4] = 1.0;

    /* Absorption = total - elastic */
    nuc->xs_a[0] = 90.0;   nuc->xs_a[1] = 40.0;
    nuc->xs_a[2] = 10.0;   nuc->xs_a[3] = 5.0;
    nuc->xs_a[4] = 2.0;

    /* Nu-bar */
    nuc->nu[0] = 2.43;     nuc->nu[1] = 2.43;
    nuc->nu[2] = 2.43;     nuc->nu[3] = 2.50;
    nuc->nu[4] = 2.60;
}

/* ---- Exact grid point lookup ---- */

static void t_xs_exact(void)
{
    mo_nucl_t nuc;
    double st, sa, sf, sel, nu;

    mk_nuc(&nuc);
    xs_look(&nuc, 1.0, &st, &sa, &sf, &sel, &nu);

    CHFLT(st,  50.0, 0.01);
    CHFLT(sa,  40.0, 0.01);
    CHFLT(sf,  30.0, 0.01);
    CHFLT(sel, 10.0, 0.01);
    CHFLT(nu,  2.43, 0.001);
    PASS();
}
TH_REG("xs", t_xs_exact)

/* ---- Interpolation between grid points ---- */

static void t_xs_interp(void)
{
    mo_nucl_t nuc;
    double st, sa, sf, sel, nu;

    mk_nuc(&nuc);

    /* Midpoint between E[1]=1.0 (xs_t=50) and E[2]=100.0 (xs_t=20)
     * at E=50.5: f = (50.5-1)/(100-1) ≈ 0.5, so xs_t ≈ 35 */
    xs_look(&nuc, 50.5, &st, &sa, &sf, &sel, &nu);
    CHFLT(st, 35.0, 1.0);  /* approximate */
    PASS();
}
TH_REG("xs", t_xs_interp)

/* ---- Below grid minimum ---- */

static void t_xs_below(void)
{
    mo_nucl_t nuc;
    double st, sa, sf, sel, nu;

    mk_nuc(&nuc);
    xs_look(&nuc, 1.0e-8, &st, &sa, &sf, &sel, &nu);

    /* Should clamp to first point */
    CHFLT(st, 100.0, 0.01);
    PASS();
}
TH_REG("xs", t_xs_below)

/* ---- Above grid maximum ---- */

static void t_xs_above(void)
{
    mo_nucl_t nuc;
    double st, sa, sf, sel, nu;

    mk_nuc(&nuc);
    xs_look(&nuc, 1.0e10, &st, &sa, &sf, &sel, &nu);

    /* Should clamp to last point */
    CHFLT(st, 5.0, 0.01);
    PASS();
}
TH_REG("xs", t_xs_above)

/* ---- Macroscopic XS computation ---- */

static void t_xs_macro(void)
{
    static mo_prob_t prob;
    mo_xs_t xs;

    memset(&prob, 0, sizeof(prob));

    /* One nuclide */
    mk_nuc(&prob.nucl[0]);
    prob.n_nucl = 1;

    /* One material: pure U-235, fraction 1.0 */
    prob.mat[0].n_nuc = 1;
    prob.mat[0].nuc_id[0] = 0;
    prob.mat[0].frac[0] = 1.0;
    prob.n_mat = 1;

    /* One cell: material 0, density 0.04794 at/b-cm */
    prob.cell[0].mat = 0;
    prob.cell[0].rho = 0.04794;
    prob.n_cell = 1;

    /* At E = 1.0 eV: micro xs_t = 50 barns
     * Macro = 0.04794 * 1.0 * 50 = 2.397 /cm */
    xs = xs_macro(&prob, 0, 1.0);
    CHFLT(xs.sig_t, 2.397, 0.01);
    PASS();
}
TH_REG("xs", t_xs_macro)
