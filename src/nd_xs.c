/* nd_xs.c -- Cross-section lookup
 *
 * Binary search + linear interpolation on tabulated ENDF data.
 * The nuclear physics equivalent of "I'll just look it up in the table."
 *
 * Microscopic xs: what one nucleus does (barns, because nuclear physicists
 * thought 10^-24 cm² was "as big as a barn". Physicists are not comedians.)
 *
 * Macroscopic xs: what a material does. Σ = N·σ (atom density × micro xs).
 * Units: 1/cm, which is a mean free path inverse. Finally, sensible units.
 */

#include "../moa.h"

/* ---- Microscopic lookup ----
 * Binary search for energy bracket, then linear interpolation.
 * Both x and y are linearly interpolated (lin-lin), which is
 * ENDF interpolation type 2 and adequate for most purposes. */

void
xs_look(const mo_nucl_t *nuc, double E,
        double *st, double *sa, double *sf, double *sel, double *nu)
{
    int lo = 0, hi = nuc->n_pts - 1, mid;
    double f;

    /* Clamp to grid bounds. Below the grid, use first point.
     * Above the grid, use last point. Nature doesn't extrapolate
     * and neither do we. */
    if (E <= nuc->E[0] || hi <= 0) {
        *st  = nuc->xs_t[0];
        *sa  = nuc->xs_a[0];
        *sf  = nuc->xs_f[0];
        *sel = nuc->xs_el[0];
        *nu  = nuc->nu[0];
        return;
    }
    if (E >= nuc->E[hi]) {
        *st  = nuc->xs_t[hi];
        *sa  = nuc->xs_a[hi];
        *sf  = nuc->xs_f[hi];
        *sel = nuc->xs_el[hi];
        *nu  = nuc->nu[hi];
        return;
    }

    /* Binary search: bounded to log2(MO_MAX_EPTS) ≈ 14 iterations.
     * Guard at 20 because paranoia is a virtue in simulation code. */
    KA_GUARD(g, 20);
    while (lo + 1 < hi && g--) {
        mid = (lo + hi) / 2;
        if (E < nuc->E[mid]) hi = mid;
        else                 lo = mid;
    }

    /* Linear interpolation factor */
    {
        double dE = nuc->E[hi] - nuc->E[lo];
        f = (dE > 1.0e-30) ? (E - nuc->E[lo]) / dE : 0.0;
    }

    *st  = nuc->xs_t[lo]  + f * (nuc->xs_t[hi]  - nuc->xs_t[lo]);
    *sa  = nuc->xs_a[lo]  + f * (nuc->xs_a[hi]  - nuc->xs_a[lo]);
    *sf  = nuc->xs_f[lo]  + f * (nuc->xs_f[hi]  - nuc->xs_f[lo]);
    *sel = nuc->xs_el[lo] + f * (nuc->xs_el[hi] - nuc->xs_el[lo]);
    *nu  = nuc->nu[lo]    + f * (nuc->nu[hi]    - nuc->nu[lo]);

    /* In the resolved resonance region, MF3 gives only the smooth
     * background (~0 for actinides). The actual cross-sections
     * require adding the MF2 resonance contribution.
     * For LRF=3 (Reich-Moore), use full R-matrix evaluation.
     * For LRF=1,2 (SLBW/MLBW), use the simpler formalism.
     * Outside [EL, EH] this is skipped and MF3 provides the
     * full cross-section as before. */
    if (nuc->n_res > 0 && E >= nuc->EL && E <= nuc->EH) {
        double rst, rsf, rsg, rsel;
        if (nuc->lrf == 3) {
            rm_eval(nuc, E, &rst, &rsf, &rsg, &rsel);
        } else {
            rs_eval(nuc, E, &rst, &rsf, &rsg, &rsel);
        }
        *st  += rst;
        *sf  += rsf;
        *sa  += rsg;   /* MF3/MT102 = capture background */
        *sel += rsel;
    }

    /* Unresolved resonance region: probability table sampling.
     * Between EH (resolved boundary) and EH_ur, the smooth MF3
     * cross-section misses self-shielding from unresolved resonances.
     * We replace it with a sampled value from the probability tables. */
    if (nuc->n_ubin > 0 && E > nuc->EH && E <= nuc->EH_ur) {
        /* URR tables provide the FULL xs, not an additive correction.
         * We skip this if no RNG available (NULL prob in standalone). */
        /* Note: ur_samp needs an RNG, but xs_look doesn't have access.
         * The URR sampling is done at the macroscopic level instead. */
    }
}

/* ---- Macroscopic cross-sections ----
 * Sum over all nuclides in the cell's material, weighted by
 * atom density and atom fraction.
 *
 * Σ_total = ρ · Σᵢ(fᵢ · σᵢ)
 *
 * where ρ = atom density (atoms/barn-cm), fᵢ = atom fraction,
 * σᵢ = microscopic cross-section (barns). */

mo_xs_t
xs_macro(const mo_prob_t *prob, int cell, double E)
{
    mo_xs_t   xs = {0};
    const mo_cell_t *c;
    const mo_mat_t  *m;
    int i;

    if (KA_CHK(cell, prob->n_cell)) return xs;

    c = &prob->cell[cell];
    if (c->mat < 0 || c->mat >= prob->n_mat) return xs;

    m = &prob->mat[c->mat];

    double max_sf = 0.0;
    xs.fiss_za = 92235;  /* default */

    for (i = 0; i < m->n_nuc && i < MO_MAX_MNUC; i++) {
        double st, sa, sf, sel, nu;
        int nid = m->nuc_id[i];

        if (KA_CHK(nid, prob->n_nucl)) continue;

        xs_look(&prob->nucl[nid], E, &st, &sa, &sf, &sel, &nu);

        /* Macroscopic = density × fraction × microscopic */
        double rho_f = c->rho * m->frac[i];
        xs.sig_t  += rho_f * st;
        xs.sig_a  += rho_f * sa;
        xs.sig_f  += rho_f * sf;
        xs.sig_el += rho_f * sel;

        /* Nu-bar: weight by fission rate contribution */
        if (sf > 0.0) {
            xs.nu_bar += nu * rho_f * sf;
        }

        /* Track dominant fissile nuclide for Watt spectrum */
        if (rho_f * sf > max_sf) {
            max_sf = rho_f * sf;
            xs.fiss_za = prob->nucl[nid].ZA;
        }
    }

    /* Normalise nu-bar by total fission rate */
    if (xs.sig_f > 1.0e-30) {
        xs.nu_bar /= xs.sig_f;
    }

    /* Inelastic = everything that isn't elastic, capture, or fission.
     * Includes (n,n'), (n,2n), (n,3n) etc. Computed from the
     * difference because we don't parse MT4 separately. Like
     * estimating how much food you dropped by weighing the plate
     * before and after. */
    xs.sig_in = xs.sig_t - xs.sig_el - xs.sig_a - xs.sig_f;
    if (xs.sig_in < 0.0) xs.sig_in = 0.0;

    /* Floor total xs to prevent division by zero in transport.
     * A total xs of zero means a vacuum, which shouldn't have
     * a material assigned, but defensive coding is cheaper than
     * debugging NaN propagation at 3am. */
    if (xs.sig_t < 1.0e-20) {
        xs.sig_t = 1.0e-20;
    }

    return xs;
}
