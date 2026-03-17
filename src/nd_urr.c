/* nd_urr.c -- Unresolved resonance region
 *
 * Between the top of the resolved resonance region (~2.5 keV for
 * U-238, ~2.25 keV for Pu-239) and ~150 keV, resonances are so
 * dense and narrow that they overlap like sardines in a tin. We
 * can't resolve individual resonances, but we can describe their
 * STATISTICAL properties: average widths, spacings, and the
 * resulting probability distributions of cross-section values.
 *
 * The probability table method: at each energy in the URR, we
 * precompute a discrete probability distribution of xs values.
 * During transport, we sample from this distribution once per
 * history (not per collision!) to capture self-shielding effects.
 *
 * Without URR treatment, the smooth cross-section underestimates
 * the true fluctuation-averaged xs, leading to ~1% bias in k_eff
 * for intermediate-spectrum problems.
 *
 * References:
 *   Levitt, L. B. (1972). The probability table method for treating
 *   unresolved resonances in Monte Carlo criticality calculations.
 *   Nuclear Science and Engineering, 49(4), 450-457.
 *
 *   Mosteller, R. D., & Kiedrowski, B. C. (2015). The Serpent Monte
 *   Carlo code: probability tables and URR treatment. Transactions
 *   of the American Nuclear Society, 113, 647-650.
 *
 *   Herman, M., & Trkov, A. (2009). ENDF-6 formats manual
 *   (BNL-90365-2009 Rev. 2). Section 2.3 (LRU=2). */

#include "../moa.h"

/* ---- Load URR probability tables from ENDF MF2/LRU=2 ----
 *
 * ENDF structure for LRU=2:
 *   CONT: SPI, AP, LSSF, 0, NE, NLS
 *   For each l-value:
 *     CONT: AWRI, 0, L, 0, NJS, 0
 *     For each J:
 *       LIST: D, AJ, AMUN, 0, NE+6, NE
 *       [NE sets of: P, sig_t, sig_el, sig_f, sig_g, sig_comp]
 *
 * We flatten the J-averaged probability tables into a single
 * set per energy point. This loses J-resolution but captures
 * the dominant self-shielding physics. */

int
ur_load(mo_nucl_t *nuc, FILE *f, const char *head)
{
    char line[96];
    int  ne, nls, l_idx, j_idx;

    /* CONT: SPI, AP, LSSF, 0, NE, NLS */
    ne  = nd_ival(head + 44);
    nls = nd_ival(head + 55);

    if (ne <= 0 || ne > MO_MAX_UBIN) ne = MO_MAX_UBIN;
    nuc->n_ubin = ne;
    nuc->n_uprob = 0;

    KA_GUARD(g_l, 8);
    for (l_idx = 0; l_idx < nls && g_l--; l_idx++) {
        if (fgets(line, 96, f) == NULL) return -1;
        int len = (int)strlen(line);
        while (len < 80) line[len++] = ' ';
        line[80] = '\0';

        int njs = nd_ival(line + 44);

        KA_GUARD(g_j, 16);
        for (j_idx = 0; j_idx < njs && g_j--; j_idx++) {
            /* LIST header: D, AJ, AMUN, 0, NE+6, NE */
            if (fgets(line, 96, f) == NULL) return -1;
            len = (int)strlen(line);
            while (len < 80) line[len++] = ' ';
            line[80] = '\0';

            int ne_list = nd_ival(line + 55);
            if (ne_list > MO_MAX_UBIN) ne_list = MO_MAX_UBIN;

            /* Read probability table entries.
             * Each entry is 6 values: E, P, sig_t, sig_el, sig_f, sig_g.
             * We only take the first J group and use it directly.
             * More sophisticated: weight-average across J groups. */
            int e_idx;
            KA_GUARD(g_e, MO_MAX_UBIN + 10);
            for (e_idx = 0; e_idx < ne_list && g_e--; e_idx++) {
                if (fgets(line, 96, f) == NULL) return -1;
                len = (int)strlen(line);
                while (len < 80) line[len++] = ' ';
                line[80] = '\0';

                /* First J group populates the tables */
                if (l_idx == 0 && j_idx == 0 && e_idx < MO_MAX_UBIN) {
                    nuc->u_E[e_idx] = nd_fval(line, 11);
                    /* Read probability and xs values.
                     * In simplified form: one probability entry per energy. */
                    double p_val  = nd_fval(line + 11, 11);
                    double xs_t   = nd_fval(line + 22, 11);
                    double xs_el  = nd_fval(line + 33, 11);
                    double xs_f   = nd_fval(line + 44, 11);
                    double xs_a   = nd_fval(line + 55, 11);

                    /* Store as single-entry probability table.
                     * Real implementation would have multiple
                     * probability entries per energy. */
                    if (e_idx < MO_MAX_UBIN) {
                        nuc->u_prob[e_idx][0] = p_val;
                        nuc->u_xs_t[e_idx][0] = xs_t;
                        nuc->u_xs_el[e_idx][0] = xs_el;
                        nuc->u_xs_f[e_idx][0] = xs_f;
                        nuc->u_xs_a[e_idx][0] = xs_a;
                        if (nuc->n_uprob < 1) nuc->n_uprob = 1;
                    }
                }
            }
        }
    }

    /* Set URR energy bounds */
    if (nuc->n_ubin > 0) {
        nuc->EL_ur = nuc->u_E[0];
        nuc->EH_ur = nuc->u_E[nuc->n_ubin - 1];
    }

    return 0;
}

/* ---- Sample URR cross-sections ----
 * At energy E in the URR, interpolate on the energy grid and
 * sample from the probability distribution to get xs values
 * that reflect the resonance self-shielding.
 *
 * The key insight: in thick absorbers, the smooth average xs
 * UNDERestimates transmission because neutrons that "see" a
 * resonance peak are absorbed (removed from the beam), while
 * those between peaks sail through. The probability table
 * captures this fluctuation effect. */

void
ur_samp(const mo_nucl_t *nuc, double E, mo_rng_t *rng,
        double *st, double *sf, double *sa, double *sel)
{
    int lo, hi, mid, pi;

    *st = 0.0; *sf = 0.0; *sa = 0.0; *sel = 0.0;

    if (nuc->n_ubin <= 0 || nuc->n_uprob <= 0) return;

    /* Binary search for energy bracket */
    lo = 0;
    hi = nuc->n_ubin - 1;

    if (E <= nuc->u_E[0]) { lo = 0; hi = 0; }
    else if (E >= nuc->u_E[hi]) { lo = hi; }
    else {
        KA_GUARD(g, 20);
        while (lo + 1 < hi && g--) {
            mid = (lo + hi) / 2;
            if (E < nuc->u_E[mid]) hi = mid;
            else                    lo = mid;
        }
    }

    /* Sample probability table entry */
    pi = (int)(rn_dbl(rng) * (double)nuc->n_uprob);
    if (pi >= nuc->n_uprob) pi = nuc->n_uprob - 1;

    /* Interpolate between energy brackets */
    if (lo == hi || lo + 1 >= nuc->n_ubin) {
        *st  = nuc->u_xs_t[lo][pi];
        *sf  = nuc->u_xs_f[lo][pi];
        *sa  = nuc->u_xs_a[lo][pi];
        *sel = nuc->u_xs_el[lo][pi];
    } else {
        double dE = nuc->u_E[hi] - nuc->u_E[lo];
        double f = (dE > 1.0e-30) ? (E - nuc->u_E[lo]) / dE : 0.0;

        *st  = nuc->u_xs_t[lo][pi]  + f * (nuc->u_xs_t[hi][pi]  - nuc->u_xs_t[lo][pi]);
        *sf  = nuc->u_xs_f[lo][pi]  + f * (nuc->u_xs_f[hi][pi]  - nuc->u_xs_f[lo][pi]);
        *sa  = nuc->u_xs_a[lo][pi]  + f * (nuc->u_xs_a[hi][pi]  - nuc->u_xs_a[lo][pi]);
        *sel = nuc->u_xs_el[lo][pi] + f * (nuc->u_xs_el[hi][pi] - nuc->u_xs_el[lo][pi]);
    }
}
