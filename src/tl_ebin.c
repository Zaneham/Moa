/* tl_ebin.c -- Energy-binned tallies
 *
 * Score flux in discrete energy bins because "the spectrum is
 * thermal" doesn't cut it when your supervisor wants to see the
 * 1/E epithermal slope and the fission peak at 2 MeV.
 *
 * Default grid is log-uniform from MO_EMIN to MO_EMAX. Users
 * can override with CUSTOM edges if they know what they're doing
 * (they don't, but we let them anyway).
 *
 * Lifecycle mirrors tl_score.c: init → zero → score → norm → stat.
 * Except now there's an extra dimension, because one dimension
 * of suffering was insufficient.
 *
 * Reference:
 *   Lewis, E. E., & Miller, W. F. (1984). Computational methods
 *   of neutron transport. John Wiley & Sons. Chapter 3. */

#include "../moa.h"

/* ---- Init ---- */

void
te_init(mo_etly_t *t)
{
    memset(t, 0, sizeof(*t));
}

/* ---- Set up energy bins ----
 * Log-uniform grid: edges at E_i = emin * (emax/emin)^(i/n).
 * This gives equal lethargy widths, which is the natural scale
 * for neutron slowing down. Linear bins would put 99% of the
 * resolution in the fast region where nothing interesting happens. */

void
te_set(mo_etly_t *t, int n_bin, double emin, double emax)
{
    int i;

    if (n_bin <= 0) return;
    if (n_bin > MO_MAX_EBIN) n_bin = MO_MAX_EBIN;
    if (emin <= 0.0) emin = MO_EMIN;
    if (emax <= emin) emax = MO_EMAX;

    t->n_bin = n_bin;
    t->active = 1;

    double ratio = emax / emin;
    double inv_n = 1.0 / (double)n_bin;

    for (i = 0; i <= n_bin; i++) {
        t->edge[i] = emin * pow(ratio, (double)i * inv_n);
    }
}

/* ---- Score into energy bin ----
 * Binary search on edge[] to find which bin E falls in, then
 * accumulate wt*dist as track-length estimator. Bounded by
 * KA_GUARD because we're civilised. */

void
te_scor(mo_etly_t *t, int reg, double wt, double dist, double E)
{
    int lo, hi, mid, ri;

    if (!t->active) return;
    if (t->n_bin <= 0) return;

    /* Find which registered region this is */
    ri = -1;
    KA_GUARD(g1, MO_MAX_REG);
    for (lo = 0; lo < t->n_reg && g1--; lo++) {
        if (t->reg[lo] == reg) { ri = lo; break; }
    }
    if (ri < 0) return;

    /* Below or above the grid — don't score */
    if (E < t->edge[0] || E >= t->edge[t->n_bin]) return;

    /* Binary search for the bin */
    lo = 0;
    hi = t->n_bin;
    KA_GUARD(g2, 20);
    while (lo + 1 < hi && g2--) {
        mid = (lo + hi) / 2;
        if (E < t->edge[mid]) hi = mid;
        else                   lo = mid;
    }

    if (KA_CHK(ri, MO_MAX_REG)) return;
    if (KA_CHK(lo, MO_MAX_EBIN)) return;

    t->bat[ri][lo] += wt * dist;
}

/* ---- Normalise batch ----
 * Same pattern as tl_norm: divide bat by n_part, fold into
 * sum/sq, zero bat. Repeat until statistically significant
 * or until you run out of patience. */

void
te_norm(mo_etly_t *t, int n_part)
{
    int r, b;
    double inv;

    if (!t->active || n_part <= 0) return;
    inv = 1.0 / (double)n_part;

    for (r = 0; r < t->n_reg; r++) {
        for (b = 0; b < t->n_bin; b++) {
            double val = t->bat[r][b] * inv;
            t->sum[r][b] += val;
            t->sq[r][b]  += val * val;
            t->bat[r][b]  = 0.0;
        }
    }
    t->n_bat++;
}

/* ---- Statistics for one bin ---- */

void
te_stat(const mo_etly_t *t, int reg, int bin, int n_bat,
        double *mean, double *sdev)
{
    *mean = 0.0;
    *sdev = 0.0;

    if (KA_CHK(reg, MO_MAX_REG)) return;
    if (KA_CHK(bin, MO_MAX_EBIN)) return;
    if (n_bat <= 0) return;

    *mean = t->sum[reg][bin] / (double)n_bat;

    if (n_bat > 1) {
        double var = (t->sq[reg][bin] / (double)n_bat
                    - (*mean) * (*mean))
                   / (double)(n_bat - 1);
        *sdev = (var > 0.0) ? sqrt(var) : 0.0;
    }
}

/* ---- Print energy spectrum ----
 * Tabulate each bin's flux and relative error. The output that
 * makes reviewers nod approvingly before asking for more bins. */

void
te_prnt(const mo_etly_t *t, int n_bat)
{
    int r, b;

    if (!t->active || n_bat <= 0) return;

    printf("\n     ---- Energy-Binned Flux (%d bins, %d active batches) ----\n",
           t->n_bin, n_bat);

    for (r = 0; r < t->n_reg; r++) {
        printf("     Region %d (cell %d):\n", r, t->reg[r]);
        printf("     %12s  %12s  %14s  %12s\n",
               "E_lo (eV)", "E_hi (eV)", "flux", "+/- sigma");

        for (b = 0; b < t->n_bin; b++) {
            double mean, sdev;
            te_stat(t, r, b, n_bat, &mean, &sdev);
            if (mean < 1.0e-30 && sdev < 1.0e-30) continue;

            printf("     %12.4e  %12.4e  %14.5e  %12.3e\n",
                   t->edge[b], t->edge[b + 1], mean, sdev);
        }
    }
}
