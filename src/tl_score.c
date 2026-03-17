/* tl_score.c -- Tally system
 *
 * Accumulates physics results with the enthusiasm of an accountant
 * at year-end. Track-length and collision estimators for flux,
 * plus fission rate tallies for k-eigenvalue.
 *
 * Statistics: running sum and sum-of-squares because we need to
 * report confidence intervals. Nobody trusts a number without
 * error bars. Especially nuclear engineers.
 *
 * The tally lifecycle, for the uninitiated:
 *   tl_init  — zero everything (start of problem)
 *   tl_zero  — zero bat[] only (start of each batch)
 *   tl_coll/tl_trak/tl_fiss — score into bat[] during transport
 *   tl_norm  — normalize bat[] by n_part, fold into sum[]/sq[]
 *   tl_stat  — compute mean +/- sigma from sum[]/sq[]/n_bat
 *
 * Previous version zeroed sum[]/sq[] every batch, which is rather
 * like balancing the books by burning them. Fixed now.
 *
 * Reference:
 *   Lux, I., & Koblinger, L. (1991). Monte Carlo particle
 *   transport methods: Neutron and photon calculations.
 *   CRC Press. Chapter 7.
 */

#include "../moa.h"

/* ---- Init / zero ---- */

void
tl_init(mo_taly_t *t)
{
    memset(t, 0, sizeof(*t));
}

void
tl_zero(mo_taly_t *t)
{
    /* Zero the per-batch accumulator only. sum[] and sq[] are
     * sacrosanct — they carry cross-batch statistics. Zeroing
     * them was the original sin of this file. */
    int i;
    for (i = 0; i < MO_MAX_REG; i++)
        t->bat[i] = 0.0;
}

/* ---- Collision estimator ----
 * Score weight / sigma_total at each collision site.
 * Unbiased estimator for flux. Higher variance than track-length
 * but useful as a cross-check and for point detectors. */

void
tl_coll(mo_taly_t *t, int reg, double wt, double sig_t)
{
    if (KA_CHK(reg, MO_MAX_REG)) return;
    if (sig_t < 1.0e-30) return;  /* vacuum has no collisions */
    t->bat[reg] += wt / sig_t;
}

/* ---- Track-length estimator ----
 * Score weight * distance for every track segment.
 * Lower variance than collision for volume-averaged flux.
 * The workhorse estimator. Reliable, unglamorous, effective.
 * Like a Toyota Corolla. */

void
tl_trak(mo_taly_t *t, int reg, double wt, double dist)
{
    if (KA_CHK(reg, MO_MAX_REG)) return;
    t->bat[reg] += wt * dist;
}

/* ---- Fission rate tally ----
 * Score nu * sigma_f / sigma_t * weight at each collision.
 * This feeds into the k-eigenvalue estimate. */

void
tl_fiss(mo_taly_t *t, int reg, double wt, double nu,
        double sig_f, double sig_t)
{
    if (KA_CHK(reg, MO_MAX_REG)) return;
    if (sig_t < 1.0e-30) return;
    t->bat[reg] += wt * nu * sig_f / sig_t;
}

/* ---- Batch normalisation ----
 * After each batch, divide bat[] by n_part to get per-particle
 * averages, fold into running sum[] and sq[], then zero bat[].
 * This is how you accumulate statistics without amnesia. */

void
tl_norm(mo_taly_t *t, int n_part)
{
    int i;
    double inv;

    if (n_part <= 0) return;
    inv = 1.0 / (double)n_part;

    for (i = 0; i < MO_MAX_REG; i++) {
        double val = t->bat[i] * inv;
        t->sum[i] += val;
        t->sq[i]  += val * val;
        t->bat[i]  = 0.0;
    }
    t->n_bat++;
}

/* ---- Statistics ----
 * Compute mean and standard deviation of the mean from the
 * running sums. The sdev is the standard error — how confident
 * we are in the mean, not how spread the individual batches are.
 *
 * mean = sum / N
 * sigma_mean = sqrt( (<x^2> - <x>^2) / (N-1) )
 *
 * Yes, N-1, not N. Bessel's correction. Even statisticians
 * don't trust N. */

void
tl_stat(const mo_taly_t *t, int reg, int n_bat,
        double *mean, double *sdev)
{
    *mean = 0.0;
    *sdev = 0.0;

    if (KA_CHK(reg, MO_MAX_REG)) return;
    if (n_bat <= 0) return;

    *mean = t->sum[reg] / (double)n_bat;

    if (n_bat > 1) {
        double var = (t->sq[reg] / (double)n_bat
                    - (*mean) * (*mean))
                   / (double)(n_bat - 1);
        *sdev = (var > 0.0) ? sqrt(var) : 0.0;
    }
}
