/* nd_dopl.c -- Doppler broadening of cross-sections
 *
 * At non-zero temperature, target nuclei are in thermal motion.
 * The cross-section a neutron "sees" is not σ(E) but a convolution
 * of σ with the Maxwell-Boltzmann velocity distribution of the
 * target. Narrow resonances get smeared out (broadened), and their
 * peak values decrease while their wings increase.
 *
 * This is the ψ/χ formalism (SIGMA1 method): each resonance
 * contribution is convolved with the Doppler kernel, parameterised
 * by the Doppler width Δ = √(4kTE_r/A).
 *
 * We apply broadening once at init time to the pointwise MF3
 * cross-sections. This is approximate (proper broadening should
 * operate on the resonance parameters, not the reconstructed xs),
 * but captures the dominant effect: resonance peaks shrink, dips
 * fill in, and the problem runs hotter.
 *
 * The alternative — on-the-fly broadening at every xs lookup —
 * is 100x slower and reserved for people with cluster access.
 *
 * References:
 *   Cullen, D. E. (2010). PREPRO 2010 — ENDF/B pre-processing
 *   codes (IAEA-NDS-39 Rev. 14). Sections on SIGMA1.
 *
 *   Bethe, H. A., & Placzek, G. (1937). Resonance effects in
 *   nuclear processes. Physical Review, 51(6), 450-484.
 *   (The original Doppler broadening theory.) */

#include "../moa.h"

/* ---- Doppler kernel (psi function approximation) ----
 *
 * The broadened cross-section at energy E is:
 *   σ̃(E) = (1/√π) ∫ σ(E') exp(-(E'-E)²/Δ²) d(E'/Δ)
 *
 * where Δ = √(4kT·E/A) is the Doppler width.
 *
 * For pointwise data, we convolve the tabulated xs with a
 * Gaussian kernel. Integration by trapezoidal rule over the
 * energy grid, bounded to ±4Δ (covers 99.994% of the Gaussian). */

static double
dp_kern(double x)
{
    /* Gaussian: exp(-x²), already normalised by caller */
    return exp(-x * x);
}

/* ---- Broaden all cross-sections of a nuclide ----
 *
 * For each energy point E_i in the nuclide's grid, replace
 * σ(E_i) with the Doppler-convolved value at temperature T.
 * We broaden total, elastic, absorption, and fission.
 *
 * Called once after nd_load when problem temperature > 0 K.
 * Modifies the nuclide in-place, so the original 0K data is
 * lost. If you need both, load the nuclide twice. We don't,
 * because memory is finite and the resolved region already
 * uses 15 MB. */

void
dp_brod(mo_nucl_t *nuc, double T)
{
    int i, j;
    double kT;

    if (T < 1.0) return;  /* absolute zero is close enough */
    if (nuc->n_pts <= 0) return;
    if (nuc->awr < 1.0) return;

    kT = T * MO_KB;

    /* Temporary arrays for broadened values.
     * Static because 80KB on the stack makes Windows nervous. */
    static double br_t[MO_MAX_EPTS];
    static double br_a[MO_MAX_EPTS];
    static double br_f[MO_MAX_EPTS];
    static double br_el[MO_MAX_EPTS];

    for (i = 0; i < nuc->n_pts; i++) {
        double E = nuc->E[i];
        if (E < 1.0e-20) {
            br_t[i]  = nuc->xs_t[i];
            br_a[i]  = nuc->xs_a[i];
            br_f[i]  = nuc->xs_f[i];
            br_el[i] = nuc->xs_el[i];
            continue;
        }

        /* Doppler width at this energy */
        double delta = sqrt(4.0 * kT * E / nuc->awr);
        if (delta < 1.0e-10) {
            br_t[i]  = nuc->xs_t[i];
            br_a[i]  = nuc->xs_a[i];
            br_f[i]  = nuc->xs_f[i];
            br_el[i] = nuc->xs_el[i];
            continue;
        }

        double inv_d = 1.0 / delta;
        double sum_t = 0.0, sum_a = 0.0, sum_f = 0.0, sum_el = 0.0;
        double sum_w = 0.0;

        /* Convolve over neighbouring grid points within ±4Δ */
        KA_GUARD(g, MO_MAX_EPTS);
        for (j = 0; j < nuc->n_pts && g--; j++) {
            double Ep = nuc->E[j];
            double x = (Ep - E) * inv_d;

            /* Skip points outside ±4Δ */
            if (x > 4.0 || x < -4.0) continue;

            double w = dp_kern(x);

            /* Trapezoidal weight: half-width of the surrounding interval */
            double dE = 0.0;
            if (j > 0 && j < nuc->n_pts - 1) {
                dE = (nuc->E[j + 1] - nuc->E[j - 1]) * 0.5;
            } else if (j == 0 && nuc->n_pts > 1) {
                dE = nuc->E[1] - nuc->E[0];
            } else if (j == nuc->n_pts - 1 && nuc->n_pts > 1) {
                dE = nuc->E[j] - nuc->E[j - 1];
            }

            double wdE = w * dE * inv_d;
            sum_t  += nuc->xs_t[j]  * wdE;
            sum_a  += nuc->xs_a[j]  * wdE;
            sum_f  += nuc->xs_f[j]  * wdE;
            sum_el += nuc->xs_el[j] * wdE;
            sum_w  += wdE;
        }

        /* Normalise by total kernel weight */
        if (sum_w > 1.0e-30) {
            br_t[i]  = sum_t / sum_w;
            br_a[i]  = sum_a / sum_w;
            br_f[i]  = sum_f / sum_w;
            br_el[i] = sum_el / sum_w;
        } else {
            br_t[i]  = nuc->xs_t[i];
            br_a[i]  = nuc->xs_a[i];
            br_f[i]  = nuc->xs_f[i];
            br_el[i] = nuc->xs_el[i];
        }
    }

    /* Copy broadened values back */
    memcpy(nuc->xs_t,  br_t,  (size_t)nuc->n_pts * sizeof(double));
    memcpy(nuc->xs_a,  br_a,  (size_t)nuc->n_pts * sizeof(double));
    memcpy(nuc->xs_f,  br_f,  (size_t)nuc->n_pts * sizeof(double));
    memcpy(nuc->xs_el, br_el, (size_t)nuc->n_pts * sizeof(double));

    printf("moa: Doppler broadened ZA=%d at T=%.1f K (kT=%.4f eV)\n",
           nuc->ZA, T, kT);
}
