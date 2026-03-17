/* nd_thrm.c -- Free-gas thermal scattering model
 *
 * Below ~4 eV (specifically 4kT ≈ 0.1 eV at room temp), the
 * thermal motion of the target nucleus matters. A 0.025 eV neutron
 * hitting a stationary U-238 nucleus always loses energy. But the
 * same neutron hitting a U-238 nucleus that's jiggling at 293 K
 * can GAIN energy — the nucleus is moving toward the neutron
 * and imparts a Doppler kick. This is "upscattering" and it's
 * what thermalises neutrons in water-moderated reactors.
 *
 * The free-gas model treats each nucleus as having a Maxwell-
 * Boltzmann velocity distribution at temperature T. It's correct
 * for noble gases and metals, wrong for bound atoms (H in H2O,
 * C in graphite) where chemical binding affects the scattering
 * kernel. For bound atoms, use S(α,β) tables instead.
 *
 * Algorithm: DBRC (Doppler Broadening Rejection Correction)
 * simplified to the standard target-at-rest → free-gas crossover.
 * Sample target velocity from Maxwellian, transform to CM frame,
 * scatter isotropically, transform back to lab.
 *
 * References:
 *   Williams, M. M. R. (1966). The slowing down and
 *   thermalization of neutrons. North-Holland. Chapter 4.
 *
 *   Coveyou, R. R., Cain, V. R., & Yost, K. J. (1966). Adjoint
 *   and importance in Monte Carlo application. Nuclear Science
 *   and Engineering, 27(2), 219-234. (The original free-gas
 *   sampling algorithm.) */

#include "../moa.h"

/* ---- Sample Maxwellian target speed ----
 * The target nucleus has speed v_t drawn from a Maxwell-Boltzmann
 * distribution: P(v) ∝ v² exp(-A v²/(2kT)).
 *
 * We use the standard technique: sample E_target = kT × (-ln(ξ₁)
 * - ln(ξ₂)cos²(πξ₃/2)) for the 3D Maxwellian. This gives the
 * target kinetic energy in the gas frame.
 *
 * Returns target speed in cm/s... just kidding, we work in natural
 * units where speed² ∝ energy/mass. Return √(2·E_t/(A·m_n)). */

double
th_maxw(mo_rng_t *rng, double kT, double A)
{
    double r1, r2, r3, E_t;

    /* Maxwell speed distribution: E_t has mean 3/2 kT.
     * Sampled via sum of exponentials method. */
    r1 = rn_dbl(rng);
    r2 = rn_dbl(rng);
    r3 = rn_dbl(rng);
    if (r1 < 1.0e-30) r1 = 1.0e-30;
    if (r2 < 1.0e-30) r2 = 1.0e-30;

    {
        double cos_val = cos(MO_PI * 0.5 * r3);
        E_t = -kT * (log(r1) + log(r2) * cos_val * cos_val);
    }

    /* Speed = √(2E/m) in neutron mass units where m = A */
    return sqrt(2.0 * E_t / A);
}

/* ---- Free-gas elastic scattering ----
 * The full thermal treatment: sample target velocity from
 * Maxwellian, compute relative velocity, scatter isotropically
 * in the CM frame, transform back to lab.
 *
 * The rejection loop ensures we correctly sample the
 * cross-section-weighted velocity distribution: the probability
 * of interaction is proportional to |v_rel| × σ(E_rel), not
 * just the Maxwellian. For 1/v absorbers this simplifies
 * beautifully; for elastic scatter with resonances, less so.
 * We use the constant-XS approximation (rejection on |v_rel|
 * only) which is fine for thermal energies where elastic XS
 * is roughly constant.
 *
 * Output: E_out (eV), mu_lab (cos of scattering angle in lab). */

void
th_fgas(mo_rng_t *rng, double E, double A, double kT,
        double *E_out, double *mu_lab)
{
    double v_n, v_t, v_rx, v_ry, v_rz;
    double v_rel = 0.0, v_cm;

    /* Neutron speed (natural units: v² = 2E/m, m=1 for neutron) */
    v_n = sqrt(2.0 * E);
    if (v_n < 1.0e-20) v_n = 1.0e-20;

    /* Rejection loop: sample target velocity, accept with
     * probability |v_rel|/v_max where v_max = v_n + 4√(kT/A).
     * The factor 4 covers 99.99% of the Maxwellian tail. */
    double v_max = v_n + 4.0 * sqrt(2.0 * kT / A);

    KA_GUARD(g, 1000);
    while (g--) {
        /* Target velocity: speed from Maxwellian, isotropic direction */
        v_t = th_maxw(rng, kT, A);

        double mu_t = 2.0 * rn_dbl(rng) - 1.0;
        double phi_t = 2.0 * MO_PI * rn_dbl(rng);
        double st_t = sqrt(1.0 - mu_t * mu_t);

        double vtx = v_t * st_t * cos(phi_t);
        double vty = v_t * st_t * sin(phi_t);
        double vtz = v_t * mu_t;

        /* Neutron velocity: assume moving along z for simplicity.
         * (We only need the relative speed and CM energy, not
         * the absolute lab direction, which gets randomised
         * by the isotropic CM scatter anyway.) */
        v_rx = -vtx;
        v_ry = -vty;
        v_rz = v_n - vtz;

        v_rel = sqrt(v_rx * v_rx + v_ry * v_ry + v_rz * v_rz);

        /* Rejection: accept with probability v_rel / v_max */
        if (rn_dbl(rng) * v_max < v_rel) break;
    }

    if (v_rel < 1.0e-20) {
        /* Target and neutron co-moving. Energy barely changes. */
        *E_out = E;
        *mu_lab = 1.0;
        return;
    }

    /* CM velocity magnitude */
    v_cm = v_rel / (A + 1.0);

    /* Isotropic scatter in CM frame */
    double mu_cm = 2.0 * rn_dbl(rng) - 1.0;

    /* Outgoing neutron speed in CM = A/(A+1) × v_rel (elastic) */
    double v_out_cm = A / (A + 1.0) * v_rel;

    /* Transform back to lab: vector addition of CM scatter + CM velocity.
     * Lab energy = ½(v_out_cm² + v_cm² + 2·v_out_cm·v_cm·μ_cm)
     * Plus the target CM velocity contribution. */
    double v_lab_sq = v_out_cm * v_out_cm + v_cm * v_cm
                    + 2.0 * v_out_cm * v_cm * mu_cm;

    *E_out = 0.5 * v_lab_sq;
    if (*E_out < MO_EMIN) *E_out = MO_EMIN;

    /* Lab scattering cosine (approximate — exact requires tracking
     * all three velocity components through the transform, which
     * we don't need because tp_scat sets isotropic direction after
     * thermal scatter anyway) */
    if (v_lab_sq > 1.0e-30 && v_n > 1.0e-20) {
        *mu_lab = (v_out_cm * mu_cm + v_cm)
                / sqrt(v_lab_sq);
        if (*mu_lab > 1.0) *mu_lab = 1.0;
        if (*mu_lab < -1.0) *mu_lab = -1.0;
    } else {
        *mu_lab = 0.0;
    }
}
