/* nd_res.c -- Resonance cross-section reconstruction
 *
 * Single-Level Breit-Wigner (SLBW) approximation of ENDF MF2
 * resonance parameters. At a given energy E in the resolved
 * resonance region, sum over all resonances to compute the
 * cross-sections that the MF3 smooth background doesn't provide.
 *
 * For Reich-Moore evaluated nuclides (LRF=3, e.g. U-235), we
 * approximate by combining the two fission channels into a
 * single width: Γ_f = √(GFA² + GFB²). This loses inter-channel
 * interference information but captures the dominant resonance
 * physics. Full R-matrix would require complex-valued 3×3 matrix
 * inversions per energy point per spin group — correct but
 * expensive and fiddly. SLBW is the "good enough" that gets us
 * from +3.6% bias to something respectable.
 *
 * Cross-section formulas (l=0 only, 0 K, no Doppler):
 *
 *   σ_total = σ_pot + Σ_{J,r} g_J (4π/k²)(Γ_n/Γ)
 *             × [cos(2φ)ψ + sin(2φ)χ]
 *
 *   σ_fission = Σ_{J,r} g_J (π/k²) Γ_n Γ_f / D
 *
 *   σ_capture = Σ_{J,r} g_J (π/k²) Γ_n Γ_γ / D
 *
 *   σ_elastic = σ_total - σ_fission - σ_capture
 *
 * where:
 *   ψ = (Γ/2)² / D,  χ = (E−E_r)(Γ/2) / D,  D = (E−E_r)² + (Γ/2)²
 *   g_J = (2J+1) / (2(2I+1))  (statistical spin factor)
 *   Γ_n(E) = Γ_n(E_r) × √(E/|E_r|)  (energy-dependent, l=0)
 *   φ = ka  (hard sphere phase shift, l=0)
 *   σ_pot = (4π/k²) sin²(ka)  (potential scattering)
 *
 * References:
 *   Duderstadt, J. J., & Hamilton, L. J. (1976). Nuclear reactor
 *   analysis. John Wiley & Sons. Chapter 2.
 *
 *   Herman, M., & Trkov, A. (Eds.). (2009). ENDF-6 formats manual
 *   (BNL-90365-2009 Rev. 2). Sections 2.1–2.4.
 *
 *   MacFarlane, R. E., & Muir, D. W. (2012). The NJOY nuclear
 *   data processing system, version 2012 (LA-UR-12-27079).
 *   Los Alamos National Laboratory.
 */

#include "../moa.h"

/* ---- Physical constants ----
 * Wave number: k = CRSS × √(E_cm)  [1/fm]
 * where E_cm = E × AWRI/(AWRI+1) in eV.
 *
 * CRSS = √(2·m_n) / (ℏc)
 * m_n = 939.56542 MeV, ℏc = 197.32698 MeV·fm
 * CRSS² = 2 × 939.56542e6 / (197.32698e6)² = 4.8261e-8 [1/(fm²·eV)] */

#define RS_K2F  4.8261e-8

/* ---- Evaluate resonance cross-sections at energy E ----
 *
 * Returns the FULL MF2 contribution: potential scattering +
 * resonance peaks. This gets ADDED to the MF3 background
 * (which is ~0 in the resolved region for actinides).
 *
 * sig_t:  total (potential + resonances)
 * sig_f:  fission
 * sig_g:  capture (radiative, γ)
 * sig_el: elastic (= sig_t - sig_f - sig_g)
 *
 * Only handles l=0. Higher partial waves would need
 * Coulomb functions and more coffee. */

void
rs_eval(const mo_nucl_t *nuc, double E,
        double *sig_t, double *sig_f, double *sig_g, double *sig_el)
{
    double awri, ecm, k2, k, a_fm, rho;
    double pikq, sin_phi, pot;
    double cos2p, sin2p;
    double sum_t, sum_f, sum_g;
    double gfac;
    int    r;

    *sig_t = 0.0; *sig_f = 0.0; *sig_g = 0.0; *sig_el = 0.0;

    if (nuc->n_res <= 0 || E <= 0.0) return;

    awri = nuc->awr;
    if (awri < 1.0) return;

    /* Centre-of-mass energy */
    ecm = E * awri / (awri + 1.0);
    if (ecm < 1.0e-20) return;

    /* Wave number k [1/fm] */
    k2 = RS_K2F * ecm;
    k  = sqrt(k2);

    /* Channel radius: ENDF AP is in units of 10⁻¹² cm.
     * 10⁻¹² cm = 10 fm. So a [fm] = AP × 10. */
    a_fm = fabs(nuc->AP) * 10.0;

    /* Dimensionless ka */
    rho = k * a_fm;

    /* π/k² in barns (1 barn = 100 fm²) */
    pikq = MO_PI / (k2 * 100.0);

    /* Hard sphere phase shift for l=0: φ₀ = ρ */
    sin_phi = sin(rho);

    /* Potential scattering: (4π/k²) sin²(φ₀) */
    pot = 4.0 * pikq * sin_phi * sin_phi;

    /* Precompute cos(2φ) and sin(2φ) for interference terms */
    cos2p = cos(2.0 * rho);
    sin2p = sin(2.0 * rho);

    /* g_J denominator: 2(2I+1). Precompute once. */
    gfac = 2.0 * (2.0 * nuc->SPI + 1.0);
    if (gfac < 1.0) gfac = 1.0;

    /* Accumulate resonance contributions */
    sum_t = 0.0;
    sum_f = 0.0;
    sum_g = 0.0;

    for (r = 0; r < nuc->n_res; r++) {
        const mo_res_t *res = &nuc->res[r];
        double Er, AJ, Gn_r, Gg, Gf;
        double gJ, Gn, Gt, dE, D;
        double psi, chi;

        Er   = res->Er;
        AJ   = fabs(res->AJ);
        Gn_r = fabs(res->Gn);
        Gg   = fabs(res->Gg);

        /* Combined fission width.
         * For SLBW (LRF≤2): Gfb=0, so Gf = |Gfa|.
         * For RM (LRF=3): Gf = √(Gfa² + Gfb²). */
        Gf = sqrt(res->Gfa * res->Gfa + res->Gfb * res->Gfb);

        /* Statistical spin factor: g_J = (2J+1) / (2(2I+1)) */
        gJ = (2.0 * AJ + 1.0) / gfac;

        /* Energy-dependent neutron width for l=0:
         * Γ_n(E) = Γ_n(E_r) × √(E / |E_r|)
         * The penetrability P₀ = ka is proportional to √E,
         * so Γ_n scales as P₀(E)/P₀(E_r) = √(E/|E_r|). */
        {
            double Er_abs = fabs(Er);
            Gn = (Er_abs > 1.0e-30)
               ? Gn_r * sqrt(E / Er_abs)
               : Gn_r;
        }

        /* Total width */
        Gt = Gn + Gg + Gf;

        /* Breit-Wigner denominator: (E - E_r)² + (Γ/2)² */
        dE = E - Er;
        D  = dE * dE + 0.25 * Gt * Gt;

        if (D < 1.0e-40) continue;

        /* Line shape functions:
         * ψ = (Γ/2)² / D  (symmetric Lorentzian, peaks at 1)
         * χ = (E−E_r)(Γ/2) / D  (asymmetric, zero at resonance) */
        psi = 0.25 * Gt * Gt / D;
        chi = dE * 0.5 * Gt / D;

        /* Total (from optical theorem):
         * Σ g_J (Γ_n/Γ) [cos(2φ)ψ + sin(2φ)χ] */
        sum_t += gJ * (Gn / Gt) * (cos2p * psi + sin2p * chi);

        /* Fission: g_J Γ_n Γ_f / D */
        sum_f += gJ * Gn * Gf / D;

        /* Capture: g_J Γ_n Γ_γ / D */
        sum_g += gJ * Gn * Gg / D;
    }

    /* Assemble cross-sections */
    *sig_t  = pot + 4.0 * pikq * sum_t;
    *sig_f  = pikq * sum_f;
    *sig_g  = pikq * sum_g;
    *sig_el = *sig_t - *sig_f - *sig_g;

    /* Elastic can go negative near resonance peaks due to the
     * SLBW approximation of overlapping resonances. Clamp it.
     * Full Reich-Moore wouldn't have this problem, but full
     * Reich-Moore would also require a bottle of wine and a
     * whiteboard. */
    if (*sig_el < 0.0) *sig_el = 0.0;
}
