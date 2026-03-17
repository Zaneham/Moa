/* nd_rmat.c -- Reich-Moore R-matrix resonance evaluation
 *
 * Full R-matrix (LRF=3) for nuclides where the SLBW approximation
 * fails — primarily Pu-239 and U-235, where overlapping resonances
 * and inter-channel interference make single-level formulas about
 * as accurate as weather forecasts for next month.
 *
 * The R-matrix is constructed per spin group: for each J,π with
 * potentially overlapping resonances, we build the R-matrix in
 * channel space {elastic, fission_a, fission_b}, compute the
 * collision matrix U = (1 - iKR)^(-1)(1 + iKR), and extract
 * cross-sections from |1-U|² (elastic) and |U_fi|² (fission).
 *
 * The 3×3 complex matrix inversion is done analytically using
 * Cramer's rule. Not elegant, but correct and fast. No LAPACK
 * dependency, no dynamic allocation, no wine required.
 *
 * This should fix the ~2% k_eff bias on Jezebel (Pu-239) that
 * SLBW produces by ignoring resonance interference.
 *
 * References:
 *   Lane, A. M., & Thomas, R. G. (1958). R-matrix theory of
 *   nuclear reactions. Reviews of Modern Physics, 30(2), 257-353.
 *
 *   Larson, N. M. (2008). Updated users' guide for SAMMY
 *   (ORNL/TM-9179/R8). Oak Ridge National Laboratory.
 *   Chapter 3: Reich-Moore approximation.
 *
 *   Herman, M., & Trkov, A. (2009). ENDF-6 formats manual
 *   (BNL-90365-2009 Rev. 2). Section 2.2.1.3: Reich-Moore. */

#include "../moa.h"

/* ---- Physical constants (same as nd_res.c) ---- */
#define RM_K2F  4.8261e-8   /* √(2m_n)/(ℏc) squared, 1/(fm²·eV) */

/* ---- Complex number helpers ----
 * Because C99 _Complex is fine until you want to debug it
 * on MSVC where it doesn't exist. Two doubles, multiply by
 * hand, carry on. */

typedef struct { double re, im; } cpx_t;

static cpx_t cx_add(cpx_t a, cpx_t b)
{ return (cpx_t){ a.re + b.re, a.im + b.im }; }

static cpx_t cx_sub(cpx_t a, cpx_t b)
{ return (cpx_t){ a.re - b.re, a.im - b.im }; }

static cpx_t cx_mul(cpx_t a, cpx_t b)
{ return (cpx_t){ a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re }; }

static cpx_t cx_div(cpx_t a, cpx_t b)
{
    double d = b.re * b.re + b.im * b.im;
    if (d < 1.0e-40) return (cpx_t){ 0.0, 0.0 };
    return (cpx_t){ (a.re*b.re + a.im*b.im) / d,
                    (a.im*b.re - a.re*b.im) / d };
}

static double cx_abs2(cpx_t a)
{ return a.re * a.re + a.im * a.im; }

/* ---- 2×2 complex matrix inverse ----
 * M = [a b; c d], M⁻¹ = [d -b; -c a] / det
 * Used for the reduced 2-channel case when fission_b width is zero. */

/* ---- 3×3 complex determinant (Sarrus) ---- */
static cpx_t
det3(cpx_t m[3][3])
{
    cpx_t d = {0, 0};
    /* Sarrus expansion: bounded, no loops needed */
    d = cx_add(d, cx_mul(m[0][0], cx_mul(m[1][1], m[2][2])));
    d = cx_add(d, cx_mul(m[0][1], cx_mul(m[1][2], m[2][0])));
    d = cx_add(d, cx_mul(m[0][2], cx_mul(m[1][0], m[2][1])));
    d = cx_sub(d, cx_mul(m[0][2], cx_mul(m[1][1], m[2][0])));
    d = cx_sub(d, cx_mul(m[0][1], cx_mul(m[1][0], m[2][2])));
    d = cx_sub(d, cx_mul(m[0][0], cx_mul(m[1][2], m[2][1])));
    return d;
}

/* ---- 3×3 complex matrix inverse via Cramer ---- */
static void
inv3(cpx_t m[3][3], cpx_t out[3][3])
{
    cpx_t d = det3(m);
    int i, j;

    /* Cofactor matrix, transposed (adjugate) */
    out[0][0] = cx_sub(cx_mul(m[1][1], m[2][2]), cx_mul(m[1][2], m[2][1]));
    out[0][1] = cx_sub(cx_mul(m[0][2], m[2][1]), cx_mul(m[0][1], m[2][2]));
    out[0][2] = cx_sub(cx_mul(m[0][1], m[1][2]), cx_mul(m[0][2], m[1][1]));
    out[1][0] = cx_sub(cx_mul(m[1][2], m[2][0]), cx_mul(m[1][0], m[2][2]));
    out[1][1] = cx_sub(cx_mul(m[0][0], m[2][2]), cx_mul(m[0][2], m[2][0]));
    out[1][2] = cx_sub(cx_mul(m[0][2], m[1][0]), cx_mul(m[0][0], m[1][2]));
    out[2][0] = cx_sub(cx_mul(m[1][0], m[2][1]), cx_mul(m[1][1], m[2][0]));
    out[2][1] = cx_sub(cx_mul(m[0][1], m[2][0]), cx_mul(m[0][0], m[2][1]));
    out[2][2] = cx_sub(cx_mul(m[0][0], m[1][1]), cx_mul(m[0][1], m[1][0]));

    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            out[i][j] = cx_div(out[i][j], d);
}

/* ---- Evaluate Reich-Moore cross-sections ----
 *
 * For each spin group J, accumulate the R-matrix from all
 * resonances with that J value:
 *   R_cc' = Σ_r (γ_c,r × γ_c',r) / (E_r - E)
 *
 * where γ_c,r = √(Γ_c,r / (2π)). Channel widths:
 *   c=0: elastic (neutron), Γ_n(E) = Γ_n(E_r)·√(E/|E_r|)
 *   c=1: fission_a, Γ_fa (constant)
 *   c=2: fission_b, Γ_fb (constant)
 *
 * Capture is computed as: σ_γ = σ_tot - σ_el - σ_f
 * (it's the "everything else" channel). */

void
rm_eval(const mo_nucl_t *nuc, double E,
        double *sig_t, double *sig_f, double *sig_g, double *sig_el)
{
    double awri, ecm, k2, k, a_fm, rho;
    double pikq, sin_phi, pot;
    double gfac;
    int    r;

    /* Accumulate per-J cross-sections */
    double sum_el = 0.0, sum_f = 0.0, sum_g = 0.0;

    *sig_t = 0.0; *sig_f = 0.0; *sig_g = 0.0; *sig_el = 0.0;

    if (nuc->n_res <= 0 || E <= 0.0) return;

    awri = nuc->awr;
    if (awri < 1.0) return;

    ecm = E * awri / (awri + 1.0);
    if (ecm < 1.0e-20) return;

    k2  = RM_K2F * ecm;
    k   = sqrt(k2);
    a_fm = fabs(nuc->AP) * 10.0;
    rho  = k * a_fm;
    pikq = MO_PI / (k2 * 100.0);

    sin_phi = sin(rho);
    pot = 4.0 * pikq * sin_phi * sin_phi;

    gfac = 2.0 * (2.0 * nuc->SPI + 1.0);
    if (gfac < 1.0) gfac = 1.0;

    /* Process each spin group separately.
     * First, find all unique J values (bounded scan). */
    double j_seen[64];
    int n_jgrp = 0;

    for (r = 0; r < nuc->n_res && r < MO_MAX_RES; r++) {
        double aj = fabs(nuc->res[r].AJ);
        int j, dup = 0;
        for (j = 0; j < n_jgrp; j++) {
            if (fabs(j_seen[j] - aj) < 0.01) { dup = 1; break; }
        }
        if (!dup && n_jgrp < 64) {
            j_seen[n_jgrp++] = aj;
        }
    }

    /* For each spin group, build R-matrix and compute U */
    int jg;
    KA_GUARD(g_jg, 64);
    for (jg = 0; jg < n_jgrp && g_jg--; jg++) {
        double AJ = j_seen[jg];
        double gJ = (2.0 * AJ + 1.0) / gfac;

        /* Build R-matrix: 3×3, channels = {n, fa, fb} */
        cpx_t R[3][3];
        memset(R, 0, sizeof(R));

        KA_GUARD(g_r, MO_MAX_RES);
        for (r = 0; r < nuc->n_res && g_r--; r++) {
            if (fabs(fabs(nuc->res[r].AJ) - AJ) > 0.01) continue;

            double Er   = nuc->res[r].Er;
            double Gn_r = fabs(nuc->res[r].Gn);
            double Gg   = fabs(nuc->res[r].Gg);
            double Gfa  = nuc->res[r].Gfa;
            double Gfb  = nuc->res[r].Gfb;

            /* Energy-dependent neutron width for l=0 */
            double Er_abs = fabs(Er);
            double Gn = (Er_abs > 1.0e-30)
                      ? Gn_r * sqrt(E / Er_abs) : Gn_r;

            /* Half-widths (γ²_c = Γ_c / 2) */
            double gn2 = Gn * 0.5;
            double gfa2 = fabs(Gfa) * 0.5;
            double gfb2 = fabs(Gfb) * 0.5;

            /* Denominator: 1 / (E_r - E - iΓ_γ/2) */
            double re_d = Er - E;
            double im_d = -Gg * 0.5;
            double d2 = re_d * re_d + im_d * im_d;
            if (d2 < 1.0e-40) continue;

            cpx_t inv_d = { re_d / d2, -im_d / d2 };

            /* Sign of fission widths encodes interference */
            double sgn_fa = (Gfa >= 0.0) ? 1.0 : -1.0;
            double sgn_fb = (Gfb >= 0.0) ? 1.0 : -1.0;

            /* R_cc' += γ_c × γ_c' / (E_r - E - iΓ_γ/2) */
            double g[3] = { sqrt(gn2), sgn_fa * sqrt(gfa2),
                           sgn_fb * sqrt(gfb2) };
            int ci, cj;
            for (ci = 0; ci < 3; ci++) {
                for (cj = 0; cj < 3; cj++) {
                    cpx_t term = { g[ci] * g[cj] * inv_d.re,
                                   g[ci] * g[cj] * inv_d.im };
                    R[ci][cj] = cx_add(R[ci][cj], term);
                }
            }
        }

        /* Build (I - iP^½ R P^½) where P₀ = ρ for l=0.
         * For the elastic channel, penetrability = ρ = ka.
         * For fission channels, penetrability = 1 (by convention). */
        cpx_t A_mat[3][3], A_inv[3][3];
        double p_fac[3] = { rho, 1.0, 1.0 };
        int ci, cj;

        for (ci = 0; ci < 3; ci++) {
            for (cj = 0; cj < 3; cj++) {
                double pf = sqrt(p_fac[ci] * p_fac[cj]);
                /* A = I - 2iP^½ R P^½ */
                A_mat[ci][cj].re = -2.0 * pf * (-R[ci][cj].im);
                A_mat[ci][cj].im = -2.0 * pf * R[ci][cj].re;
                if (ci == cj) A_mat[ci][cj].re += 1.0;
            }
        }

        inv3(A_mat, A_inv);

        /* U = A⁻¹ × B, where B = I + 2iP^½ R P^½ (conjugate of what we subtracted) */
        cpx_t B_mat[3][3];
        for (ci = 0; ci < 3; ci++) {
            for (cj = 0; cj < 3; cj++) {
                double pf = sqrt(p_fac[ci] * p_fac[cj]);
                B_mat[ci][cj].re = 2.0 * pf * (-R[ci][cj].im);
                B_mat[ci][cj].im = 2.0 * pf * R[ci][cj].re;
                if (ci == cj) B_mat[ci][cj].re += 1.0;
            }
        }

        cpx_t U[3][3];
        for (ci = 0; ci < 3; ci++) {
            for (cj = 0; cj < 3; cj++) {
                U[ci][cj] = (cpx_t){0, 0};
                int ck;
                for (ck = 0; ck < 3; ck++) {
                    U[ci][cj] = cx_add(U[ci][cj],
                                       cx_mul(A_inv[ci][ck], B_mat[ck][cj]));
                }
            }
        }

        /* Cross-sections from U matrix:
         * σ_el = (π/k²) g_J |1 - U_nn|²  (nn = elastic channel)
         * σ_f  = (π/k²) g_J (|U_nfa|² + |U_nfb|²)
         * σ_g  not directly available — compute from total below */

        /* Phase-shifted U_nn: account for hard-sphere phase */
        cpx_t exp_2iphi = { cos(2.0 * rho), sin(2.0 * rho) };
        cpx_t U_nn_phased = cx_mul(exp_2iphi, U[0][0]);

        cpx_t one_minus = cx_sub((cpx_t){1.0, 0.0}, U_nn_phased);
        double sig_el_j = pikq * gJ * cx_abs2(one_minus);

        double sig_f_j = pikq * gJ * (cx_abs2(U[0][1]) + cx_abs2(U[0][2]));

        /* Total from optical theorem (not from |1 - U|²):
         * σ_tot = (2π/k²) g_J (1 - Re(U_nn_phased)) */
        double sig_t_j = 2.0 * pikq * gJ * (1.0 - U_nn_phased.re);

        /* Capture = total - elastic - fission */
        double sig_g_j = sig_t_j - sig_el_j - sig_f_j;
        if (sig_g_j < 0.0) sig_g_j = 0.0;

        sum_el += sig_el_j;
        sum_f  += sig_f_j;
        sum_g  += sig_g_j;
    }

    *sig_el = pot + sum_el;
    *sig_f  = sum_f;
    *sig_g  = sum_g;
    *sig_t  = *sig_el + *sig_f + *sig_g;

    if (*sig_el < 0.0) *sig_el = 0.0;
}
