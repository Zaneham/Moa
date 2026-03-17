/* tp_loop.c -- Transport loop
 *
 * The heart of Monte Carlo: follow one neutron through a geometry
 * until it leaks out, loses Russian roulette, or we lose patience.
 *
 * Uses survival biasing (implicit capture): instead of randomly
 * killing neutrons on absorption, we reduce their weight and let
 * them scatter forever. They only die from leakage, energy cutoff,
 * or Russian roulette when their weight drops below MO_WMIN.
 *
 * This is what MCNP does by default and it reduces variance by
 * 2-4x for free. The neutron doesn't know it's been implicitly
 * captured — it just feels lighter and keeps going.
 *
 * References:
 *   Lewis, E. E., & Miller, W. F. (1984). Computational methods
 *   of neutron transport. John Wiley & Sons. Chapter 2.
 *
 *   Brown, F. B. (2009). Fundamentals of Monte Carlo particle
 *   transport (LA-UR-05-4983 Rev.). Los Alamos National Laboratory.
 */

#include "../moa.h"

/* ---- OpenMP thread-local state ----
 * Each neutron history is independent — embarrassingly parallel,
 * like cats at a convention. We privatise the RNG and tallies
 * per thread; geometry and cross-section lookups are read-only.
 *
 * Energy tallies and mesh tallies use shared prob copies.
 * The resulting += race is benign: two threads occasionally
 * score the same bin, losing one contribution. The statistical
 * impact is negligible vs Monte Carlo noise — like worrying
 * about your umbrella in a tsunami. */

#ifdef _OPENMP
#include <omp.h>

static mo_rng_t   g_trng[MO_MAX_THRD];
static mo_taly_t  g_tflx[MO_MAX_THRD];
static mo_taly_t  g_tfis[MO_MAX_THRD];
static int        g_omp;          /* nonzero = parallel region active */
static int        g_fbmax = MO_MAX_PART;

#define TP_RNG(p)   (g_omp ? &g_trng[omp_get_thread_num()] : &(p)->rng)
#define TP_FLX(p)   (g_omp ? &g_tflx[omp_get_thread_num()] : &(p)->tl_flux)
#define TP_FIS(p)   (g_omp ? &g_tfis[omp_get_thread_num()] : &(p)->tl_fiss)
#define TP_FBMAX    g_fbmax

#else  /* serial */

#define TP_RNG(p)   (&(p)->rng)
#define TP_FLX(p)   (&(p)->tl_flux)
#define TP_FIS(p)   (&(p)->tl_fiss)
#define TP_FBMAX    MO_MAX_PART

#endif

/* ---- Move particle ----
 * Advance position along direction by distance d. */

void
tp_move(mo_part_t *p, double d)
{
    p->x += d * p->u;
    p->y += d * p->v;
    p->z += d * p->w;
}

/* ---- Isotropic direction ----
 * Sample a random direction uniformly on the unit sphere.
 * Uses the standard cos(θ)/φ method.
 *
 * μ = cos(θ) = 2ξ₁ - 1    (uniform on [-1, 1])
 * φ = 2π·ξ₂               (uniform on [0, 2π])
 * then: u = √(1-μ²)·cos(φ), v = √(1-μ²)·sin(φ), w = μ */

static void
tp_idir(mo_rng_t *rng, mo_part_t *p)
{
    double mu  = 2.0 * rn_dbl(rng) - 1.0;
    double phi = 2.0 * MO_PI * rn_dbl(rng);
    double st  = sqrt(1.0 - mu * mu);

    p->u = st * cos(phi);
    p->v = st * sin(phi);
    p->w = mu;
}

/* ---- Average scattering cosine lookup ----
 * Binary search + lerp on the μ̄(E) table parsed from MF4/MT2.
 * Returns the average CM scattering cosine at incident energy E.
 *
 * If the nuclide has no angular data (n_mu == 0), returns 0.0 —
 * isotropic, which is what we had before and what you get for
 * light nuclei where compound nucleus scattering genuinely
 * doesn't care about direction. For heavy nuclei without data,
 * it's wrong, but at least it's conservatively wrong. */

static double
tp_mubar(const mo_nucl_t *nuc, double E)
{
    int lo, hi, mid;

    if (nuc->n_mu <= 0) return 0.0;
    if (E <= nuc->mu_E[0]) return nuc->mu_bar[0];

    hi = nuc->n_mu - 1;
    if (E >= nuc->mu_E[hi]) return nuc->mu_bar[hi];

    lo = 0;
    KA_GUARD(g, 20);
    while (lo + 1 < hi && g--) {
        mid = (lo + hi) / 2;
        if (E < nuc->mu_E[mid]) hi = mid;
        else                     lo = mid;
    }

    double dE = nuc->mu_E[hi] - nuc->mu_E[lo];
    if (dE < 1.0e-30) return nuc->mu_bar[lo];

    double f = (E - nuc->mu_E[lo]) / dE;
    return nuc->mu_bar[lo] + f * (nuc->mu_bar[hi] - nuc->mu_bar[lo]);
}

/* ---- Scatter (elastic + inelastic) ----
 *
 * At each scatter event, branch between elastic and inelastic
 * proportional to their macroscopic cross-sections.
 *
 * Elastic: isotropic CM scattering, lab energy E'=E·(A²+1+2Aμ)/(A+1)².
 * For heavy nuclei (A≈235), the energy loss per elastic collision
 * is about 0.004%. Neutrons bounce around like ping-pong balls
 * in a shipping container — lots of collisions, very little change.
 *
 * Inelastic: the neutron excites a discrete nuclear level, depositing
 * a specific quantum of energy in the nucleus. We sample uniformly
 * among accessible levels and subtract the excitation energy. Above
 * the highest discrete level, an evaporation spectrum takes over.
 *
 * Both reactions use isotropic lab-frame angular distribution —
 * adequate for compound nucleus reactions on heavy nuclei.
 *
 * References:
 *   Hauser, W., & Feshbach, H. (1952). The inelastic scattering
 *   of neutrons. Physical Review, 87(2), 366-373.
 *
 *   Weisskopf, V. F. (1937). Statistics and nuclear reactions.
 *   Physical Review, 52(4), 295-303. */

static void
tp_scat(mo_prob_t *prob, mo_part_t *p, const mo_xs_t *xs)
{
    const mo_cell_t *c = &prob->cell[p->cell];
    const mo_mat_t  *m = &prob->mat[c->mat];
    double A = 1.0;
    int nuc_idx = -1;
    int i;

    /* Total scatter = elastic + inelastic */
    double sig_scat = xs->sig_el + xs->sig_in;
    if (sig_scat < 1.0e-30) { p->alive = 0; return; }

    /* Branch: elastic vs inelastic.
     * The weight has already been reduced by survival biasing,
     * so this is just about which energy transfer model to use. */
    int is_inel = 0;
    if (xs->sig_in > 1.0e-30) {
        is_inel = (rn_dbl(TP_RNG(prob)) * sig_scat > xs->sig_el);
    }

    /* Pick which nuclide we scattered off: proportional to
     * that nuclide's contribution to the selected reaction */
    {
        double xi  = rn_dbl(TP_RNG(prob));
        double cum = 0.0;
        double target = is_inel ? xs->sig_in : xs->sig_el;

        for (i = 0; i < m->n_nuc; i++) {
            int nid = m->nuc_id[i];
            if (KA_CHK(nid, prob->n_nucl)) continue;

            double st, sa, sf, sel, nu;
            xs_look(&prob->nucl[nid], p->E, &st, &sa, &sf, &sel, &nu);

            double contrib;
            if (is_inel) {
                double s_in = st - sel - sa - sf;
                if (s_in < 0.0) s_in = 0.0;
                contrib = c->rho * m->frac[i] * s_in;
            } else {
                contrib = c->rho * m->frac[i] * sel;
            }
            cum += contrib;

            if (cum >= xi * target) {
                A = prob->nucl[nid].awr;
                nuc_idx = nid;
                break;
            }
        }
    }

    if (is_inel) {
        /* ---- Inelastic scatter: discrete levels + continuum ----
         * Count accessible levels (excitation Q < E), pick one
         * uniformly, subtract the excitation energy. This is crude
         * (real branching ratios aren't uniform) but captures the
         * dominant physics: small energy losses from low-lying
         * discrete levels, not the wildly optimistic evaporation
         * model that thinks every inelastic scatter removes 1.5 MeV.
         *
         * For energies above all discrete levels, or nuclides
         * without parsed level data, fall back to evaporation. */
        const mo_nucl_t *nuc = NULL;
        int n_acc = 0;

        if (nuc_idx >= 0 && !KA_CHK(nuc_idx, prob->n_nucl))
            nuc = &prob->nucl[nuc_idx];

        /* Count accessible discrete levels */
        if (nuc && nuc->n_lev > 0) {
            for (i = 0; i < nuc->n_lev; i++) {
                /* Threshold: E must exceed Q × (A+1)/A for the
                 * reaction to be kinematically allowed.
                 * For A >> 1, this is approximately Q. */
                if (nuc->lev_Q[i] * (A + 1.0) / A < p->E)
                    n_acc++;
            }
        }

        if (n_acc > 0) {
            /* Sample uniformly from accessible levels */
            int pick = (int)(rn_dbl(TP_RNG(prob)) * n_acc);
            if (pick >= n_acc) pick = n_acc - 1;
            double Q = nuc->lev_Q[pick];

            /* Outgoing energy in lab frame (A >> 1 approximation):
             * E' ≈ E - Q × (A+1)/A */
            p->E -= Q * (A + 1.0) / A;
            if (p->E < MO_EMIN) p->E = MO_EMIN;
        } else {
            /* Continuum: evaporation spectrum for high energies
             * or nuclides without discrete level data.
             * P(E') ∝ E' · exp(-E'/T), T = √(E/(A/8)) in eV.
             * Sampled as E' = -T · ln(ξ₁·ξ₂). */
            double T = sqrt(8.0e6 * p->E / A);
            double r1 = rn_dbl(TP_RNG(prob));
            double r2 = rn_dbl(TP_RNG(prob));
            if (r1 < 1.0e-30) r1 = 1.0e-30;
            if (r2 < 1.0e-30) r2 = 1.0e-30;
            double E_out = -T * log(r1 * r2);
            if (E_out >= p->E) E_out = p->E * 0.5;
            p->E = E_out;
        }
    } else {
        /* ---- Thermal branch ----
         * Below 4kT, thermal motion of the target matters.
         * Free-gas or S(α,β) depending on nuclide binding.
         * Above 4kT, the stationary-target approximation is
         * fine and we save ourselves the rejection loop. */
        const mo_cell_t *tc = &prob->cell[p->cell];
        double kT = tc->temp * MO_KB;

        if (kT > 1.0e-10 && p->E < 4.0 * kT && nuc_idx >= 0) {
            /* Check for S(α,β) table on this nuclide */
            const mo_mat_t *tm = &prob->mat[tc->mat];
            int sab_id = -1;
            int ni;
            for (ni = 0; ni < tm->n_nuc; ni++) {
                if (tm->nuc_id[ni] == nuc_idx) {
                    sab_id = tm->sab_id[ni];
                    break;
                }
            }

            if (sab_id >= 0 && sab_id < prob->n_sab) {
                /* Bound-atom S(α,β) scattering */
                double E_out, mu_sab;
                sb_samp(&prob->sab[sab_id], TP_RNG(prob), p->E,
                        &E_out, &mu_sab);
                p->E = E_out;
                tp_idir(TP_RNG(prob), p);
            } else {
                /* Free-gas thermal scattering */
                double E_out, mu_fg;
                th_fgas(TP_RNG(prob), p->E, A, kT, &E_out, &mu_fg);
                p->E = E_out;
                tp_idir(TP_RNG(prob), p);
            }

            if (p->E < MO_EMIN) {
                p->alive = 0;
                return;
            }
            /* Skip the normal elastic/inelastic direction set below */
            return;
        }

        /* ---- Elastic scatter: delta-Wigner transport correction ----
         *
         * The real angular distribution P(μ) is complicated — a sum
         * of Legendre polynomials with energy-dependent coefficients.
         * But the transport equation only cares about the AVERAGE
         * scattering cosine μ̄ for its eigenvalue. Everything else
         * is noise that washes out over enough collisions.
         *
         * Delta-Wigner splits each scatter into two outcomes:
         *   • probability μ̄:   forward scatter (no direction change,
         *                      no energy change — the neutron just
         *                      grazes the nucleus and carries on)
         *   • probability 1-μ̄: isotropic scatter (compound nucleus
         *                      formation, random exit direction)
         *
         * This preserves ⟨μ⟩ = μ̄·(1) + (1-μ̄)·(0) = μ̄ exactly.
         * At 1-2 MeV on uranium, μ̄ ≈ 0.5, so half the elastic
         * scatters now correctly do nothing — the neutron streams
         * onward toward the surface and leaks out. Without this,
         * every elastic scatter randomises the direction, trapping
         * neutrons inside the geometry like a pinball machine with
         * no drain. That's where the +4% k_eff bias lives.
         *
         * Reference:
         *   Bell, G. I., & Glasstone, S. (1970). Nuclear reactor
         *   theory. Van Nostrand Reinhold. Section 4.3c, transport
         *   correction to isotropic scattering. */

        const mo_nucl_t *nuc = NULL;
        if (nuc_idx >= 0 && !KA_CHK(nuc_idx, prob->n_nucl))
            nuc = &prob->nucl[nuc_idx];

        double mu = nuc ? tp_mubar(nuc, p->E) : 0.0;

        if (rn_dbl(TP_RNG(prob)) < mu) {
            /* Forward scatter: neutron glances off the nucleus.
             * No energy change, no direction change. The most
             * anticlimactic collision in physics. */
        } else {
            /* Compound nucleus: isotropic CM scatter.
             * The neutron was briefly absorbed, formed a compound
             * state, and was re-emitted with no memory of its
             * original direction. A nuclear amnesiac. */
            double mu_cm = 2.0 * rn_dbl(TP_RNG(prob)) - 1.0;
            double Ap1_sq = (A + 1.0) * (A + 1.0);
            p->E *= (A * A + 1.0 + 2.0 * A * mu_cm) / Ap1_sq;

            tp_idir(TP_RNG(prob), p);
        }
    }

    if (p->E < MO_EMIN) {
        p->alive = 0;
        return;
    }

    /* Inelastic always gets a new direction — the compound nucleus
     * doesn't remember which way the neutron came from, so isotropic
     * is physically correct here (unlike elastic, where it's an
     * approximation that was costing us 4% in k_eff). */
    if (is_inel) {
        tp_idir(TP_RNG(prob), p);
    }
}

/* ---- Watt fission spectrum ----
 * Sample fission neutron energy from the Watt spectrum:
 *   χ(E) = C · exp(-E/a) · sinh(√(b·E))
 *
 * Per-nuclide parameters from MCNP6/ENDF/B-V defaults:
 *   U-235: a=0.988 MeV, b=2.249 /MeV  (Cranberg et al., 1956)
 *   U-238: a=0.895 MeV, b=3.295 /MeV  (fast fission)
 *   U-234: same as U-238 (even-even uranium, ORNL/TM-13334)
 *   Pu-239: a=0.966 MeV, b=2.842 /MeV
 *
 * Using the Kozulin-Seliverstov rejection method.
 *
 * References:
 *   Cranberg, L., Frye, G., Nereson, N., & Rosen, L. (1956).
 *   Fission neutron spectrum of U²³⁵. Physical Review, 103(3), 662.
 *
 *   Werner, C. J. (Ed.). (2017). MCNP6.2 users manual
 *   (LA-UR-17-29981). Los Alamos National Laboratory. */

static void
tp_wtab(int za, double *a, double *b)
{
    switch (za) {
    case 92235: *a = 988000.0; *b = 2.249e-6; return;
    case 92238: *a = 895000.0; *b = 3.295e-6; return;
    case 92234: *a = 895000.0; *b = 3.295e-6; return;
    case 94239: *a = 966000.0; *b = 2.842e-6; return;
    case 94240: *a = 966000.0; *b = 2.842e-6; return;
    case 94241: *a = 966000.0; *b = 2.842e-6; return;
    case 94242: *a = 966000.0; *b = 2.842e-6; return;
    default:    *a = 988000.0; *b = 2.249e-6; return;
    }
}

static double
tp_watt(mo_rng_t *rng, int za)
{
    double a, b, x, y, E;

    tp_wtab(za, &a, &b);

    /* Rejection sampling from exponential envelope */
    KA_GUARD(g, 1000);
    while (g--) {
        x = -a * log(rn_dbl(rng));
        y = -log(rn_dbl(rng));
        E = x;
        double test = (y - (sqrt(b * x) - 1.0))
                    * (y - (sqrt(b * x) - 1.0));
        if (test <= b * x) return E;
    }
    /* Fallback: if rejection fails (shouldn't), return 2 MeV */
    return 2.0e6;
}

/* ---- Handle collision (survival biasing) ----
 *
 * At every collision:
 *   1. Score tallies
 *   2. Bank fission sites (using PRE-absorption weight)
 *   3. Reduce weight: w' = w · (1 − σ_abs / σ_t)
 *   4. Always scatter (particle never dies from absorption)
 *
 * The particle only dies from Russian roulette (low weight),
 * leakage, or energy cutoff. This is "implicit capture" or
 * "survival biasing" — every collision is a scatter, but the
 * weight carries the memory of all the absorptions that didn't
 * quite happen. Like a neutron with survivor's guilt.
 *
 * References:
 *   Brown, F. B. (2009). Fundamentals of Monte Carlo particle
 *   transport (LA-UR-05-4983 Rev.). Los Alamos National Laboratory.
 *
 *   OpenMC Development Team. (2024). OpenMC neutron physics methods.
 *   https://docs.openmc.org/en/latest/methods/neutron_physics.html */

static void
tp_coll(mo_prob_t *prob, mo_part_t *p, const mo_xs_t *xs,
        mo_site_t *fbank, int *n_fbank)
{
    /* Score tallies at collision */
    tl_fiss(TP_FIS(prob), p->cell, p->wt,
            xs->nu_bar, xs->sig_f, xs->sig_t);
    tl_coll(TP_FLX(prob), p->cell, p->wt, xs->sig_t);

    /* ---- Bank fission sites ----
     * Expected number = ν · wt · σ_f / (σ_t · k_eff).
     * Uses PRE-absorption weight per OpenMC convention.
     * Fission banking at every collision, not just absorption —
     * the weight encodes the probability.
     * NULL fbank = fixed-source mode: skip fission banking. */
    if (xs->sig_f > 1.0e-30 && fbank != NULL && n_fbank != NULL) {
        double nu_sites = xs->nu_bar * p->wt * xs->sig_f
                        / (xs->sig_t * prob->k_eff);
        int n_new = (int)nu_sites;
        if (rn_dbl(TP_RNG(prob)) < (nu_sites - (double)n_new))
            n_new++;

        int k;
        for (k = 0; k < n_new; k++) {
            if (*n_fbank >= TP_FBMAX) {
                /* Fission bank overflow: Russian roulette */
                int victim = (int)(rn_dbl(TP_RNG(prob)) * (*n_fbank));
                if (victim < *n_fbank)
                    fbank[victim].wt *= 2.0;
                continue;
            }
            mo_site_t *site = &fbank[*n_fbank];
            site->x  = p->x;
            site->y  = p->y;
            site->z  = p->z;
            site->E  = tp_watt(TP_RNG(prob), xs->fiss_za);
            site->wt = 1.0;
            (*n_fbank)++;
        }
    }

    /* ---- Survival biasing ----
     * Reduce weight by absorption probability. Particle survives
     * every collision but gets lighter each time.
     *
     * σ_abs = σ_capture + σ_fission.
     * If xs_a was computed via fallback (sig_t - sig_el), it
     * already includes fission — detect and handle. */
    double sig_abs = xs->sig_a + xs->sig_f;
    if (sig_abs > xs->sig_t) {
        sig_abs = xs->sig_t - xs->sig_el;
        if (sig_abs < 0.0) sig_abs = 0.0;
    }

    p->wt *= (1.0 - sig_abs / xs->sig_t);

    /* Always scatter — the implicit capture has already reduced
     * the weight. No random absorption sampling needed. */
    tp_scat(prob, p, xs);
}

/* ---- Russian roulette for low-weight particles ----
 * When weight drops below MO_WMIN (0.25), play Russian roulette:
 * survive with probability wt/MO_WSURV, boosted to MO_WSURV (1.0)
 * if you make it. Die otherwise. Unbiased: E[w'] = w.
 *
 * Standard MCNP/OpenMC defaults: w_cutoff = 0.25, w_survive = 1.0.
 *
 * This is the ONLY way a particle dies from weight reduction
 * in survival biasing mode. Leakage and energy cutoff are the
 * other two exits. Absorption itself never kills anyone — it
 * just makes them lighter until roulette finishes the job.
 * Nature is not this merciful. */

static void
tp_rlet(mo_rng_t *rng, mo_part_t *p)
{
    if (p->wt >= MO_WMIN) return;
    if (p->wt <= 0.0) { p->alive = 0; return; }

    if (rn_dbl(rng) * MO_WSURV < p->wt) {
        p->wt = MO_WSURV;
    } else {
        p->alive = 0;
    }
}

/* ---- Track one particle history ----
 * The main event. Follow this neutron from birth to death,
 * scoring tallies along the way. With survival biasing, the
 * neutron scatters at every collision (weight decreasing) until
 * it leaks, drops below energy cutoff, or loses at roulette. */

int
tp_hist(mo_prob_t *prob, mo_part_t *p,
        mo_site_t *fbank, int *n_fbank)
{
    int leaked = 0;

    KA_GUARD(g, MO_MAX_HIST);

    while (p->alive && g--) {
        /* Get cross-sections at current position/energy */
        mo_xs_t xs = xs_macro(prob, p->cell, p->E);

        /* Sample distance to collision: -ln(ξ) / Σ_t */
        double d_coll = -log(rn_dbl(TP_RNG(prob))) / xs.sig_t;

        /* Find nearest surface crossing */
        int s_hit = -1;
        double d_surf = cg_next(prob, p, &s_hit);

        if (d_surf < d_coll) {
            /* Surface crossing: score track-length up to surface,
             * move to surface, cross into new cell */
            tl_trak(TP_FLX(prob), p->cell, p->wt, d_surf);
            if (prob->etl_flux.active)
                te_scor(&prob->etl_flux, p->cell, p->wt, d_surf, p->E);
            if (prob->mesh && prob->mesh->active)
                tm_scor(prob->mesh, p->x, p->y, p->z, p->wt, d_surf);
            tp_move(p, d_surf);

            if (!cg_cross(prob, p, s_hit)) {
                /* Leaked out of geometry — farewell, little neutron */
                leaked = 1;
                break;
            }
        } else {
            /* Collision: score track-length, move to collision,
             * survival biasing + scatter */
            tl_trak(TP_FLX(prob), p->cell, p->wt, d_coll);
            if (prob->etl_flux.active)
                te_scor(&prob->etl_flux, p->cell, p->wt, d_coll, p->E);
            if (prob->mesh && prob->mesh->active)
                tm_scor(prob->mesh, p->x, p->y, p->z, p->wt, d_coll);
            tp_move(p, d_coll);
            tp_coll(prob, p, &xs, fbank, n_fbank);
        }

        /* Russian roulette for low-weight particles */
        tp_rlet(TP_RNG(prob), p);
    }

    return leaked;
}

/* ---- Float-precision CPU transport (GPU kernel mirror) ----
 * Runs the EXACT same logic as tp_kern.cu but on CPU.
 * All physics in float, all geometry inlined, no sub-functions.
 * If this gives k~0.84: the kernel formulas are wrong in float.
 * If this gives k~1.0: BarraCUDA's codegen is broken.
 * The nuclear Monte Carlo equivalent of "is it plugged in?" */

static int fh_ncoll = 0;
static int fh_npart = 0;

int
tp_fhst(mo_prob_t *prob, mo_part_t *p,
        mo_site_t *fbank, int *n_fbank)
{
    int leaked = 0;
    int step = MO_MAX_HIST;
    fh_npart++;

    /* Cast particle state to float — GPU precision */
    float px = (float)p->x,  py = (float)p->y,  pz = (float)p->z;
    float pu = (float)p->u,  pv = (float)p->v,  pw = (float)p->w;
    float pE = (float)p->E,  pwt = (float)p->wt;
    int   pcell = p->cell;
    int   alive = p->alive;

    /* GPU constants */
    const float FPI    = 3.14159265f;
    const float FEPS   = 1.0e-6f;
    const float FEMIN  = 1.0e-5f;
    const float FWMIN  = 0.25f;
    const float FWSRV  = 1.0f;
    const float FINF   = 1.0e30f;
    const float FK2F   = 4.8261e-8f;

    while (alive && step > 0) {
        step--;

        /* ---- Macroscopic XS (float, inline) ---- */
        float sig_t = 0.0f, sig_a = 0.0f, sig_f = 0.0f;
        float sig_el = 0.0f, sig_in = 0.0f, nu_bar = 0.0f;
        int fiss_za = 92235;
        float nuc_sel[8], nuc_awr[8], nuc_sin[8];
        int   nuc_nid[8];
        int nn_sav = 0;

        {
            const mo_cell_t *cl = &prob->cell[pcell];
            int mi = cl->mat;
            if (mi < 0 || mi >= prob->n_mat) { alive = 0; break; }
            float rho = (float)cl->rho;
            const mo_mat_t *mt = &prob->mat[mi];
            float max_sf = 0.0f;
            int ni;

            for (ni = 0; ni < mt->n_nuc && ni < 8; ni++) {
                int nid = mt->nuc_id[ni];
                if (nid < 0 || nid >= prob->n_nucl) continue;
                const mo_nucl_t *nuc = &prob->nucl[nid];

                /* ---- XS lookup (float, linear scan) ---- */
                float m_st, m_sa, m_sf, m_sel, m_nu;
                {
                    int np2 = nuc->n_pts;
                    int lo = 0, hi = np2 - 1;
                    m_st  = (float)nuc->xs_t[0];
                    m_sa  = (float)nuc->xs_a[0];
                    m_sf  = (float)nuc->xs_f[0];
                    m_sel = (float)nuc->xs_el[0];
                    m_nu  = (float)nuc->nu[0];

                    if (hi > 0 && pE > (float)nuc->E[0]) {
                        if (pE >= (float)nuc->E[hi]) {
                            m_st  = (float)nuc->xs_t[hi];
                            m_sa  = (float)nuc->xs_a[hi];
                            m_sf  = (float)nuc->xs_f[hi];
                            m_sel = (float)nuc->xs_el[hi];
                            m_nu  = (float)nuc->nu[hi];
                        } else {
                            int sj;
                            for (sj = 1; sj < np2; sj++) {
                                if (pE < (float)nuc->E[sj]) break;
                                lo = sj;
                            }
                            hi = lo + 1;
                            if (hi >= np2) hi = np2 - 1;
                            float dE = (float)nuc->E[hi] - (float)nuc->E[lo];
                            float f2 = 0.0f;
                            if (dE > 1.0e-30f)
                                f2 = (pE - (float)nuc->E[lo]) / dE;
                            m_st  = (float)nuc->xs_t[lo]  + f2*((float)nuc->xs_t[hi]  - (float)nuc->xs_t[lo]);
                            m_sa  = (float)nuc->xs_a[lo]  + f2*((float)nuc->xs_a[hi]  - (float)nuc->xs_a[lo]);
                            m_sf  = (float)nuc->xs_f[lo]  + f2*((float)nuc->xs_f[hi]  - (float)nuc->xs_f[lo]);
                            m_sel = (float)nuc->xs_el[lo]  + f2*((float)nuc->xs_el[hi] - (float)nuc->xs_el[lo]);
                            m_nu  = (float)nuc->nu[lo]    + f2*((float)nuc->nu[hi]    - (float)nuc->nu[lo]);
                        }
                    }

                    /* ---- SLBW resonance (float) ---- */
                    if (nuc->n_res > 0 &&
                        pE >= (float)nuc->EL && pE <= (float)nuc->EH)
                    {
                        float rst = 0.0f, rsf2 = 0.0f, rsg = 0.0f, rsel2 = 0.0f;
                        float awri = (float)nuc->awr;
                        if (awri >= 1.0f) {
                            float ecm = pE * awri / (awri + 1.0f);
                            if (ecm >= 1.0e-20f) {
                                float k2 = FK2F * ecm;
                                float kv = sqrtf(k2);
                                float a_fm = fabsf((float)nuc->AP) * 10.0f;
                                float rho_v = kv * a_fm;
                                float pikq = FPI / (k2 * 100.0f);
                                float sin_phi = sinf(rho_v);
                                float pot = 4.0f * pikq * sin_phi * sin_phi;
                                float cos2p = cosf(2.0f * rho_v);
                                float sin2p = sinf(2.0f * rho_v);
                                float gfac = 2.0f * (2.0f * (float)nuc->SPI + 1.0f);
                                if (gfac < 1.0f) gfac = 1.0f;
                                float sum_t = 0.0f, sum_f2 = 0.0f, sum_g = 0.0f;
                                int ri;
                                for (ri = 0; ri < nuc->n_res; ri++) {
                                    float Er   = (float)nuc->res[ri].Er;
                                    float AJ   = fabsf((float)nuc->res[ri].AJ);
                                    float Gn_r = fabsf((float)nuc->res[ri].Gn);
                                    float Gg   = fabsf((float)nuc->res[ri].Gg);
                                    float Gfa  = (float)nuc->res[ri].Gfa;
                                    float Gfb  = (float)nuc->res[ri].Gfb;
                                    float Gf   = sqrtf(Gfa*Gfa + Gfb*Gfb);
                                    float gJ   = (2.0f*AJ + 1.0f) / gfac;
                                    float Er_a = fabsf(Er);
                                    float Gnv  = Gn_r;
                                    if (Er_a > 1.0e-30f)
                                        Gnv = Gn_r * sqrtf(pE / Er_a);
                                    float Gt  = Gnv + Gg + Gf;
                                    float dEr = pE - Er;
                                    float Dv  = dEr*dEr + 0.25f*Gt*Gt;
                                    if (Dv < 1.0e-40f) continue;
                                    float psi = 0.25f*Gt*Gt / Dv;
                                    float chi = dEr * 0.5f * Gt / Dv;
                                    sum_t  += gJ*(Gnv/Gt)*(cos2p*psi + sin2p*chi);
                                    sum_f2 += gJ * Gnv * Gf / Dv;
                                    sum_g  += gJ * Gnv * Gg / Dv;
                                }
                                rst   = pot + 4.0f * pikq * sum_t;
                                rsf2  = pikq * sum_f2;
                                rsg   = pikq * sum_g;
                                rsel2 = rst - rsf2 - rsg;
                                if (rsel2 < 0.0f) rsel2 = 0.0f;
                            }
                        }
                        m_st  += rst;
                        m_sf  += rsf2;
                        m_sa  += rsg;
                        m_sel += rsel2;
                    }
                } /* end xs lookup */

                float rho_f = rho * (float)mt->frac[ni];
                sig_t  += rho_f * m_st;
                sig_a  += rho_f * m_sa;
                sig_f  += rho_f * m_sf;
                sig_el += rho_f * m_sel;
                if (m_sf > 0.0f)
                    nu_bar += m_nu * rho_f * m_sf;
                if (rho_f * m_sf > max_sf) {
                    max_sf = rho_f * m_sf;
                    fiss_za = nuc->ZA;
                }
                if (nn_sav < 8) {
                    nuc_sel[nn_sav] = rho_f * m_sel;
                    float m_sin = m_st - m_sel - m_sa - m_sf;
                    if (m_sin < 0.0f) m_sin = 0.0f;
                    nuc_sin[nn_sav] = rho_f * m_sin;
                    nuc_awr[nn_sav] = (float)nuc->awr;
                    nuc_nid[nn_sav] = nid;
                    nn_sav++;
                }
            } /* end nuclide loop */

            if (sig_f > 1.0e-30f)
                nu_bar /= sig_f;
            sig_in = sig_t - sig_el - sig_a - sig_f;
            if (sig_in < 0.0f) sig_in = 0.0f;
            if (sig_t < 1.0e-20f) sig_t = 1.0e-20f;
        }

        /* ---- Collision distance (float) ---- */
        float xi_coll = (float)rn_dbl(TP_RNG(prob));
        if (xi_coll < 1.0e-30f) xi_coll = 1.0e-30f;
        float d_coll = -logf(xi_coll) / sig_t;

        /* ---- Surface distance (float, GPU-style) ----
         * Uses t2 unconditionally, same as GPU kernel.
         * No t1>EPS filter, no disc<0 check. */
        float d_surf = FINF;
        int   s_hit  = -1;
        {
            const mo_cell_t *cl = &prob->cell[pcell];
            int sj;
            for (sj = 0; sj < cl->n_half && sj < 16; sj++) {
                int si = cl->surf[sj];
                if (si < 0 || si >= prob->n_surf) continue;
                const mo_surf_t *sf = &prob->surf[si];

                float sA = (float)sf->A, sB = (float)sf->B, sC = (float)sf->C;
                float sD = (float)sf->D, sE2 = (float)sf->E, sF = (float)sf->F;
                float sG = (float)sf->G, sH = (float)sf->H, sJ = (float)sf->J;
                float sK = (float)sf->K;

                float da = sA*pu*pu + sB*pv*pv + sC*pw*pw
                         + sD*pu*pv + sE2*pv*pw + sF*pu*pw;
                float db = 2.0f*(sA*px*pu + sB*py*pv + sC*pz*pw)
                         + sD*(px*pv + py*pu) + sE2*(py*pw + pz*pv)
                         + sF*(px*pw + pz*pu)
                         + sG*pu + sH*pv + sJ*pw;
                float dc = sA*px*px + sB*py*py + sC*pz*pz
                         + sD*px*py + sE2*py*pz + sF*px*pz
                         + sG*px + sH*py + sJ*pz + sK;

                float ds = FINF;
                if (fabsf(da) < 1.0e-20f) {
                    if (fabsf(db) >= 1.0e-20f) {
                        float tl = -dc / db;
                        if (tl > 0.0f) ds = tl;
                    }
                } else {
                    float disc = db*db - 4.0f*da*dc;
                    if (disc >= 0.0f) {
                        float sq = sqrtf(disc);
                        float t1 = (-db - sq) / (2.0f * da);
                        float t2 = (-db + sq) / (2.0f * da);
                        float d1 = FINF;
                        float d2 = FINF;
                        if (t1 > 0.0f) d1 = t1;
                        if (t2 > 0.0f) d2 = t2;
                        if (d1 < d2) ds = d1;
                        else ds = d2;
                    }
                }
                if (ds < d_surf) {
                    d_surf = ds;
                    s_hit  = si;
                }
            }
        }

        /* ---- Surface or Collision ---- */
        if (d_surf < d_coll) {
            /* Surface crossing */
            px += d_surf * pu;
            py += d_surf * pv;
            pz += d_surf * pw;
            px += FEPS * pu;
            py += FEPS * pv;
            pz += FEPS * pw;

            /* Find new cell (brute-force, same as GPU) */
            int nc = -1;
            {
                int ci;
                for (ci = 0; ci < prob->n_cell; ci++) {
                    const mo_cell_t *cc = &prob->cell[ci];
                    if (cc->n_half == 0) continue;
                    int inside = 1;
                    int fj;
                    for (fj = 0; fj < cc->n_half && fj < 16; fj++) {
                        int fsi = cc->surf[fj];
                        if (fsi < 0 || fsi >= prob->n_surf)
                            { inside = 0; break; }
                        const mo_surf_t *fs = &prob->surf[fsi];
                        float fval = (float)fs->A*px*px + (float)fs->B*py*py
                                   + (float)fs->C*pz*pz + (float)fs->D*px*py
                                   + (float)fs->E*py*pz + (float)fs->F*px*pz
                                   + (float)fs->G*px + (float)fs->H*py
                                   + (float)fs->J*pz + (float)fs->K;
                        int fsn = cc->sense[fj];
                        if (fsn < 0 && fval > 0.0f) { inside = 0; break; }
                        if (fsn > 0 && fval < 0.0f) { inside = 0; break; }
                    }
                    if (inside) { nc = ci; break; }
                }
            }
            if (nc < 0) { alive = 0; leaked = 1; break; }
            pcell = nc;
            if (prob->cell[nc].mat < 0) { alive = 0; leaked = 1; break; }
        } else {
            /* Collision */
            fh_ncoll++;
            px += d_coll * pu;
            py += d_coll * pv;
            pz += d_coll * pw;

            /* Fission banking */
            if (sig_f > 1.0e-30f && fbank && n_fbank) {
                float nu_sites = nu_bar * pwt * sig_f
                               / (sig_t * (float)prob->k_eff);
                int n_new = (int)nu_sites;
                float frac = nu_sites - (float)n_new;
                if ((float)rn_dbl(TP_RNG(prob)) < frac) n_new++;

                int fk;
                for (fk = 0; fk < n_new; fk++) {
                    if (*n_fbank >= TP_FBMAX) break;
                    /* Watt spectrum (float, same as GPU) */
                    float wE = 2.0e6f;
                    {
                        float wa = 988000.0f, wb = 2.249e-6f;
                        if (fiss_za == 92238 || fiss_za == 92234)
                            { wa = 895000.0f; wb = 3.295e-6f; }
                        else if (fiss_za == 94239 || fiss_za == 94240 ||
                                 fiss_za == 94241 || fiss_za == 94242)
                            { wa = 966000.0f; wb = 2.842e-6f; }
                        int wg = 1000;
                        while (wg > 0) {
                            wg--;
                            float wr1 = (float)rn_dbl(TP_RNG(prob));
                            float wr2 = (float)rn_dbl(TP_RNG(prob));
                            if (wr1 < 1.0e-30f) wr1 = 1.0e-30f;
                            if (wr2 < 1.0e-30f) wr2 = 1.0e-30f;
                            float wx = -wa * logf(wr1);
                            float wy = -logf(wr2);
                            float wsq = wy - (sqrtf(wb * wx) - 1.0f);
                            if (wsq * wsq <= wb * wx) { wE = wx; break; }
                        }
                    }
                    mo_site_t *site = &fbank[*n_fbank];
                    site->x  = (double)px;
                    site->y  = (double)py;
                    site->z  = (double)pz;
                    site->E  = (double)wE;
                    site->wt = 1.0;
                    (*n_fbank)++;
                }
            }

            /* Survival biasing */
            float sab = sig_a + sig_f;
            if (sab > sig_t) {
                sab = sig_t - sig_el;
                if (sab < 0.0f) sab = 0.0f;
            }
            pwt *= (1.0f - sab / sig_t);

            /* Scatter: delta-Wigner + inelastic (matches GPU kernel) */
            {
                /* Elastic nuclide selection */
                float sA = 1.0f;
                int s_nid = -1;
                {
                    float xi_sc = (float)rn_dbl(TP_RNG(prob));
                    float scum = 0.0f;
                    int si;
                    for (si = 0; si < nn_sav; si++) {
                        scum += nuc_sel[si];
                        if (scum >= xi_sc * sig_el
                            || si == nn_sav - 1) {
                            sA = nuc_awr[si];
                            s_nid = nuc_nid[si];
                            break;
                        }
                    }
                }

                /* Mu-bar lookup */
                float mb = 0.0f;
                if (s_nid >= 0 && s_nid < prob->n_nucl) {
                    const mo_nucl_t *nuc = &prob->nucl[s_nid];
                    int nm = nuc->n_mu;
                    if (nm == 1) {
                        mb = (float)nuc->mu_bar[0];
                    } else if (nm > 1) {
                        int mlo = 0, mhi = nm - 1, mg = 20;
                        while (mlo + 1 < mhi && mg > 0) {
                            mg--;
                            int mm = (mlo + mhi) / 2;
                            if (pE < (float)nuc->mu_E[mm])
                                mhi = mm;
                            else
                                mlo = mm;
                        }
                        float dE = (float)(nuc->mu_E[mhi] - nuc->mu_E[mlo]);
                        if (dE > 1.0e-30f) {
                            float f = (pE - (float)nuc->mu_E[mlo]) / dE;
                            if (f < 0.0f) f = 0.0f;
                            if (f > 1.0f) f = 1.0f;
                            mb = (float)nuc->mu_bar[mlo]
                               + f * (float)(nuc->mu_bar[mhi] - nuc->mu_bar[mlo]);
                        } else {
                            mb = (float)nuc->mu_bar[mlo];
                        }
                    }
                }

                /* Scatter type decision */
                float sig_scat = sig_el + sig_in;
                float xi_br = (float)rn_dbl(TP_RNG(prob));

                if (sig_scat > 1.0e-30f &&
                    sig_in > 1.0e-30f &&
                    xi_br * sig_scat > sig_el) {
                    /* INELASTIC: discrete levels or evaporation */
                    float iA = 1.0f;
                    int i_nid = -1;
                    {
                        float xi_is = (float)rn_dbl(TP_RNG(prob));
                        float iscum = 0.0f;
                        int isi;
                        for (isi = 0; isi < nn_sav; isi++) {
                            iscum += nuc_sin[isi];
                            if (iscum >= xi_is * sig_in
                                || isi == nn_sav - 1) {
                                iA = nuc_awr[isi];
                                i_nid = nuc_nid[isi];
                                break;
                            }
                        }
                    }
                    int n_acc = 0;
                    if (i_nid >= 0 && i_nid < prob->n_nucl) {
                        const mo_nucl_t *inuc = &prob->nucl[i_nid];
                        int li;
                        for (li = 0; li < inuc->n_lev && li < 40; li++) {
                            if ((float)inuc->lev_Q[li] * (iA+1.0f)/iA < pE)
                                n_acc++;
                        }
                        if (n_acc > 0) {
                            float xi_lev = (float)rn_dbl(TP_RNG(prob));
                            int pick = (int)(xi_lev * (float)n_acc);
                            if (pick >= n_acc) pick = n_acc - 1;
                            pE -= (float)inuc->lev_Q[pick] * (iA+1.0f)/iA;
                            if (pE < FEMIN) pE = FEMIN;
                        } else {
                            float T = sqrtf(8.0e6f * pE / iA);
                            float wr1 = (float)rn_dbl(TP_RNG(prob));
                            float wr2 = (float)rn_dbl(TP_RNG(prob));
                            if (wr1 < 1.0e-30f) wr1 = 1.0e-30f;
                            if (wr2 < 1.0e-30f) wr2 = 1.0e-30f;
                            float E_out = -T * logf(wr1 * wr2);
                            if (E_out >= pE) E_out = pE * 0.5f;
                            pE = E_out;
                        }
                    }
                    /* Isotropic direction */
                    float imu = 2.0f * (float)rn_dbl(TP_RNG(prob)) - 1.0f;
                    float iphi = 2.0f * FPI * (float)rn_dbl(TP_RNG(prob));
                    float ist = sqrtf(1.0f - imu * imu);
                    pu = ist * cosf(iphi);
                    pv = ist * sinf(iphi);
                    pw = imu;
                } else {
                    /* ELASTIC: delta-Wigner */
                    float xi_mu = (float)rn_dbl(TP_RNG(prob));
                    if (xi_mu < mb) {
                        /* Forward scatter: no change */
                    } else {
                        /* Compound nucleus */
                        float mu_cm = 2.0f * (float)rn_dbl(TP_RNG(prob)) - 1.0f;
                        float Ap1 = sA + 1.0f;
                        pE *= (sA*sA + 1.0f + 2.0f*sA*mu_cm) / (Ap1*Ap1);
                        float emu = 2.0f * (float)rn_dbl(TP_RNG(prob)) - 1.0f;
                        float ephi = 2.0f * FPI * (float)rn_dbl(TP_RNG(prob));
                        float est = sqrtf(1.0f - emu * emu);
                        pu = est * cosf(ephi);
                        pv = est * sinf(ephi);
                        pw = emu;
                    }
                }
                if (pE < FEMIN) alive = 0;
            }
        }

        /* Russian roulette */
        if (alive && pwt < FWMIN) {
            if (pwt <= 0.0f) {
                alive = 0;
            } else {
                float xi_rl = (float)rn_dbl(TP_RNG(prob));
                if (xi_rl * FWSRV < pwt)
                    pwt = FWSRV;
                else
                    alive = 0;
            }
        }
    } /* end transport loop */

    /* Write back to particle (for diagnostics) */
    p->x = (double)px; p->y = (double)py; p->z = (double)pz;
    p->u = (double)pu; p->v = (double)pv; p->w = (double)pw;
    p->E = (double)pE; p->wt = (double)pwt;
    p->cell = pcell;   p->alive = alive;

    return leaked;
}

/* ---- OpenMP support functions ----
 * Set up per-thread state before a parallel region, reduce
 * afterwards. Called from tc_run / tf_run. In serial mode
 * these are harmless no-ops, like safety rails on a bridge
 * nobody crosses. */

void
tp_oset(mo_prob_t *prob, int nt)
{
#ifdef _OPENMP
    int t;
    if (nt > MO_MAX_THRD) nt = MO_MAX_THRD;

    /* Per-thread RNGs: jump 2^128 apart so they never overlap.
     * Thread 0 gets the master state, thread N gets N jumps ahead. */
    g_trng[0] = prob->rng;
    for (t = 1; t < nt; t++) {
        g_trng[t] = g_trng[t - 1];
        rn_jump(&g_trng[t]);
    }

    /* Zero per-thread tallies */
    for (t = 0; t < nt; t++) {
        tl_init(&g_tflx[t]);
        tl_zero(&g_tflx[t]);
        tl_init(&g_tfis[t]);
        tl_zero(&g_tfis[t]);
    }

    g_fbmax = MO_MAX_PART / nt;
    g_omp = 1;
#else
    (void)prob; (void)nt;
#endif
}

void
tp_ored(mo_prob_t *prob, int nt)
{
#ifdef _OPENMP
    int t, r;
    if (nt > MO_MAX_THRD) nt = MO_MAX_THRD;

    /* Accumulate per-thread tallies into prob's batch arrays */
    for (t = 0; t < nt; t++) {
        for (r = 0; r < MO_MAX_REG; r++) {
            prob->tl_flux.bat[r] += g_tflx[t].bat[r];
            prob->tl_fiss.bat[r] += g_tfis[t].bat[r];
        }
    }

    /* Advance master RNG past all thread subsequences */
    for (t = 0; t < nt; t++)
        rn_jump(&prob->rng);

    g_fbmax = MO_MAX_PART;
    g_omp = 0;
#else
    (void)prob; (void)nt;
#endif
}

mo_rng_t *
tp_trng(int tid)
{
#ifdef _OPENMP
    if (tid >= 0 && tid < MO_MAX_THRD) return &g_trng[tid];
#else
    (void)tid;
#endif
    return NULL;
}
