/* nd_parse.c -- ENDF-6 format parser
 *
 * Parses the ENDF-6 nuclear data format, which is what happens when
 * FORTRAN programmers design a file format in 1966 and everyone is
 * too polite to say anything for sixty years.
 *
 * 80-character fixed-width records. Six 11-character floating-point
 * fields that use a notation where the 'E' in scientific notation is
 * optional. Because why make things easy.
 *
 * We only parse:
 *   MF1/MT451 — general info (AWR)
 *   MF1/MT452 — nu-bar (neutrons per fission)
 *   MF3       — cross-section data (MT1=total, MT2=elastic,
 *               MT18=fission, MT102=capture)
 *
 * IMPORTANT: each MF3 reaction has its OWN energy grid. We parse
 * MT1 first (always first in ENDF ordering), use its grid as master,
 * and interpolate all other reactions onto it. Learned this the hard
 * way when everything came out as zero. Ask me how I know.
 *
 * NOTE: In the resolved/unresolved resonance region, MF3 only gives
 * the background cross-section. The actual XS requires MF2 resonance
 * reconstruction (not implemented). For fast-spectrum problems like
 * Godiva, the MF3 data is correct above ~150 keV which is where most
 * of the action happens anyway.
 *
 * Reference:
 *   Herman, M., & Trkov, A. (Eds.). (2009). ENDF-6 formats manual:
 *   Data formats and procedures for the evaluated nuclear data files
 *   ENDF/B-VI and ENDF/B-VII (BNL-90365-2009 Rev. 2). Brookhaven
 *   National Laboratory.
 */

#include "../moa.h"

/* ---- ENDF record layout ----
 * Cols  1-11: field 1 (float or int)
 * Cols 12-22: field 2
 * Cols 23-33: field 3
 * Cols 34-44: field 4
 * Cols 45-55: field 5
 * Cols 56-66: field 6
 * Cols 67-70: MAT number
 * Cols 71-72: MF number
 * Cols 73-75: MT number
 * Cols 76-80: line sequence
 *
 * Yes, this is a real format. No, we're not joking. */

#define ENDF_RECLEN 81

/* Read one ENDF record. Returns 0 on success, -1 on EOF/error. */
static int
nd_read(char *buf, FILE *f)
{
    if (!fgets(buf, ENDF_RECLEN + 16, f)) return -1;
    int len = (int)strlen(buf);
    while (len < 80) buf[len++] = ' ';
    buf[80] = '\0';
    return 0;
}

/* Extract MF and MT from cols 71-75 (0-indexed: 70-74).
 * MF is 2 chars wide, MT is 3 chars wide. Do NOT use nd_ival
 * here — its 11-char field width reads into the line sequence
 * number, turning MT=0 into MT=99999. That particular bug ate
 * an entire afternoon. */
static int
nd_mf(const char *line)
{
    char buf[3] = { line[70], line[71], '\0' };
    return atoi(buf);
}

static int
nd_mt(const char *line)
{
    char buf[4] = { line[72], line[73], line[74], '\0' };
    return atoi(buf);
}

/* ---- Float parser ----
 * ENDF uses FORTRAN E-notation where the 'E' is optional.
 * "1.23456+003" means 1.23456e+003. "-2.34-01" means -2.34e-01.
 * Field width: 11 characters. */

double
nd_fval(const char *s, int n)
{
    char buf[24];
    int  i, j = 0, got_exp = 0;

    for (i = 0; i < n && j < 22; i++) {
        char c = s[i];
        if (c == ' ' || c == '\0') continue;

        if ((c == '+' || c == '-') && j > 0 && !got_exp) {
            char prev = buf[j - 1];
            if (prev != 'E' && prev != 'e' && prev != 'D' && prev != 'd'
                && prev != '+' && prev != '-') {
                buf[j++] = 'E';
                got_exp = 1;
            }
        }
        if (c == 'D' || c == 'd') {
            buf[j++] = 'E';
            got_exp = 1;
            continue;
        }
        if (c == 'E' || c == 'e') got_exp = 1;
        buf[j++] = c;
    }
    buf[j] = '\0';

    if (j == 0) return 0.0;
    return strtod(buf, NULL);
}

/* Parse integer from fixed-width field (max 11 chars).
 * Must respect the field boundary or it'll eat the MAT number
 * for dessert. Discovered when NP=439 became NP=4399225. */
int
nd_ival(const char *s)
{
    int val = 0, neg = 0, i = 0;

    while (i < 11 && s[i] == ' ') i++;
    if (i < 11 && s[i] == '-') { neg = 1; i++; }
    else if (i < 11 && s[i] == '+') i++;

    while (i < 11 && s[i] >= '0' && s[i] <= '9') {
        val = val * 10 + (s[i] - '0');
        i++;
    }
    return neg ? -val : val;
}

/* ---- TAB1 record parser ----
 * Parses tabulated (x,y) pairs with interpolation info.
 * Returns number of points parsed. */

int
nd_tab1(FILE *f, double *x, double *y, int max_pts)
{
    char line[96];
    int  nr, np, n, i;

    if (nd_read(line, f)) return 0;

    nr = nd_ival(line + 44);
    np = nd_ival(line + 55);

    if (np <= 0) return 0;

    /* Skip interpolation range records: ceil(2*NR / 6) lines */
    int nr_lines = (2 * nr + 5) / 6;
    KA_GUARD(g1, 100);
    for (i = 0; i < nr_lines && g1--; i++) {
        if (nd_read(line, f)) return 0;
    }

    /* Cap at max_pts but CONSUME all data lines from the file
     * so the file pointer is left at the right position */
    int np_read = np;
    n = 0;
    KA_GUARD(g2, 20000);
    while (n < np_read && g2--) {
        if (nd_read(line, f)) break;
        int j;
        for (j = 0; j < 3 && n < np_read; j++) {
            if (n < max_pts) {
                x[n] = nd_fval(line + j * 22, 11);
                y[n] = nd_fval(line + j * 22 + 11, 11);
            }
            n++;
        }
    }
    return (np_read < max_pts) ? np_read : max_pts;
}

/* Section end: MT=0 */
static int
is_send(const char *line)
{
    return nd_mt(line) == 0;
}

/* ---- Interpolate src grid onto dst grid ----
 * Linear interpolation of (src_e, src_y, n_src) evaluated at
 * each point in (dst_e, n_dst), storing results in dst_y.
 * Clamps to boundary values outside source range.
 * Returns 0 below first source point (threshold reactions). */

static void
nd_interp(const double *src_e, const double *src_y, int n_src,
          const double *dst_e, double *dst_y, int n_dst)
{
    int i, lo = 0;

    for (i = 0; i < n_dst; i++) {
        double E = dst_e[i];

        /* Below source range: zero (threshold) */
        if (n_src <= 0 || E < src_e[0]) {
            dst_y[i] = 0.0;
            continue;
        }
        /* Above source range: clamp to last value */
        if (E >= src_e[n_src - 1]) {
            dst_y[i] = src_y[n_src - 1];
            continue;
        }

        /* Advance lo to bracket E (sequential search since dst is sorted) */
        while (lo + 1 < n_src && src_e[lo + 1] < E) lo++;
        if (lo + 1 >= n_src) lo = n_src - 2;

        double dE = src_e[lo + 1] - src_e[lo];
        if (dE > 1.0e-30) {
            double f = (E - src_e[lo]) / dE;
            dst_y[i] = src_y[lo] + f * (src_y[lo + 1] - src_y[lo]);
        } else {
            dst_y[i] = src_y[lo];
        }
    }
}

/* ---- Parse MF2/MT151 — Resonance parameters ----
 * The resolved resonance region, where nuclear physics gets
 * personal. Each resonance is a quantum state of the compound
 * nucleus, characterised by energy, spin, and partial widths.
 *
 * MF2 record structure (resolved, LRF=1/2/3):
 *   HEAD: ZA, AWR, 0, 0, NIS, 0
 *   For each isotope:
 *     CONT: ZAI, ABN, 0, LFW, NER, 0
 *     For each energy range:
 *       CONT: EL, EH, LRU, LRF, NRO, NAPS
 *       [TAB1 if NRO>0: energy-dependent AP]
 *       CONT: SPI, AP, LAD, 0, NLS, NLSC
 *       For each l-value:
 *         CONT: AWRI, APL, L, 0, 6*NRS, NRS
 *         NRS lines of 6 parameters each
 *
 * For LRF=1,2 (SLBW/MLBW): ER, AJ, GT, GN, GG, GF
 * For LRF=3 (Reich-Moore):  ER, AJ, GN, GG, GFA, GFB
 *
 * We only handle LRU=1 (resolved). Unresolved (LRU=2) is
 * skipped because life is short and approximations are long.
 *
 * Reference:
 *   Herman, M., & Trkov, A. (Eds.). (2009). ENDF-6 formats
 *   manual (BNL-90365-2009 Rev. 2). Sections 2.1–2.4. */

static int
nd_mf2(mo_nucl_t *nuc, FILE *f, const char *head)
{
    char line[96];
    int  nis, iso;

    nis = nd_ival(head + 44);

    KA_GUARD(g_iso, 4);
    for (iso = 0; iso < nis && g_iso--; iso++) {
        if (nd_read(line, f)) return -1;
        /* CONT: ZAI, ABN, 0, LFW, NER, 0 */
        int ner = nd_ival(line + 44);

        int er;
        KA_GUARD(g_er, 4);
        for (er = 0; er < ner && g_er--; er++) {
            if (nd_read(line, f)) return -1;
            /* CONT: EL, EH, LRU, LRF, NRO, NAPS */
            double EL  = nd_fval(line, 11);
            double EH  = nd_fval(line + 11, 11);
            int    lru = nd_ival(line + 22);
            int    lrf = nd_ival(line + 33);
            int    nro = nd_ival(line + 44);

            /* Energy-dependent scattering radius: consume TAB1 */
            if (nro > 0) {
                static double dummy_x[100], dummy_y[100];
                nd_tab1(f, dummy_x, dummy_y, 100);
            }

            /* Unresolved resonances (LRU=2): parse probability tables */
            if (lru == 2) {
                nuc->EL_ur = EL;
                nuc->EH_ur = EH;
                ur_load(nuc, f, line);
                continue;
            }

            /* We only handle resolved resonances (LRU=1). */
            if (lru != 1 || lrf < 1 || lrf > 3) {
                /* Skip to SEND for this subsection. */
                if (nd_read(line, f)) return -1;
                int nls = nd_ival(line + 44);
                int l_idx;
                KA_GUARD(g_skip, 8);
                for (l_idx = 0; l_idx < nls && g_skip--; l_idx++) {
                    if (nd_read(line, f)) return -1;
                    int nrs = nd_ival(line + 55);
                    int r;
                    KA_GUARD(g_sr, 10000);
                    for (r = 0; r < nrs && g_sr--; r++) {
                        if (nd_read(line, f)) return -1;
                    }
                }
                continue;
            }

            /* CONT: SPI, AP, LAD, 0, NLS, NLSC */
            if (nd_read(line, f)) return -1;
            nuc->SPI = nd_fval(line, 11);
            nuc->AP  = nd_fval(line + 11, 11);
            int nls  = nd_ival(line + 44);

            nuc->EL  = EL;
            nuc->EH  = EH;
            nuc->lrf = lrf;
            nuc->n_res = 0;

            /* For each l-value subsection */
            int l_idx;
            KA_GUARD(g_l, 8);
            for (l_idx = 0; l_idx < nls && g_l--; l_idx++) {
                if (nd_read(line, f)) return -1;
                /* CONT: AWRI, APL, L, 0, 6*NRS, NRS */
                double apl = nd_fval(line + 11, 11);
                int nrs = nd_ival(line + 55);

                /* Use APL as scattering radius if AP is zero.
                 * U-238 has AP=0 at isotope level but APL=0.948
                 * in the l-subsection. Without this, we'd compute
                 * zero potential scattering. */
                if (fabs(nuc->AP) < 1.0e-10 && fabs(apl) > 1.0e-10)
                    nuc->AP = apl;

                /* Read NRS resonances, one per line (6 params each) */
                int r;
                KA_GUARD(g_r, MO_MAX_RES + 100);
                for (r = 0; r < nrs && g_r--; r++) {
                    if (nd_read(line, f)) return -1;

                    if (nuc->n_res >= MO_MAX_RES) continue;

                    mo_res_t *res = &nuc->res[nuc->n_res];

                    if (lrf <= 2) {
                        /* SLBW/MLBW: ER, AJ, GT, GN, GG, GF */
                        res->Er  = nd_fval(line,      11);
                        res->AJ  = nd_fval(line + 11, 11);
                        /* skip GT (field 3) — redundant */
                        res->Gn  = nd_fval(line + 33, 11);
                        res->Gg  = nd_fval(line + 44, 11);
                        res->Gfa = nd_fval(line + 55, 11);
                        res->Gfb = 0.0;
                    } else {
                        /* Reich-Moore: ER, AJ, GN, GG, GFA, GFB */
                        res->Er  = nd_fval(line,      11);
                        res->AJ  = nd_fval(line + 11, 11);
                        res->Gn  = nd_fval(line + 22, 11);
                        res->Gg  = nd_fval(line + 33, 11);
                        res->Gfa = nd_fval(line + 44, 11);
                        res->Gfb = nd_fval(line + 55, 11);
                    }
                    nuc->n_res++;
                }
            }
        }
    }
    return 0;
}

/* ---- Parse MF4/MT2 — Elastic angular distributions ----
 *
 * Nuclear scattering is not isotropic. At MeV energies on heavy
 * nuclei, the neutron has a strong preference for continuing in
 * roughly the same direction — like a lorry glancing off a
 * roundabout rather than doing a U-turn. The angular distribution
 * is expanded in Legendre polynomials P_l(cos θ), and the first
 * coefficient a₁ IS the average scattering cosine μ̄ in the CM
 * frame. We don't need the higher moments; the delta-Wigner
 * transport correction only uses μ̄, which is all that separates
 * our +4% k_eff bias from a correct answer.
 *
 * MF4 structure for LTT=1 or LTT=3 (Legendre part):
 *   HEAD: ZA, AWR, 0, LTT, 0, 0
 *   CONT: 0, 0, LI, 0, NM, 0     (LI=1 means isotropic — skip)
 *   TAB2: 0, 0, 0, 0, NR, NE     (NE incident energies)
 *   [interpolation lines]
 *   For each energy:
 *     LIST: 0, E, LT, 0, NL, 0   (NL Legendre coefficients)
 *     [coefficient lines: a₁, a₂, ... a_NL]
 *
 * For LTT=3, there's a second TAB2 with tabulated P(μ) at higher
 * energies. We skip it — the Legendre range covers the fission
 * spectrum peak, and life is too short for numerical quadrature
 * on a Sunday afternoon.
 *
 * Reference:
 *   Herman, M., & Trkov, A. (Eds.). (2009). ENDF-6 formats
 *   manual (BNL-90365-2009 Rev. 2). Section 4.2. */

static int
nd_mf4(mo_nucl_t *nuc, FILE *f, const char *head)
{
    char line[96];
    int  ltt, ne, i;

    ltt = nd_ival(head + 33);

    /* CONT record: 0, 0, LI, 0, NM, 0.
     * LI=1 means "isotropic in all frames" — nothing to parse.
     * This would mean the evaluators gave up on angular data,
     * which is their prerogative but leaves us with nothing. */
    if (nd_read(line, f)) return -1;
    int li = nd_ival(line + 22);
    if (li == 1) return 0;  /* isotropic — no μ̄ table needed */

    /* LTT=0 means "all isotropic" despite LI=0. ENDF is not
     * a format designed by people who feared redundancy. */
    if (ltt == 0) return 0;

    /* TAB2 header: 0, 0, 0, 0, NR, NE.
     * NE is the number of incident energies in the Legendre
     * subsection. For U-235 this is 110 — plenty to interpolate. */
    if (nd_read(line, f)) return -1;
    int nr = nd_ival(line + 44);
    ne     = nd_ival(line + 55);

    /* Skip interpolation parameter lines (same formula as TAB1) */
    int nr_lines = (2 * nr + 5) / 6;
    KA_GUARD(g_nr, 100);
    for (i = 0; i < nr_lines && g_nr--; i++) {
        if (nd_read(line, f)) return -1;
    }

    /* Read Legendre coefficients at each incident energy.
     * We only want a₁ — the average scattering cosine. The rest
     * describe the detailed angular shape (P₂ forward/backward
     * asymmetry, P₃ side lobes, etc.) which the delta-Wigner
     * correction doesn't use. Grab and discard, like reading a
     * 300-page novel for one quote on page 47. */
    nuc->n_mu = 0;

    KA_GUARD(g_e, MO_MAX_MUBAR + 100);
    for (i = 0; i < ne && g_e--; i++) {
        /* LIST header: 0, E_inc, LT, 0, NL, 0 */
        if (nd_read(line, f)) return -1;
        double E_inc = nd_fval(line + 11, 11);
        int    nl    = nd_ival(line + 44);

        /* Coefficient data: ceil(NL/6) lines, 6 values per line.
         * a₁ is the first value on the first line. */
        int n_data = (nl + 5) / 6;
        double a1 = 0.0;

        int k;
        KA_GUARD(g_d, 200);
        for (k = 0; k < n_data && g_d--; k++) {
            if (nd_read(line, f)) return -1;
            if (k == 0) {
                a1 = nd_fval(line, 11);
            }
        }

        /* Store (E, μ̄) pair. Clamp μ̄ to [0,1] because negative
         * average cosines would mean the nucleus is somehow
         * reflecting neutrons backwards on average, which happens
         * for hydrogen at some energies but never for A > 10.
         * And if the ENDF evaluator put a₁ > 1.0, that's their
         * problem, not ours. */
        if (nuc->n_mu < MO_MAX_MUBAR) {
            double mu = a1;
            if (mu < 0.0) mu = 0.0;
            if (mu > 1.0) mu = 1.0;
            nuc->mu_E[nuc->n_mu]   = E_inc;
            nuc->mu_bar[nuc->n_mu] = mu;
            nuc->n_mu++;
        }
    }

    /* If LTT=3, there's a second subsection with tabulated P(μ)
     * at higher energies. We don't parse it — the Legendre range
     * covers the fission spectrum peak where it matters, and
     * extracting μ̄ from tabulated distributions requires numerical
     * integration, which is more effort than the +0.1% improvement
     * would justify. The remaining records will be consumed by
     * the SEND-seeking loop in nd_load. */

    return 0;
}

/* ---- Load nuclide data from ENDF file ----
 *
 * Two-pass approach:
 *   Pass 1: Read file, parse MT1 for master grid, stash other MTs
 *   Pass 2: Interpolate all reactions onto master grid
 *
 * Actually single-pass since ENDF guarantees MT1 comes first in MF3.
 * We parse MT1 directly into nuc, then interpolate other MTs as we go.
 *
 * Returns 0 on success, -1 on failure. */

int
nd_load(mo_nucl_t *nuc, const char *path, int za)
{
    FILE *f;
    char  line[96];
    int   mf, mt;

    memset(nuc, 0, sizeof(*nuc));
    nuc->ZA = za;

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "moa: cannot open ENDF file: %s\n", path);
        return -1;
    }

    /* Temp arrays for each reaction's own grid.
     * Static because 80KB × 2 on the stack would make Windows faint. */
    static double tmp_e[MO_MAX_EPTS], tmp_y[MO_MAX_EPTS];

    /* Separate stash for nu-bar (parsed from MF1 before MF3) */
    static double nu_e[MO_MAX_EPTS], nu_y[MO_MAX_EPTS];
    int n_nu = 0;
    int got_fiss = 0, got_cap = 0;

    /* Scan through the file */
    KA_GUARD(g, 500000);
    while (g-- && nd_read(line, f) == 0) {
        mf = nd_mf(line);
        mt = nd_mt(line);

        /* MF1/MT451: general info — grab AWR from field 2 */
        if (mf == 1 && mt == 451) {
            nuc->awr = nd_fval(line + 11, 11);
            KA_GUARD(g2, 10000);
            while (g2-- && nd_read(line, f) == 0) {
                if (is_send(line)) break;
            }
            continue;
        }

        /* MF1/MT452: total nu-bar */
        if (mf == 1 && mt == 452) {
            int lnu = nd_ival(line + 33);
            if (lnu == 2) {
                n_nu = nd_tab1(f, nu_e, nu_y, MO_MAX_EPTS);
            }
            KA_GUARD(g2, 10000);
            while (g2-- && nd_read(line, f) == 0) {
                if (is_send(line)) break;
            }
            continue;
        }

        /* MF2/MT151: resonance parameters */
        if (mf == 2 && mt == 151) {
            nd_mf2(nuc, f, line);
            KA_GUARD(g2, 10000);
            while (g2-- && nd_read(line, f) == 0) {
                if (is_send(line)) break;
            }
            continue;
        }

        /* MF4/MT2: elastic angular distribution.
         * We only want the average scattering cosine μ̄(E) — first
         * Legendre coefficient a₁ at each incident energy. The rest
         * of MF4 (higher Legendre moments, other MTs) can keep its
         * secrets. We're not building MCNP, we're building something
         * that gives the right answer for the right reasons. */
        if (mf == 4 && mt == 2) {
            nd_mf4(nuc, f, line);
            KA_GUARD(g2, 10000);
            while (g2-- && nd_read(line, f) == 0) {
                if (is_send(line)) break;
            }
            continue;
        }

        /* Skip non-MF1/MF2/MF3 sections entirely to avoid burning guard */
        if (mf != 1 && mf != 3 && mt != 0) continue;

        /* MF3: reaction cross-sections */
        if (mf == 3) {
            /* Discrete inelastic levels (MT51-89): we only need the
             * Q-value from the TAB1 header, not the actual XS data.
             * We compute σ_inel from the total-minus-parts difference,
             * but need the Q-values for outgoing energy. Grab QI
             * before nd_tab1 eats the header for breakfast. */
            if (mt >= 51 && mt <= 89
                && nuc->n_lev < MO_MAX_DLEV) {
                /* Read TAB1 header: QM(f1), QI(f2), 0, LR, NR, NP */
                if (nd_read(line, f) == 0) {
                    double qi = nd_fval(line + 11, 11);
                    if (qi < -1.0) {
                        nuc->lev_Q[nuc->n_lev++] = -qi;
                    }
                    /* Skip interpolation + data records */
                    int nr_q  = nd_ival(line + 44);
                    int np_q  = nd_ival(line + 55);
                    int n_skip = (2 * nr_q + 5) / 6
                               + (np_q + 2) / 3;
                    int s;
                    KA_GUARD(gs, 2000);
                    for (s = 0; s < n_skip && gs--; s++) {
                        if (nd_read(line, f)) break;
                    }
                }
            } else {
                /* All other MF3: parse cross-section data */
                int np = nd_tab1(f, tmp_e, tmp_y, MO_MAX_EPTS);

                if (mt == 1) {
                    /* MT1: total — this IS the master grid */
                    nuc->n_pts = np;
                    int i;
                    for (i = 0; i < np; i++) {
                        nuc->E[i]    = tmp_e[i];
                        nuc->xs_t[i] = tmp_y[i];
                    }
                } else if (mt == 2 && nuc->n_pts > 0) {
                    /* MT2: elastic — interpolate onto master grid */
                    nd_interp(tmp_e, tmp_y, np,
                              nuc->E, nuc->xs_el, nuc->n_pts);
                } else if (mt == 18 && nuc->n_pts > 0) {
                    /* MT18: fission — interpolate onto master grid */
                    nd_interp(tmp_e, tmp_y, np,
                              nuc->E, nuc->xs_f, nuc->n_pts);
                    got_fiss = 1;
                } else if (mt == 102 && nuc->n_pts > 0) {
                    /* MT102: capture — interpolate onto master grid */
                    nd_interp(tmp_e, tmp_y, np,
                              nuc->E, nuc->xs_a, nuc->n_pts);
                    got_cap = 1;
                }
            }
            /* Skip to SEND (catches any remaining records) */

            KA_GUARD(g2, 10000);
            while (g2-- && nd_read(line, f) == 0) {
                if (is_send(line)) break;
            }
            continue;
        }
    }

    fclose(f);

    if (nuc->n_pts == 0) {
        fprintf(stderr, "moa: no MF3/MT1 data in %s\n", path);
        return -1;
    }

    /* Interpolate nu-bar onto master grid if tabulated */
    if (n_nu > 0) {
        nd_interp(nu_e, nu_y, n_nu,
                  nuc->E, nuc->nu, nuc->n_pts);
    }

    /* If absorption not loaded, compute: sig_a = sig_t - sig_el */
    if (!got_cap) {
        int i;
        for (i = 0; i < nuc->n_pts; i++) {
            nuc->xs_a[i] = nuc->xs_t[i] - nuc->xs_el[i];
            if (nuc->xs_a[i] < 0.0) nuc->xs_a[i] = 0.0;
        }
    }

    /* If nu-bar not loaded, use constant for fissile nuclides */
    if (n_nu == 0 && got_fiss) {
        int i;
        for (i = 0; i < nuc->n_pts; i++) {
            nuc->nu[i] = 2.43;
        }
    }

    if (nuc->awr < 1.0) {
        nuc->awr = (double)(za % 1000);
    }

    if (nuc->n_res > 0) {
        printf("moa: ZA=%d: %d resonances (LRF=%d) in [%.1f, %.1f] eV, "
               "SPI=%.1f, AP=%.4f\n",
               za, nuc->n_res, nuc->lrf, nuc->EL, nuc->EH,
               nuc->SPI, nuc->AP);
    }
    if (nuc->n_lev > 0) {
        printf("moa: ZA=%d: %d discrete inelastic levels, "
               "Q1=%.0f eV, Q%d=%.0f eV\n",
               za, nuc->n_lev,
               nuc->lev_Q[0], nuc->n_lev,
               nuc->lev_Q[nuc->n_lev - 1]);
    }
    if (nuc->n_mu > 0) {
        printf("moa: ZA=%d: %d mu-bar points, "
               "mu(%.0f eV)=%.3f, mu(%.0f eV)=%.3f\n",
               za, nuc->n_mu,
               nuc->mu_E[0], nuc->mu_bar[0],
               nuc->mu_E[nuc->n_mu - 1],
               nuc->mu_bar[nuc->n_mu - 1]);
    }

    return 0;
}
