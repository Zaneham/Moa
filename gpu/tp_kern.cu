/* tp_kern.cu -- GPU transport kernel (GODIVA edition)
 *
 * Monte Carlo neutron transport on AMD GPUs via BarraCUDA.
 * One thread per particle, because neutrons are the original
 * social distancers -- they don't interact with each other,
 * only with the geometry and nuclear data.
 *
 * All float, all the time. Double precision is for people
 * who measure curtains before buying them. We're doing
 * Monte Carlo -- the statistical error dwarfs the float
 * precision loss like a moa dwarfs a kiwi.
 *
 * GODIVA -- Go Diva! First real-world BarraCUDA workload.
 * If this works, we've built a nuclear reactor simulator
 * that compiles through a hand-rolled CUDA compiler onto
 * AMD hardware. The audacity is the point.
 *
 * Everything is inlined into one monolithic kernel because
 * BarraCUDA doesn't do device function calls yet. NVIDIA's
 * compiler would inline everything anyway, so we're just
 * doing its job for it. More honest this way.
 *
 * Target: MI300X (gfx942, CDNA3) via BarraCUDA -> .hsaco
 * Style:  JPL-style, HLASM naming, British/Kiwi absurdism
 */

/* ---- Constants ---- */

#define GP_PI     3.14159265f
#define GP_EPS    1.0e-4f   /* nudge distance (100 um) */
#define GP_EMIN   1.0e-5f
#define GP_WMIN   0.25f
#define GP_WSRV   1.0f
#define GP_INF    1.0e30f
#define GP_MHIST  4096
#define GP_MAXH   16
#define GP_MAXN   8
#define GP_K2F    4.8261e-8f

#define GP_CHK(i, n) ((unsigned)(i) >= (unsigned)(n))

/* ---- GPU Struct Definitions ---- */

/* xorwow RNG -- cuRAND's default, period 2^192 - 2^32. */
struct gp_rng_t {
    unsigned s0;
    unsigned s1;
    unsigned s2;
    unsigned s3;
    unsigned s4;
    unsigned d;
};

/* Particle state */
struct gp_part_t {
    float x;
    float y;
    float z;
    float u;
    float v;
    float w;
    float E;
    float wt;
    int   cell;
    int   alive;
};

/* General quadric surface */
struct gp_surf_t {
    int   type;
    float A;
    float B;
    float C;
    float D;
    float E;
    float F;
    float G;
    float H;
    float J;
    float K;
};

/* Cell -- halfspace surfaces flattened into csur[]/csen[] */
struct gp_cell_t {
    int   n_half;
    int   mat;
    float rho;
    int   sh_off;
};

/* Material -- nuclide IDs/fracs flattened into mnid[]/mfrc[] */
struct gp_mat_t {
    int n_nuc;
    int mn_off;
    int pad;
};

/* Nuclide header -- scalar metadata */
struct gp_nhdr_t {
    int   ZA;
    float awr;
    int   n_pts;
    int   lrf;
    float SPI;
    float AP;
    float EL;
    float EH;
    int   n_res;
    int   xs_off;
    int   res_off;
    int   n_mu;
    int   mu_off;
    int   n_lev;
    int   lev_off;
    int   pad;
};

/* Resonance parameters */
struct gp_res_t {
    float Er;
    float AJ;
    float Gn;
    float Gg;
    float Gfa;
    float Gfb;
};

/* Cross-section result (stack-local) */
struct gp_xs_t {
    float sig_t;
    float sig_a;
    float sig_f;
    float sig_el;
    float sig_in;
    float nu_bar;
    int   fiss_za;
};

/* Fission bank site */
struct gp_site_t {
    float x;
    float y;
    float z;
    float E;
};


/* ---- Helper Macros ----
 * Single-line because BarraCUDA's preprocessor doesn't do
 * backslash-newline continuation. Like writing telegrams
 * but with semicolons instead of STOP. */

/* xorwow next -- Marsaglia's xorshift + Weyl sequence */
#define GP_RNXT(r, _out) do { unsigned _t = (r).s4; unsigned _s = (r).s0; (r).s4 = (r).s3; (r).s3 = (r).s2; (r).s2 = (r).s1; (r).s1 = _s; _t = _t ^ (_t >> 2); _t = _t ^ (_t << 1); _t = _t ^ (_s ^ (_s << 4)); (r).s0 = _t; (r).d = (r).d + 362437u; (_out) = _t + (r).d; } while(0)

/* Uniform float in [0, 1) */
#define GP_RDBL(r, _out) do { unsigned _rv; GP_RNXT(r, _rv); (_out) = (float)(_rv >> 8) * (1.0f / 16777216.0f); } while(0)

/* Quadric surface evaluation */
#define GP_EVAL(s, px, py, pz, _out) (_out) = (s).A*(px)*(px) + (s).B*(py)*(py) + (s).C*(pz)*(pz) + (s).D*(px)*(py) + (s).E*(py)*(pz) + (s).F*(px)*(pz) + (s).G*(px) + (s).H*(py) + (s).J*(pz) + (s).K

/* Advance particle along direction */
#define GP_MOVE(p, d) do { (p).x += (d) * (p).u; (p).y += (d) * (p).v; (p).z += (d) * (p).w; } while(0)


/* ==== THE KERNEL ====
 * One function to rule them all, one function to find them,
 * one function to bring them all, and in the wavefront bind them.
 *
 * Everything from geometry to physics to tallying, inlined
 * into a single __global__. This is what happens when your
 * compiler is younger than the code it compiles. */

__global__ void gp_kern(
    gp_part_t  *parts,
    gp_rng_t   *rngs,
    gp_site_t  *fbank,
    int        *n_fbank,
    gp_surf_t  *surfs,
    gp_cell_t  *cells,
    int        *csur,
    int        *csen,
    gp_mat_t   *mats,
    int        *mnid,
    float      *mfrc,
    gp_nhdr_t  *nhdr,
    float      *nE,
    float      *nst,
    float      *nsa,
    float      *nsf,
    float      *nse,
    float      *nnu,
    gp_res_t   *nres,
    int         n_part,
    int         n_surf,
    int         n_cell,
    int         n_mat,
    int         n_nuc,
    float       k_eff,
    int         fb_max,
    float      *muE,
    float      *muV,
    float      *levQ,
    int        *dbg
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n_part) {
    /* Guard: wrap entire body in if(tid < n_part) rather than early return,
     * because BarraCUDA's divergent-return codegen is broken on GFX942.
     * Same semantics, different exec mask pattern. */

    /* Load per-thread state. One particle, one RNG, one destiny. */
    gp_part_t p;
    gp_rng_t  rng;
    p   = parts[tid];
    rng = rngs[tid];

    /* Per-nuclide elastic xs saved during macro xs for scatter */
    float nuc_sel[GP_MAXN];
    float nuc_awr[GP_MAXN];
    int   nuc_nid[GP_MAXN];
    float nuc_sin[GP_MAXN];
    int   nn_sav = 0;

    /* XS stash: first-collision values saved for post-loop diagnostic */
    int d_sigt = 0;
    int d_siga = 0;
    int d_sigf = 0;
    int d_sel  = 0;
    int d_nu   = 0;
    int d_E0   = 0;

    /* ---- Transport loop ---- */
    int step = GP_MHIST;
    int n_xcross = 0;  /* surface crossings counter */
    int n_xcoll  = 0;  /* collision counter */
    int n_xinf   = 0;  /* inf d_surf while INSIDE cell (real bug) */
    int n_xesc   = 0;  /* inf d_surf while OUTSIDE cell (escaped) */
#ifndef GP_STUB
    while (p.alive && step > 0) {
        step--;

        /* ============================================ */
        /* ---- MACROSCOPIC XS (inline gp_xmac) ----   */
        /* ============================================ */
        gp_xs_t xs;
        xs.sig_t  = 0.0f;
        xs.sig_a  = 0.0f;
        xs.sig_f  = 0.0f;
        xs.sig_el = 0.0f;
        xs.sig_in = 0.0f;
        xs.nu_bar = 0.0f;
        xs.fiss_za = 92235;
        nn_sav = 0;

        {
                int mi = cells[p.cell].mat;
            if (mi >= 0 && !GP_CHK(mi, n_mat)) {
                float rho = cells[p.cell].rho;
                int moff  = mats[mi].mn_off;
                int nn    = mats[mi].n_nuc;
                float max_sf = 0.0f;
                int ni;

                for (ni = 0; ni < nn && ni < GP_MAXN; ni++) {
                    int nid = mnid[moff + ni];
                    if (GP_CHK(nid, n_nuc)) continue;

                    /* ---- XS LOOKUP (inline gp_look) ---- */
                    float m_st;
                    float m_sa;
                    float m_sf;
                    float m_sel;
                    float m_nu;
                    {
                        int xoff = nhdr[nid].xs_off;
                        int np   = nhdr[nid].n_pts;
                        int lo   = 0;
                        int hi   = np - 1;

                        /* Default to first grid point */
                        m_st  = nst[xoff];
                        m_sa  = nsa[xoff];
                        m_sf  = nsf[xoff];
                        m_sel = nse[xoff];
                        m_nu  = nnu[xoff];

                        if (hi > 0 && p.E > nE[xoff]) {
                            if (p.E >= nE[xoff + hi]) {
                                /* Clamp to last grid point */
                                m_st  = nst[xoff + hi];
                                m_sa  = nsa[xoff + hi];
                                m_sf  = nsf[xoff + hi];
                                m_sel = nse[xoff + hi];
                                m_nu  = nnu[xoff + hi];
                            } else {
                                /* Linear scan */
                                int sj;
                                for (sj = 1; sj < np; sj++) {
                                    if (p.E < nE[xoff + sj]) break;
                                    lo = sj;
                                }
                                hi = lo + 1;
                                if (hi >= np) hi = np - 1;
                                float dE = nE[xoff + hi] - nE[xoff + lo];
                                float f = 0.0f;
                                if (dE > 1.0e-30f)
                                    f = (p.E - nE[xoff + lo]) / dE;
                                m_st  = nst[xoff+lo] + f*(nst[xoff+hi]-nst[xoff+lo]);
                                m_sa  = nsa[xoff+lo] + f*(nsa[xoff+hi]-nsa[xoff+lo]);
                                m_sf  = nsf[xoff+lo] + f*(nsf[xoff+hi]-nsf[xoff+lo]);
                                m_sel = nse[xoff+lo] + f*(nse[xoff+hi]-nse[xoff+lo]);
                                m_nu  = nnu[xoff+lo] + f*(nnu[xoff+hi]-nnu[xoff+lo]);
                            }
                        }

                        /* ---- SLBW RESONANCE (inline gp_rslv) ---- */
                        if (nhdr[nid].n_res > 0 &&
                            p.E >= nhdr[nid].EL && p.E <= nhdr[nid].EH)
                        {
                            float rst = 0.0f;
                            float rsf = 0.0f;
                            float rsg = 0.0f;
                            float rsel = 0.0f;

                            float awri = nhdr[nid].awr;
                            if (awri >= 1.0f) {
                                float ecm = p.E * awri / (awri + 1.0f);
                                if (ecm >= 1.0e-20f) {
                                    float k2 = GP_K2F * ecm;
                                    float kv = sqrtf(k2);
                                    float a_fm = fabsf(nhdr[nid].AP) * 10.0f;
                                    float rho_v = kv * a_fm;
                                    float pikq = GP_PI / (k2 * 100.0f);
                                    float sin_phi = sinf(rho_v);
                                    float pot = 4.0f * pikq * sin_phi * sin_phi;
                                    float cos2p = cosf(2.0f * rho_v);
                                    float sin2p = sinf(2.0f * rho_v);
                                    float gfac = 2.0f * (2.0f * nhdr[nid].SPI + 1.0f);
                                    if (gfac < 1.0f) gfac = 1.0f;
                                    float sum_t = 0.0f;
                                    float sum_f = 0.0f;
                                    float sum_g = 0.0f;
                                    int roff = nhdr[nid].res_off;
                                    int nr = nhdr[nid].n_res;
                                    int ri;
                                    for (ri = 0; ri < nr; ri++) {
                                        float Er   = nres[roff + ri].Er;
                                        float AJ   = nres[roff + ri].AJ;
                                        float Gn_r = nres[roff + ri].Gn;
                                        float Gg   = nres[roff + ri].Gg;
                                        float Gfa  = nres[roff + ri].Gfa;
                                        float Gfb  = nres[roff + ri].Gfb;
                                        if (AJ < 0.0f) AJ = -AJ;
                                        if (Gn_r < 0.0f) Gn_r = -Gn_r;
                                        if (Gg < 0.0f) Gg = -Gg;
                                        float Gf = sqrtf(Gfa*Gfa + Gfb*Gfb);
                                        float gJ = (2.0f*AJ + 1.0f) / gfac;
                                        float Er_abs = fabsf(Er);
                                        float Gnv = Gn_r;
                                        if (Er_abs > 1.0e-30f)
                                            Gnv = Gn_r * sqrtf(p.E / Er_abs);
                                        float Gt = Gnv + Gg + Gf;
                                        float dEr = p.E - Er;
                                        float Dv = dEr*dEr + 0.25f*Gt*Gt;
                                        if (Dv < 1.0e-40f) continue;
                                        float psi = 0.25f*Gt*Gt / Dv;
                                        float chi = dEr * 0.5f * Gt / Dv;
                                        sum_t += gJ*(Gnv/Gt)*(cos2p*psi + sin2p*chi);
                                        sum_f += gJ * Gnv * Gf / Dv;
                                        sum_g += gJ * Gnv * Gg / Dv;
                                    }
                                    rst  = pot + 4.0f * pikq * sum_t;
                                    rsf  = pikq * sum_f;
                                    rsg  = pikq * sum_g;
                                    rsel = rst - rsf - rsg;
                                    if (rsel < 0.0f) rsel = 0.0f;
                                }
                            }
                            m_st  += rst;
                            m_sf  += rsf;
                            m_sa  += rsg;
                            m_sel += rsel;
                        }
                    } /* end xs lookup */

                    /* Accumulate macroscopic xs */
                    float rho_f = rho * mfrc[moff + ni];
                    xs.sig_t  += rho_f * m_st;
                    xs.sig_a  += rho_f * m_sa;
                    xs.sig_f  += rho_f * m_sf;
                    xs.sig_el += rho_f * m_sel;
                    if (m_sf > 0.0f)
                        xs.nu_bar += m_nu * rho_f * m_sf;
                    if (rho_f * m_sf > max_sf) {
                        max_sf = rho_f * m_sf;
                        xs.fiss_za = nhdr[nid].ZA;
                    }

                    /* Save per-nuclide elastic + inelastic for scatter */
                    if (nn_sav < GP_MAXN) {
                        nuc_sel[nn_sav] = rho_f * m_sel;
                        float m_sin = m_st - m_sel - m_sa - m_sf;
                        if (m_sin < 0.0f) m_sin = 0.0f;
                        nuc_sin[nn_sav] = rho_f * m_sin;
                        nuc_awr[nn_sav] = nhdr[nid].awr;
                        nuc_nid[nn_sav] = nid;
                        nn_sav++;
                    }
                } /* end nuclide loop */

                /* Normalise nu-bar */
                if (xs.sig_f > 1.0e-30f)
                    xs.nu_bar /= xs.sig_f;
                xs.sig_in = xs.sig_t - xs.sig_el - xs.sig_a - xs.sig_f;
                if (xs.sig_in < 0.0f) xs.sig_in = 0.0f;
                if (xs.sig_t < 1.0e-20f) xs.sig_t = 1.0e-20f;

                /* DIAG: stash first-collision XS for post-loop write */
                if (tid == 0 && step == GP_MHIST - 1) {
                    d_sigt = (int)(xs.sig_t  * 1000000.0f);
                    d_siga = (int)(xs.sig_a  * 1000000.0f);
                    d_sigf = (int)(xs.sig_f  * 1000000.0f);
                    d_sel  = (int)(xs.sig_el * 1000000.0f);
                    d_nu   = (int)(xs.nu_bar * 1000.0f);
                    d_E0   = (int)p.E;
                }
            } else {
                /* void cell -- kill particle */
                p.alive = 0;
                break;
            }
        }


        /* ============================================ */
        /* ---- COLLISION DISTANCE ----                 */
        /* ============================================ */
        float xi_coll;
        GP_RDBL(rng, xi_coll);
        if (xi_coll < 1.0e-30f) xi_coll = 1.0e-30f;
        float d_coll = -logf(xi_coll) / xs.sig_t;


        /* ============================================ */
        /* ---- NEAREST SURFACE (inline gp_next) ----   */
        /* ============================================ */
        float d_surf = GP_INF;
        int   s_hit  = -1;
        if (!GP_CHK(p.cell, 256)) {
            int nh  = cells[p.cell].n_half;
            int off = cells[p.cell].sh_off;
            int sj;
            for (sj = 0; sj < nh && sj < GP_MAXH; sj++) {
                int si = csur[off + sj];
                if (GP_CHK(si, n_surf)) continue;

                /* ---- Ray-surface distance (inline gp_dist) ---- */
                float ds;
                {
                    float da = surfs[si].A*p.u*p.u + surfs[si].B*p.v*p.v
                             + surfs[si].C*p.w*p.w + surfs[si].D*p.u*p.v
                             + surfs[si].E*p.v*p.w + surfs[si].F*p.u*p.w;
                    float db = 2.0f*(surfs[si].A*p.x*p.u + surfs[si].B*p.y*p.v
                             + surfs[si].C*p.z*p.w)
                             + surfs[si].D*(p.x*p.v + p.y*p.u)
                             + surfs[si].E*(p.y*p.w + p.z*p.v)
                             + surfs[si].F*(p.x*p.w + p.z*p.u)
                             + surfs[si].G*p.u + surfs[si].H*p.v
                             + surfs[si].J*p.w;
                    float dc;
                    GP_EVAL(surfs[si], p.x, p.y, p.z, dc);

                    /* Root selection: try BOTH roots, take smaller
                     * positive one. Both-roots version is correct
                     * (tp_fhst float CPU gives k=0.997). */
                    ds = GP_INF;
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
                            float d1 = GP_INF;
                            float d2 = GP_INF;
                            if (t1 > 0.0f) d1 = t1;
                            if (t2 > 0.0f) d2 = t2;
                            if (d1 < d2) ds = d1;
                            else ds = d2;
                        }
                    }
                }
                if (ds < d_surf) {
                    d_surf = ds;
                    s_hit  = si;
                }
            }
        }

        /* ---- Discriminant clamp fallback ----
         * If d_surf is INF and the particle is in a material
         * cell, float32 cancellation killed the discriminant.
         * A particle inside a convex cell MUST intersect its
         * boundary -- disc < 0 is always float error here.
         * Redo intersection with disc clamped to max(0,disc).
         * Only fires ~20-35 times per 10K batch, so the
         * branch cost is negligible. */
        if (d_surf > 1.0e20f && cells[p.cell].mat >= 0) {
            int nh  = cells[p.cell].n_half;
            int off = cells[p.cell].sh_off;
            int sj;
            for (sj = 0; sj < nh && sj < GP_MAXH; sj++) {
                int si = csur[off + sj];
                if (GP_CHK(si, n_surf)) continue;
                float da = surfs[si].A*p.u*p.u + surfs[si].B*p.v*p.v
                         + surfs[si].C*p.w*p.w + surfs[si].D*p.u*p.v
                         + surfs[si].E*p.v*p.w + surfs[si].F*p.u*p.w;
                float db = 2.0f*(surfs[si].A*p.x*p.u + surfs[si].B*p.y*p.v
                         + surfs[si].C*p.z*p.w)
                         + surfs[si].D*(p.x*p.v + p.y*p.u)
                         + surfs[si].E*(p.y*p.w + p.z*p.v)
                         + surfs[si].F*(p.x*p.w + p.z*p.u)
                         + surfs[si].G*p.u + surfs[si].H*p.v
                         + surfs[si].J*p.w;
                float dc;
                GP_EVAL(surfs[si], p.x, p.y, p.z, dc);
                float ds = GP_INF;
                if (fabsf(da) < 1.0e-20f) {
                    if (fabsf(db) >= 1.0e-20f) {
                        float tl = -dc / db;
                        if (tl > 0.0f) ds = tl;
                    }
                } else {
                    float disc = db*db - 4.0f*da*dc;
                    if (disc < 0.0f) disc = 0.0f;
                    float sq = sqrtf(disc);
                    float t1 = (-db - sq) / (2.0f * da);
                    float t2 = (-db + sq) / (2.0f * da);
                    float d1 = GP_INF;
                    float d2 = GP_INF;
                    if (t1 > 0.0f) d1 = t1;
                    if (t2 > 0.0f) d2 = t2;
                    if (d1 < d2) ds = d1;
                    else ds = d2;
                }
                if (ds < d_surf) {
                    d_surf = ds;
                    s_hit  = si;
                }
            }
        }

        /* ============================================ */
        /* ---- SURFACE or COLLISION ----               */
        /* ============================================ */
        if (d_surf < d_coll) {
            /* ---- Surface crossing ---- */
            n_xcross++;
            atomicAdd(&dbg[n_part - 3], 1);  /* DIAG: total crossings */
            GP_MOVE(p, d_surf);

            /* Nudge past surface */
            p.x += GP_EPS * p.u;
            p.y += GP_EPS * p.v;
            p.z += GP_EPS * p.w;

            /* ---- Find new cell (inline gp_find) ---- */
            int nc = -1;
            {
                int ci;
                for (ci = 0; ci < n_cell; ci++) {
                    int fnh = cells[ci].n_half;
                    int foff = cells[ci].sh_off;
                    int inside = 1;
                    if (fnh == 0) continue;
                    int fj;
                    for (fj = 0; fj < fnh && fj < GP_MAXH; fj++) {
                        int fsi = csur[foff + fj];
                        if (GP_CHK(fsi, n_surf)) { inside = 0; break; }
                        float fval;
                        GP_EVAL(surfs[fsi], p.x, p.y, p.z, fval);
                        int fsn = csen[foff + fj];
                        if (fsn < 0 && fval > 0.0f) { inside = 0; break; }
                        if (fsn > 0 && fval < 0.0f) { inside = 0; break; }
                    }
                    if (inside) { nc = ci; break; }
                }
            }

            if (nc < 0) {
                atomicAdd(&dbg[n_part - 9], 1);  /* DIAG: leak (no cell) */
                p.alive = 0;
                break;  /* leaked */
            }
            if (nc == p.cell) {
                atomicAdd(&dbg[n_part - 10], 1); /* DIAG: phantom (same cell) */
            }
            p.cell = nc;
            if (cells[nc].mat < 0) {
                atomicAdd(&dbg[n_part - 9], 1);  /* DIAG: leak (void cell) */
                p.alive = 0;
                break;  /* void cell */
            }
        } else {
            /* ---- Collision ---- */
            n_xcoll++;
            atomicAdd(&dbg[n_part - 1], 1);  /* DIAG: total collisions */
            if (d_surf > 1.0e20f) {
                /* Check if particle is actually inside its cell.
                 * Evaluate the first surface of the current cell. */
                int ch_off = cells[p.cell].sh_off;
                int ch_si  = csur[ch_off];
                float ch_v;
                GP_EVAL(surfs[ch_si], p.x, p.y, p.z, ch_v);
                int ch_sn = csen[ch_off];
                /* sense<0 means inside when eval<0 */
                if ((ch_sn < 0 && ch_v > GP_EPS) ||
                    (ch_sn > 0 && ch_v < -GP_EPS))
                    n_xesc++;   /* escaped: outside cell boundary */
                else
                    n_xinf++;   /* true bug: inside cell but INF */
            }
            GP_MOVE(p, d_coll);

            /* ---- Post-collision cell sanity check ----
             * Last resort: if particle is outside its cell
             * after collision (float overshoot), kill it.
             * The discriminant clamp above should prevent
             * most inf_inside cases, but belt-and-braces. */
            {
                int ch_nh  = cells[p.cell].n_half;
                int ch_off = cells[p.cell].sh_off;
                int ch_ok  = 1;
                int fj;
                for (fj = 0; fj < ch_nh && fj < GP_MAXH; fj++) {
                    int fsi = csur[ch_off + fj];
                    if (GP_CHK(fsi, n_surf)) continue;
                    float fv;
                    GP_EVAL(surfs[fsi], p.x, p.y, p.z, fv);
                    int fsn = csen[ch_off + fj];
                    if (fsn < 0 && fv > 1.0e-3f) { ch_ok = 0; break; }
                    if (fsn > 0 && fv < -1.0e-3f) { ch_ok = 0; break; }
                }
                if (!ch_ok) {
                    p.alive = 0;
                    break;
                }
            }

            /* ---- Fission banking (inline gp_coll) ---- */
            if (xs.sig_f > 1.0e-30f) {
                float nu_sites = xs.nu_bar * p.wt * xs.sig_f
                               / (xs.sig_t * k_eff);
                int n_new = (int)nu_sites;
                float frac = nu_sites - (float)n_new;
                float xi_frac;
                GP_RDBL(rng, xi_frac);
                if (xi_frac < frac) n_new++;
                atomicAdd(&dbg[n_part - 2], n_new); /* DIAG: total sites */

                int fk;
                for (fk = 0; fk < n_new; fk++) {
                    int slot = atomicAdd(n_fbank, 1);
                    if (slot < fb_max) {
                        /* ---- Watt spectrum (inline gp_watt) ---- */
                        float wE = 2.0e6f;
                        {
                            /* Watt params by ZA */
                            float wa = 988000.0f;
                            float wb = 2.249e-6f;
                            int wza = xs.fiss_za;
                            if (wza == 92238 || wza == 92234) {
                                wa = 895000.0f;
                                wb = 3.295e-6f;
                            } else if (wza == 94239 || wza == 94240 ||
                                       wza == 94241 || wza == 94242) {
                                wa = 966000.0f;
                                wb = 2.842e-6f;
                            }
                            int wg = 1000;
                            while (wg > 0) {
                                wg--;
                                float wr1;
                                float wr2;
                                GP_RDBL(rng, wr1);
                                GP_RDBL(rng, wr2);
                                if (wr1 < 1.0e-30f) wr1 = 1.0e-30f;
                                if (wr2 < 1.0e-30f) wr2 = 1.0e-30f;
                                float wx = -wa * logf(wr1);
                                float wy = -logf(wr2);
                                float wsq = wy - (sqrtf(wb * wx) - 1.0f);
                                if (wsq * wsq <= wb * wx) {
                                    wE = wx;
                                    break;
                                }
                            }
                        }
                        fbank[slot].x = p.x;
                        fbank[slot].y = p.y;
                        fbank[slot].z = p.z;
                        fbank[slot].E = wE;
                    }
                }
            }

            /* ---- Survival biasing ---- */
            float sig_abs = xs.sig_a + xs.sig_f;
            if (sig_abs > xs.sig_t) {
                sig_abs = xs.sig_t - xs.sig_el;
                if (sig_abs < 0.0f) sig_abs = 0.0f;
            }
            p.wt *= (1.0f - sig_abs / xs.sig_t);

            /* ---- Scatter (inline gp_scat) ----
             * Elastic nuclide selection + mu-bar lookup live at the
             * TOP LEVEL (not inside an else branch). BarraCUDA's PTX
             * codegen silently drops code inside else branches of
             * large if/else blocks -- discovered empirically after
             * three rounds of variable renaming didn't fix it.
             *
             * Structure: select elastic nuclide, compute mu-bar,
             * THEN branch for inelastic (levels/evap + isotropic)
             * vs elastic (delta-Wigner with pre-computed mu-bar).
             *
             * Discrete levels: for U-235 at 1-2 MeV, Q1=77 eV
             * drops ~0.1% energy. Evaporation was losing ~75%.
             * Like estimating a parking fine by the cost of the car. */
            {
                /* ---- ELASTIC NUCLIDE SELECTION (always) ---- */
                float sA = 1.0f;
                int s_nid = -1;
                {
                    float xi_scat;
                    GP_RDBL(rng, xi_scat);
                    float scum = 0.0f;
                    int si;
                    for (si = 0; si < nn_sav; si++) {
                        scum += nuc_sel[si];
                        if (scum >= xi_scat * xs.sig_el
                            || si == nn_sav - 1) {
                            sA = nuc_awr[si];
                            s_nid = nuc_nid[si];
                            break;
                        }
                    }
                }

                /* ---- MU-BAR LOOKUP (always) ----
                 * Binary search + interp, natural if/else. */
                float mb = 0.0f;
                if (s_nid >= 0 && s_nid < n_nuc) {
                    int nm = nhdr[s_nid].n_mu;
                    int mo = nhdr[s_nid].mu_off;
                    if (nm == 1) {
                        mb = muV[mo];
                    } else if (nm > 1) {
                        int mlo = 0;
                        int mhi = nm - 1;
                        int mg = 20;
                        while (mlo + 1 < mhi && mg > 0) {
                            mg--;
                            int mm = (mlo + mhi) / 2;
                            if (p.E < muE[mo + mm])
                                mhi = mm;
                            else
                                mlo = mm;
                        }
                        float dE = muE[mo + mhi] - muE[mo + mlo];
                        if (dE > 1.0e-30f) {
                            float f = (p.E - muE[mo + mlo]) / dE;
                            if (f < 0.0f) f = 0.0f;
                            if (f > 1.0f) f = 1.0f;
                            mb = muV[mo + mlo] + f * (muV[mo + mhi] - muV[mo + mlo]);
                        } else {
                            mb = muV[mo + mlo];
                        }
                    }
                }

                /* ---- SCATTER TYPE DECISION ----
                 * Natural if/else now that NV_MAX_PCOPY is 8192.
                 * The root cause of the "else dropping" bug was
                 * PHI copy buffer overflow at 2048 -- every
                 * dropped copy meant a variable silently kept
                 * its wrong value at a merge point. Three days
                 * of kernel contortions, and it was a #define. */
                float sig_scat = xs.sig_el + xs.sig_in;
                float xi_br;
                GP_RDBL(rng, xi_br);

                if (sig_scat > 1.0e-30f &&
                    xs.sig_in > 1.0e-30f &&
                    xi_br * sig_scat > xs.sig_el) {
                    /* ---- INELASTIC PATH ---- */
                    atomicAdd(&dbg[n_part - 7], 1);
                    float iA = 1.0f;
                    int i_nid = -1;
                    {
                        float xi_is;
                        GP_RDBL(rng, xi_is);
                        float iscum = 0.0f;
                        int isi;
                        for (isi = 0; isi < nn_sav; isi++) {
                            iscum += nuc_sin[isi];
                            if (iscum >= xi_is * xs.sig_in
                                || isi == nn_sav - 1) {
                                iA = nuc_awr[isi];
                                i_nid = nuc_nid[isi];
                                break;
                            }
                        }
                    }

                    /* Count accessible discrete levels */
                    int n_acc = 0;
                    int nlev = 0;
                    int loff = 0;
                    if (i_nid >= 0) {
                        if (i_nid < n_nuc) {
                            nlev = nhdr[i_nid].n_lev;
                            loff = nhdr[i_nid].lev_off;
                        }
                    }
                    int li;
                    for (li = 0; li < nlev; li++) {
                        if (li >= 40) break;
                        if (levQ[loff + li] * (iA + 1.0f) / iA < p.E)
                            n_acc++;
                    }
                    if (n_acc > 0) {
                        /* Discrete level: small energy nudge */
                        atomicAdd(&dbg[n_part - 8], 1);
                        float xi_lev;
                        GP_RDBL(rng, xi_lev);
                        int pick = (int)(xi_lev * (float)n_acc);
                        if (pick >= n_acc) pick = n_acc - 1;
                        p.E -= levQ[loff + pick] * (iA + 1.0f) / iA;
                        if (p.E < GP_EMIN) p.E = GP_EMIN;
                    } else {
                        /* Evaporation: continuum fallback */
                        atomicAdd(&dbg[n_part - 9], 1);
                        float T = sqrtf(8.0e6f * p.E / iA);
                        float wr1;
                        float wr2;
                        GP_RDBL(rng, wr1);
                        GP_RDBL(rng, wr2);
                        if (wr1 < 1.0e-30f) wr1 = 1.0e-30f;
                        if (wr2 < 1.0e-30f) wr2 = 1.0e-30f;
                        float E_out = -T * logf(wr1 * wr2);
                        if (E_out >= p.E) E_out = p.E * 0.5f;
                        p.E = E_out;
                    }

                    /* Isotropic direction */
                    float imu;
                    float iphi;
                    float ist;
                    GP_RDBL(rng, imu);
                    imu = 2.0f * imu - 1.0f;
                    GP_RDBL(rng, iphi);
                    iphi = 2.0f * GP_PI * iphi;
                    ist = sqrtf(1.0f - imu * imu);
                    p.u = ist * cosf(iphi);
                    p.v = ist * sinf(iphi);
                    p.w = imu;
                } else {
                    /* ---- ELASTIC PATH ----
                     * Delta-Wigner model: fraction mb is forward
                     * (no change), rest is compound nucleus
                     * (isotropic CM with A-body kinematics). */
                    float xi_mu;
                    GP_RDBL(rng, xi_mu);
                    atomicAdd(&dbg[n_part - 6], (int)(mb * 1000.0f));

                    if (xi_mu < mb) {
                        /* Forward scatter: no direction/energy change */
                        atomicAdd(&dbg[n_part - 4], 1);
                    } else {
                        /* Compound nucleus */
                        atomicAdd(&dbg[n_part - 5], 1);
                        float mu_cm;
                        GP_RDBL(rng, mu_cm);
                        mu_cm = 2.0f * mu_cm - 1.0f;
                        float Ap1 = sA + 1.0f;
                        p.E *= (sA*sA + 1.0f + 2.0f*sA*mu_cm) / (Ap1 * Ap1);

                        float emu;
                        float ephi;
                        float est;
                        GP_RDBL(rng, emu);
                        emu = 2.0f * emu - 1.0f;
                        GP_RDBL(rng, ephi);
                        ephi = 2.0f * GP_PI * ephi;
                        est = sqrtf(1.0f - emu * emu);
                        p.u = est * cosf(ephi);
                        p.v = est * sinf(ephi);
                        p.w = emu;
                    }
                }

                if (p.E < GP_EMIN)
                    p.alive = 0;
            }
        }


        /* ---- Russian roulette (inline gp_rlet) ----
         * Below weight cutoff, flip a biased coin.
         * Win: weight boosted. Lose: particle dies. */
        if (p.alive) {
            if (p.wt < GP_WMIN) {
                if (p.wt <= 0.0f) {
                    p.alive = 0;
                } else {
                    float xi_rlet;
                    GP_RDBL(rng, xi_rlet);
                    if (xi_rlet * GP_WSRV < p.wt) {
                        p.wt = GP_WSRV;
                    } else {
                        p.alive = 0;
                    }
                }
            }
        }
    } /* end transport loop */
#endif /* GP_STUB */

    /* Encode: steps[0:11] | crossings[12:15] | inf[16:19] | esc[20:23]
     * Slots 0-7 reserved for tid=0 diagnostics, last 12 for atomics.
     * Nested ifs avoid compound && which BarraCUDA spills as
     * pred-to-local-store -- PTX doesn't allow st.u32 of %pred. */
    if (tid >= 8) {
        if (tid < n_part - 11) {
            dbg[tid] = ((GP_MHIST - step) & 0xFFF)
                     | ((n_xcross & 0xF) << 12)
                     | ((n_xinf & 0xF) << 16)
                     | ((n_xesc & 0xF) << 20);
        }
    }

    /* DIAG: post-loop, tid=0 only, slots 0-7 reserved.
     * Written AFTER dbg[tid] so thread 0's final word wins. */
    if (tid == 0) {
        dbg[0] = d_sigt;
        dbg[1] = d_siga;
        dbg[2] = d_sigf;
        dbg[3] = d_sel;
        dbg[4] = d_nu;
        dbg[5] = d_E0;
        dbg[6] = (int)p.E;
        dbg[7] = p.cell;
    }

    /* Write back. The neutron's final report card. */
    parts[tid] = p;
    rngs[tid]  = rng;
    } /* end if (tid < n_part) */
}
