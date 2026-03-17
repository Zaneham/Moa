/* tp_diag3.cu -- Transport loop step counter.
 * Uses EXACT struct layouts from tp_kern.cu.
 * Runs 1 transport step, writes diagnostics. */

#define GP_PI     3.14159265f
#define GP_EPS    1.0e-6f
#define GP_EMIN   1.0e-5f
#define GP_INF    1.0e30f
#define GP_MHIST  4096
#define GP_MAXN   8
#define GP_CHK(i, n) ((unsigned)(i) >= (unsigned)(n))

struct gp_rng_t { unsigned s0; unsigned s1; unsigned s2; unsigned s3; unsigned s4; unsigned d; };
struct gp_part_t { float x; float y; float z; float u; float v; float w; float E; float wt; int cell; int alive; };
struct gp_site_t { float x; float y; float z; float E; };
struct gp_surf_t { int type; float A; float B; float C; float D; float E; float F; float G; float H; float J; float K; };
struct gp_cell_t { int n_half; int mat; float rho; int sh_off; };
struct gp_mat_t { int n_nuc; int mn_off; int pad; };
struct gp_nhdr_t { int ZA; float awr; int n_pts; int lrf; float SPI; float AP; float EL; float EH; int n_res; int xs_off; int res_off; int pad; };
struct gp_res_t { float Er; float AJ; float Gn; float Gg; float Gfa; float Gfb; };

#define GP_RNXT(r, _out) do { unsigned _t = (r).s4; unsigned _s = (r).s0; (r).s4 = (r).s3; (r).s3 = (r).s2; (r).s2 = (r).s1; (r).s1 = _s; _t = _t ^ (_t >> 2); _t = _t ^ (_t << 1); _t = _t ^ (_s ^ (_s << 4)); (r).s0 = _t; (r).d = (r).d + 362437u; (_out) = _t + (r).d; } while(0)
#define GP_RDBL(r, _out) do { unsigned _rv; GP_RNXT(r, _rv); (_out) = (float)(_rv >> 8) * (1.0f / 16777216.0f); } while(0)
#define GP_EVAL(s, px, py, pz, _out) (_out) = (s).A*(px)*(px) + (s).B*(py)*(py) + (s).C*(pz)*(pz) + (s).D*(px)*(py) + (s).E*(py)*(pz) + (s).F*(px)*(pz) + (s).G*(px) + (s).H*(py) + (s).J*(pz) + (s).K

__global__ void gp_kern(
    gp_part_t *parts, gp_rng_t *rngs, gp_site_t *fbank, int *n_fbank,
    gp_surf_t *surfs, gp_cell_t *cells, int *csur, int *csen,
    gp_mat_t *mats, int *mnid, float *mfrc, gp_nhdr_t *nhdr,
    float *nE, float *nst, float *nsa, float *nsf, float *nse, float *nnu,
    gp_res_t *nres, int n_part, int n_surf, int n_cell, int n_mat,
    int n_nuc, float k_eff, int fb_max)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_part) {

    gp_part_t p;
    gp_rng_t  rng;
    p   = parts[tid];
    rng = rngs[tid];

    int steps_done = 0;
    int step = GP_MHIST;
    float last_sig_t = -1.0f;
    int kill_reason = 0;

    while (p.alive && step > 0) {
        step--;
        steps_done++;

        /* ---- Geometry: distance to nearest surface ---- */
        if (GP_CHK(p.cell, n_cell)) { kill_reason = 1; p.alive = 0; break; }
        gp_cell_t c = cells[p.cell];
        if (c.mat < 0 || c.mat >= n_mat) { kill_reason = 2; p.alive = 0; break; }

        float d_min = GP_INF;
        int next_sense = 0;
        int sh;
        for (sh = 0; sh < c.n_half && sh < 20; sh++) {
            int sidx = csur[c.sh_off + sh];
            int sense = csen[c.sh_off + sh];
            if (GP_CHK(sidx, n_surf)) continue;
            gp_surf_t su = surfs[sidx];

            float qa = su.A*p.u*p.u + su.B*p.v*p.v + su.C*p.w*p.w + su.D*p.u*p.v + su.E*p.u*p.w + su.F*p.v*p.w;
            float qb = 2.0f*(su.A*p.x*p.u + su.B*p.y*p.v + su.C*p.z*p.w) + su.D*(p.x*p.v + p.y*p.u) + su.E*(p.x*p.w + p.z*p.u) + su.F*(p.y*p.w + p.z*p.v) + su.G*p.u + su.H*p.v + su.J*p.w;
            float qc = su.A*p.x*p.x + su.B*p.y*p.y + su.C*p.z*p.z + su.D*p.x*p.y + su.E*p.x*p.z + su.F*p.y*p.z + su.G*p.x + su.H*p.y + su.J*p.z + su.K;

            float disc = qb*qb - 4.0f*qa*qc;
            if (disc < 0.0f) continue;
            float sq = sqrtf(disc);
            float denom = 2.0f * qa;
            if (denom > -GP_EPS && denom < GP_EPS) {
                if (qb > GP_EPS || qb < -GP_EPS) {
                    float t0 = -qc / qb;
                    if (t0 > GP_EPS && t0 < d_min) { d_min = t0; next_sense = sense; }
                }
                continue;
            }
            float t1 = (-qb - sq) / denom;
            float t2 = (-qb + sq) / denom;
            if (t1 > GP_EPS && t1 < d_min) { d_min = t1; next_sense = sense; }
            if (t2 > GP_EPS && t2 < d_min) { d_min = t2; next_sense = sense; }
        }

        /* ---- Cross section: sig_t only ---- */
        float sig_t = 0.0f;
        gp_mat_t m = mats[c.mat];
        int ni;
        for (ni = 0; ni < m.n_nuc && ni < GP_MAXN; ni++) {
            int nid = mnid[m.mn_off + ni];
            float frc = mfrc[m.mn_off + ni];
            if (GP_CHK(nid, n_nuc)) continue;
            gp_nhdr_t h = nhdr[nid];
            int lo = 0;
            int hi = h.n_pts - 1;
            int guard = 32;
            while (lo < hi - 1 && guard > 0) { guard--; int mid = (lo + hi) / 2; float eM = nE[h.xs_off + mid]; if (p.E < eM) hi = mid; else lo = mid; }
            float st = nst[h.xs_off + lo];
            sig_t += frc * st;
        }
        last_sig_t = sig_t;

        /* ---- Distance to collision ---- */
        float xi;
        GP_RDBL(rng, xi);
        float d_col = -logf(xi + 1.0e-30f) / (sig_t + 1.0e-30f);

        if (d_col > d_min) {
            /* Surface crossing */
            p.x += d_min * p.u; p.y += d_min * p.v; p.z += d_min * p.w;
            if (next_sense < 0) { p.alive = 0; kill_reason = 3; }
            else {
                /* Find which cell we entered -- simplified: check all cells */
                int found = 0;
                int ci;
                for (ci = 0; ci < n_cell && ci < 10; ci++) {
                    gp_cell_t cc = cells[ci];
                    int inside = 1;
                    int hh;
                    for (hh = 0; hh < cc.n_half && hh < 20; hh++) {
                        int sidx2 = csur[cc.sh_off + hh];
                        int sn2 = csen[cc.sh_off + hh];
                        if (GP_CHK(sidx2, n_surf)) { inside = 0; break; }
                        float eval;
                        GP_EVAL(surfs[sidx2], p.x, p.y, p.z, eval);
                        if (sn2 < 0 && eval > GP_EPS) { inside = 0; break; }
                        if (sn2 > 0 && eval < -GP_EPS) { inside = 0; break; }
                    }
                    if (inside && cc.mat >= 0) { p.cell = ci; found = 1; break; }
                }
                if (!found) { p.alive = 0; kill_reason = 4; }
            }
        } else {
            /* Collision -- just scatter isotropically */
            p.x += d_col * p.u; p.y += d_col * p.v; p.z += d_col * p.w;
            float imu; float iphi;
            GP_RDBL(rng, imu); imu = 2.0f * imu - 1.0f;
            GP_RDBL(rng, iphi); iphi = 2.0f * GP_PI * iphi;
            float ist = sqrtf(1.0f - imu * imu);
            p.u = ist * cosf(iphi); p.v = ist * sinf(iphi); p.w = imu;
            if (p.E < GP_EMIN) p.alive = 0;
        }
    }

    /* Write diagnostic unconditionally */
    int slot = atomicAdd(n_fbank, 1);
    if (slot < fb_max) {
        gp_site_t site;
        site.x = (float)steps_done;
        site.y = last_sig_t;
        site.z = p.E;
        site.E = (float)kill_reason;
        fbank[slot] = site;
    }

    parts[tid] = p;
    rngs[tid]  = rng;
    } /* end if (tid < n_part) */
}
