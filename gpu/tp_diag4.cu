/* tp_diag4.cu -- Progressive crash bisection.
 * Load particle, read one cell, write diagnostic.
 * If n_fb=10000 with this, the cell read works.
 * Then we add more until it crashes. */

struct gp_site_t { float x; float y; float z; float E; };
struct gp_part_t { float x; float y; float z; float u; float v; float w; float E; float wt; int cell; int alive; };
struct gp_rng_t { unsigned s0; unsigned s1; unsigned s2; unsigned s3; unsigned s4; unsigned d; };
struct gp_surf_t { int type; float A; float B; float C; float D; float E; float F; float G; float H; float J; float K; };
struct gp_cell_t { int mat; int n_surf; int surf_start; int sense_start; float temp; };
struct gp_mat_t { int n_nuc; int nuc_start; int frc_start; };
struct gp_nhdr_t { int n_pts; int off; int n_res; int res_off; float awr; };
struct gp_res_t { float Er; float Gg; float Gn; float Gf; int l; int pad; };

#define GP_CHK(i, n) ((unsigned)(i) >= (unsigned)(n))

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
        p = parts[tid];

        float diag1 = (float)p.cell;
        float diag2 = (float)p.alive;
        float diag3 = 0.0f;
        float diag4 = 0.0f;

        /* Test 1: read cell struct */
        if (p.alive && !GP_CHK(p.cell, n_cell)) {
            gp_cell_t c = cells[p.cell];
            diag3 = (float)c.mat;
            diag4 = (float)c.n_surf;
        }

        int slot = atomicAdd(n_fbank, 1);
        if (slot < fb_max) {
            gp_site_t site;
            site.x = diag1;
            site.y = diag2;
            site.z = diag3;
            site.E = diag4;
            fbank[slot] = site;
        }
    }
}
