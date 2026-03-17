/* gp_host.h -- GPU transport host API
 *
 * Interface between Moa's CPU world and BarraCUDA's GPU world.
 * Three functions, one context struct, zero drama.
 *
 * Usage:
 *   gp_ctx_t ctx;
 *   gp_init(&ctx, prob, "gpu/tp_kern.hsaco");
 *   for each batch:
 *     gp_batch(&ctx, prob, src_bank, n_src, fis_bank, &n_fis);
 *   gp_free(&ctx);
 */

#ifndef GP_HOST_H
#define GP_HOST_H

#include "../moa.h"

/* Forward-declare bc_runtime types — avoid dragging in the
 * full BarraCUDA header unless we're actually compiling
 * the GPU host code. */

/* ---- Device memory handles ---- */

typedef struct {
    /* bc_runtime handles */
    void *dev;       /* bc_device_t*, heap-allocated    */
    void *kern;      /* bc_kernel_t*, heap-allocated    */
    int   loaded;    /* 1 = kernel loaded, 0 = not      */

    /* Device pointers — geometry (uploaded once) */
    void *d_surfs;
    void *d_cells;
    void *d_csur;
    void *d_csen;
    void *d_mats;
    void *d_mnid;
    void *d_mfrc;

    /* Device pointers — nuclear data (uploaded once) */
    void *d_nhdr;
    void *d_nE;
    void *d_nst;
    void *d_nsa;
    void *d_nsf;
    void *d_nse;
    void *d_nnu;
    void *d_nres;

    /* Device pointers — per-batch (resized per batch) */
    void *d_parts;
    void *d_rngs;
    void *d_fbank;
    void *d_nfb;     /* int* for atomicAdd counter      */
    void *d_snap;    /* SNAP diagnostic buffer (64 dwords) */

    /* Sizes for freeing */
    int   n_part_alloc;  /* particles allocated on device */
    int   snap;          /* 1 = SNAP mode active          */

    /* Problem dimensions */
    int   n_surf;
    int   n_cell;
    int   n_mat;
    int   n_nuc;
    int   n_csur;    /* total flattened halfspace entries */
    int   n_mnid;    /* total flattened material entries  */
    int   n_xs;      /* total flattened xs entries        */
    int   n_res;     /* total flattened resonance entries */
} gp_ctx_t;

/* ---- API ---- */

/* Initialise GPU context: load kernel, upload geometry + nuclide data.
 * Returns 0 on success, <0 on error. */
int  gp_init(gp_ctx_t *ctx, const mo_prob_t *prob,
             const char *hsaco_path);

/* Run one batch of particles on GPU.
 * Reads src_bank[0..n_src-1], writes fis_bank[0..*n_fis-1].
 * Source positions come from CPU bank, directions sampled on CPU,
 * cells found on CPU. Float conversion handled internally. */
int  gp_batch(gp_ctx_t *ctx, mo_prob_t *prob,
              const mo_site_t *src_bank, int n_src,
              mo_site_t *fis_bank, int *n_fis);

/* Free all device memory and shut down GPU context. */
void gp_free(gp_ctx_t *ctx);

#endif /* GP_HOST_H */
