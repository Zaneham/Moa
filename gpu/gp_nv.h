/* gp_nv.h -- NVIDIA GPU transport host API
 *
 * Same interface as gp_host.h, different engine. If gp_host.h
 * is a Morris Minor with HSA under the bonnet, this is a Toyota
 * Hilux with CUDA. Both get you there, one just works everywhere
 * without needing a specific brand of petrol station.
 *
 * Usage:
 *   gp_ctx_t ctx;
 *   gp_init(&ctx, prob, "gpu/tp_kern.ptx");
 *   gp_batch(&ctx, prob, src_bank, n_src, fis_bank, &n_fis);
 *   gp_free(&ctx);
 */

#ifndef GP_NV_H
#define GP_NV_H

#include "../moa.h"
#include "C:/dev/compilers/barracuda/src/nvidia/nv_rt.h"

/* ---- ABEND Allocation Tracking ----
 * Mainframe-style fault diagnostics, now on NVIDIA.
 * IBM solved crash diagnostics in the 1960s. We're catching up
 * sixty years late, on a GPU, targeting a vendor that won't
 * tell us what the silicon actually does. Progress. */

#define GP_MAX_ALLOC  32
#define GP_MAX_ALBL   24

typedef struct {
    CUdevptr addr;
    size_t   size;
    char     name[GP_MAX_ALBL];
} gp_aent_t;

/* ---- Device memory handles ---- */

typedef struct {
    nv_dev_t   dev;
    nv_kern_t  kern;
    int        loaded;

    /* Device pointers — geometry (uploaded once) */
    CUdevptr d_surfs;
    CUdevptr d_cells;
    CUdevptr d_csur;
    CUdevptr d_csen;
    CUdevptr d_mats;
    CUdevptr d_mnid;
    CUdevptr d_mfrc;

    /* Device pointers — nuclear data (uploaded once) */
    CUdevptr d_nhdr;
    CUdevptr d_nE;
    CUdevptr d_nst;
    CUdevptr d_nsa;
    CUdevptr d_nsf;
    CUdevptr d_nse;
    CUdevptr d_nnu;
    CUdevptr d_nres;
    CUdevptr d_muE;
    CUdevptr d_muV;
    CUdevptr d_levQ;

    /* Device pointers — per-batch */
    CUdevptr d_parts;
    CUdevptr d_rngs;
    CUdevptr d_fbank;
    CUdevptr d_nfb;

    /* ABEND debug buffer — breadcrumb per thread.
     * Uses mapped host memory so we can read after a crash. */
    CUdevptr d_dbg;
    int     *h_dbg;       /* host-side mirror (mapped) */

    /* ABEND allocation map */
    gp_aent_t amap[GP_MAX_ALLOC];
    int       n_amap;

    /* Block-hit instrumentation (diagnostic) */
    CUdevptr d_bkhit;
    int   n_bkhit;       /* number of blocks */

    /* Sizes for freeing */
    int   n_part_alloc;
    int   snap;          /* unused on NV — compat with main.c */

    /* Problem dimensions */
    int   n_surf;
    int   n_cell;
    int   n_mat;
    int   n_nuc;
    int   n_csur;
    int   n_mnid;
    int   n_xs;
    int   n_res;
} gp_ctx_t;

/* ---- API (same signatures as gp_host.h) ---- */

int  gp_init(gp_ctx_t *ctx, const mo_prob_t *prob,
             const char *ptx_path);

int  gp_batch(gp_ctx_t *ctx, mo_prob_t *prob,
              const mo_site_t *src_bank, int n_src,
              mo_site_t *fis_bank, int *n_fis);

void gp_free(gp_ctx_t *ctx);

#endif /* GP_NV_H */
