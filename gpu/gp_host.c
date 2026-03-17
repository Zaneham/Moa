/* gp_host.c -- Host-side GPU launcher
 *
 * The boring but essential part: marshal CPU data to GPU format,
 * convert double to float, upload, dispatch, download, convert back.
 * Like being a translator at the United Nations, except one side
 * speaks double-precision and the other only understands float.
 *
 * Uses BarraCUDA's bc_runtime.h to talk to the AMD GPU via HSA.
 * Linux/ROCm only — this file won't compile on Windows because
 * dlopen doesn't exist there and HSA is an AMD-on-Linux affair.
 *
 * The kernel (tp_kern.cu → tp_kern.hsaco) is loaded at init
 * and dispatched once per batch. All geometry and nuclear data
 * is uploaded once; only particles and RNG state change per batch.
 */

#include "gp_host.h"
#include "C:/dev/compilers/barracuda/src/runtime/bc_runtime.h"
#include "C:/dev/compilers/barracuda/src/runtime/bc_abend.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ---- GPU struct mirrors (must match tp_kern.cu exactly) ---- */

typedef struct { float x,y,z,u,v,w,E,wt; int cell, alive; } h_part_t;
typedef struct { unsigned int s0,s1,s2,s3,s4,d;             } h_rng_t;
typedef struct { int type; float A,B,C,D,E,F,G,H,J,K;       } h_surf_t;
typedef struct { int n_half, mat; float rho; int sh_off;     } h_cell_t;
typedef struct { int n_nuc, mn_off, pad;                     } h_mat_t;
typedef struct { int ZA; float awr; int n_pts, lrf;
                 float SPI, AP, EL, EH;
                 int n_res, xs_off, res_off, pad;            } h_nhdr_t;
typedef struct { float Er, AJ, Gn, Gg, Gfa, Gfb;            } h_res_t;
typedef struct { float x, y, z, E;                           } h_site_t;

/* ---- SplitMix32 for seeding xorwow ---- */

static unsigned int
sm32(unsigned int *st)
{
    unsigned int z = (*st += 0x9E3779B9u);
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

/* ---- Helper: alloc + upload ---- */

static void *
gp_up(bc_device_t *dev, const void *src, size_t sz)
{
    void *d = bc_alloc(dev, sz);
    if (!d) return NULL;
    if (bc_copy_h2d(dev, d, src, sz) != BC_RT_OK) {
        bc_free(dev, d);
        return NULL;
    }
    return d;
}

/* ---- gp_init ---- */

int
gp_init(gp_ctx_t *ctx, const mo_prob_t *prob,
        const char *hsaco_path)
{
    bc_device_t *dev;
    bc_kernel_t *kern;
    int i, j, rc;

    memset(ctx, 0, sizeof(*ctx));

    /* Allocate bc handles on heap */
    dev  = (bc_device_t *)calloc(1, sizeof(bc_device_t));
    kern = (bc_kernel_t *)calloc(1, sizeof(bc_kernel_t));
    if (!dev || !kern) {
        fprintf(stderr, "gp_init: alloc fail\n");
        free(dev); free(kern);
        return -1;
    }

    rc = bc_device_init(dev);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "gp_init: device init failed (%d)\n", rc);
        free(dev); free(kern);
        return -1;
    }

    rc = bc_load_kernel(dev, hsaco_path, "gp_kern", kern);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "gp_init: kernel load failed (%d)\n", rc);
        bc_device_shutdown(dev);
        free(dev); free(kern);
        return -1;
    }

    ctx->dev    = dev;
    ctx->kern   = kern;
    ctx->loaded = 1;
    ctx->n_surf = prob->n_surf;
    ctx->n_cell = prob->n_cell;
    ctx->n_mat  = prob->n_mat;
    ctx->n_nuc  = prob->n_nucl;

    /* ---- Upload surfaces (double → float) ---- */
    {
        h_surf_t *hs = (h_surf_t *)calloc((size_t)prob->n_surf,
                                           sizeof(h_surf_t));
        if (!hs) goto fail;
        for (i = 0; i < prob->n_surf; i++) {
            const mo_surf_t *s = &prob->surf[i];
            hs[i].type = s->type;
            hs[i].A = (float)s->A; hs[i].B = (float)s->B;
            hs[i].C = (float)s->C; hs[i].D = (float)s->D;
            hs[i].E = (float)s->E; hs[i].F = (float)s->F;
            hs[i].G = (float)s->G; hs[i].H = (float)s->H;
            hs[i].J = (float)s->J; hs[i].K = (float)s->K;
        }
        ctx->d_surfs = gp_up(dev, hs,
                             (size_t)prob->n_surf * sizeof(h_surf_t));
        free(hs);
        if (!ctx->d_surfs) goto fail;
    }

    /* ---- Flatten and upload cells + csur/csen ---- */
    {
        /* Count total halfspaces */
        int tot = 0;
        for (i = 0; i < prob->n_cell; i++)
            tot += prob->cell[i].n_half;
        ctx->n_csur = tot;

        h_cell_t *hc = (h_cell_t *)calloc((size_t)prob->n_cell,
                                           sizeof(h_cell_t));
        int *hs_sur = (int *)calloc((size_t)(tot + 1), sizeof(int));
        int *hs_sen = (int *)calloc((size_t)(tot + 1), sizeof(int));
        if (!hc || !hs_sur || !hs_sen) {
            free(hc); free(hs_sur); free(hs_sen);
            goto fail;
        }

        int off = 0;
        for (i = 0; i < prob->n_cell; i++) {
            const mo_cell_t *c = &prob->cell[i];
            hc[i].n_half = c->n_half;
            hc[i].mat    = c->mat;
            hc[i].rho    = (float)c->rho;
            hc[i].sh_off = off;
            for (j = 0; j < c->n_half && j < MO_MAX_HALF; j++) {
                hs_sur[off + j] = c->surf[j];
                hs_sen[off + j] = c->sense[j];
            }
            off += c->n_half;
        }

        ctx->d_cells = gp_up(dev, hc,
                             (size_t)prob->n_cell * sizeof(h_cell_t));
        ctx->d_csur  = gp_up(dev, hs_sur,
                             (size_t)(tot + 1) * sizeof(int));
        ctx->d_csen  = gp_up(dev, hs_sen,
                             (size_t)(tot + 1) * sizeof(int));
        free(hc); free(hs_sur); free(hs_sen);
        if (!ctx->d_cells || !ctx->d_csur || !ctx->d_csen)
            goto fail;
    }

    /* ---- Flatten and upload materials + mnid/mfrc ---- */
    {
        int tot = 0;
        for (i = 0; i < prob->n_mat; i++)
            tot += prob->mat[i].n_nuc;
        ctx->n_mnid = tot;

        h_mat_t *hm  = (h_mat_t *)calloc((size_t)prob->n_mat,
                                          sizeof(h_mat_t));
        int   *mid   = (int *)calloc((size_t)(tot + 1), sizeof(int));
        float *mfc   = (float *)calloc((size_t)(tot + 1), sizeof(float));
        if (!hm || !mid || !mfc) {
            free(hm); free(mid); free(mfc);
            goto fail;
        }

        int off = 0;
        for (i = 0; i < prob->n_mat; i++) {
            const mo_mat_t *m = &prob->mat[i];
            hm[i].n_nuc  = m->n_nuc;
            hm[i].mn_off = off;
            hm[i].pad    = 0;
            for (j = 0; j < m->n_nuc && j < MO_MAX_MNUC; j++) {
                mid[off + j] = m->nuc_id[j];
                mfc[off + j] = (float)m->frac[j];
            }
            off += m->n_nuc;
        }

        ctx->d_mats = gp_up(dev, hm,
                            (size_t)prob->n_mat * sizeof(h_mat_t));
        ctx->d_mnid = gp_up(dev, mid,
                            (size_t)(tot + 1) * sizeof(int));
        ctx->d_mfrc = gp_up(dev, mfc,
                            (size_t)(tot + 1) * sizeof(float));
        free(hm); free(mid); free(mfc);
        if (!ctx->d_mats || !ctx->d_mnid || !ctx->d_mfrc)
            goto fail;
    }

    /* ---- Flatten and upload nuclide data ---- */
    {
        /* Count total xs points and resonances */
        int tot_xs = 0, tot_res = 0;
        for (i = 0; i < prob->n_nucl; i++) {
            tot_xs  += prob->nucl[i].n_pts;
            tot_res += prob->nucl[i].n_res;
        }
        ctx->n_xs  = tot_xs;
        ctx->n_res = tot_res;

        /* Headers */
        h_nhdr_t *hnh = (h_nhdr_t *)calloc((size_t)prob->n_nucl,
                                            sizeof(h_nhdr_t));
        if (!hnh) goto fail;

        /* Flat arrays */
        float *fE  = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        float *fst = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        float *fsa = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        float *fsf = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        float *fse = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        float *fnu = (float *)calloc((size_t)(tot_xs + 1), sizeof(float));
        h_res_t *fres = (h_res_t *)calloc((size_t)(tot_res + 1),
                                           sizeof(h_res_t));
        if (!fE || !fst || !fsa || !fsf || !fse || !fnu || !fres) {
            free(hnh); free(fE); free(fst); free(fsa);
            free(fsf); free(fse); free(fnu); free(fres);
            goto fail;
        }

        int xoff = 0, roff = 0;
        for (i = 0; i < prob->n_nucl; i++) {
            const mo_nucl_t *n = &prob->nucl[i];
            hnh[i].ZA      = n->ZA;
            hnh[i].awr     = (float)n->awr;
            hnh[i].n_pts   = n->n_pts;
            hnh[i].lrf     = n->lrf;
            hnh[i].SPI     = (float)n->SPI;
            hnh[i].AP      = (float)n->AP;
            hnh[i].EL      = (float)n->EL;
            hnh[i].EH      = (float)n->EH;
            hnh[i].n_res   = n->n_res;
            hnh[i].xs_off  = xoff;
            hnh[i].res_off = roff;
            hnh[i].pad     = 0;

            for (j = 0; j < n->n_pts; j++) {
                fE[xoff + j]  = (float)n->E[j];
                fst[xoff + j] = (float)n->xs_t[j];
                fsa[xoff + j] = (float)n->xs_a[j];
                fsf[xoff + j] = (float)n->xs_f[j];
                fse[xoff + j] = (float)n->xs_el[j];
                fnu[xoff + j] = (float)n->nu[j];
            }
            xoff += n->n_pts;

            for (j = 0; j < n->n_res; j++) {
                fres[roff + j].Er  = (float)n->res[j].Er;
                fres[roff + j].AJ  = (float)n->res[j].AJ;
                fres[roff + j].Gn  = (float)n->res[j].Gn;
                fres[roff + j].Gg  = (float)n->res[j].Gg;
                fres[roff + j].Gfa = (float)n->res[j].Gfa;
                fres[roff + j].Gfb = (float)n->res[j].Gfb;
            }
            roff += n->n_res;
        }

        ctx->d_nhdr = gp_up(dev, hnh,
                            (size_t)prob->n_nucl * sizeof(h_nhdr_t));
        ctx->d_nE   = gp_up(dev, fE,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nst  = gp_up(dev, fst,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nsa  = gp_up(dev, fsa,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nsf  = gp_up(dev, fsf,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nse  = gp_up(dev, fse,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nnu  = gp_up(dev, fnu,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nres = gp_up(dev, fres,
                            (size_t)(tot_res + 1) * sizeof(h_res_t));

        free(hnh); free(fE); free(fst); free(fsa);
        free(fsf); free(fse); free(fnu); free(fres);

        if (!ctx->d_nhdr || !ctx->d_nE || !ctx->d_nst ||
            !ctx->d_nsa  || !ctx->d_nsf || !ctx->d_nse ||
            !ctx->d_nnu  || !ctx->d_nres)
            goto fail;
    }

    /* Register allocations for ABEND fault correlation.
     * When the GPU faults, the dump cross-references the fault address
     * against these to tell you what you actually hit. */
    bc_trak(dev, ctx->d_surfs, (size_t)prob->n_surf * sizeof(h_surf_t),
            "surfs", AB_FL_RW);
    bc_trak(dev, ctx->d_cells, (size_t)prob->n_cell * sizeof(h_cell_t),
            "cells", AB_FL_RW);
    bc_trak(dev, ctx->d_csur,
            (size_t)(ctx->n_csur + 1) * sizeof(int), "csur", AB_FL_RW);
    bc_trak(dev, ctx->d_csen,
            (size_t)(ctx->n_csur + 1) * sizeof(int), "csen", AB_FL_RW);
    bc_trak(dev, ctx->d_mats, (size_t)prob->n_mat * sizeof(h_mat_t),
            "mats", AB_FL_RW);
    bc_trak(dev, ctx->d_mnid,
            (size_t)(ctx->n_mnid + 1) * sizeof(int), "mnid", AB_FL_RW);
    bc_trak(dev, ctx->d_mfrc,
            (size_t)(ctx->n_mnid + 1) * sizeof(float), "mfrc", AB_FL_RW);
    bc_trak(dev, ctx->d_nhdr,
            (size_t)prob->n_nucl * sizeof(h_nhdr_t), "nhdr", AB_FL_RW);
    bc_trak(dev, ctx->d_nE,  (size_t)ctx->n_xs * sizeof(float),
            "nE",  AB_FL_RW);
    bc_trak(dev, ctx->d_nst, (size_t)ctx->n_xs * sizeof(float),
            "nst", AB_FL_RW);
    bc_trak(dev, ctx->d_nsa, (size_t)ctx->n_xs * sizeof(float),
            "nsa", AB_FL_RW);
    bc_trak(dev, ctx->d_nsf, (size_t)ctx->n_xs * sizeof(float),
            "nsf", AB_FL_RW);
    bc_trak(dev, ctx->d_nse, (size_t)ctx->n_xs * sizeof(float),
            "nse", AB_FL_RW);
    bc_trak(dev, ctx->d_nnu, (size_t)ctx->n_xs * sizeof(float),
            "nnu", AB_FL_RW);

    printf("moa: GPU initialised — %d surfs, %d cells, %d nucs, "
           "%d xs pts, %d resonances\n",
           ctx->n_surf, ctx->n_cell, ctx->n_nuc,
           ctx->n_xs, ctx->n_res);
    return 0;

fail:
    fprintf(stderr, "gp_init: GPU data upload failed\n");
    gp_free(ctx);
    return -1;
}

/* ---- gp_batch ---- */

int
gp_batch(gp_ctx_t *ctx, mo_prob_t *prob,
         const mo_site_t *src_bank, int n_src,
         mo_site_t *fis_bank, int *n_fis)
{
    bc_device_t *dev = (bc_device_t *)ctx->dev;
    bc_kernel_t *kern = (bc_kernel_t *)ctx->kern;
    int i, rc;

    if (!ctx->loaded || n_src <= 0) return -1;

    int fb_max = n_src * 3;  /* generous fission bank capacity */
    if (fb_max > MO_MAX_PART) fb_max = MO_MAX_PART;

    /* ---- Build particle array on CPU ---- */
    h_part_t *hp = (h_part_t *)calloc((size_t)n_src, sizeof(h_part_t));
    h_rng_t  *hr = (h_rng_t *)calloc((size_t)n_src, sizeof(h_rng_t));
    if (!hp || !hr) { free(hp); free(hr); return -1; }

    for (i = 0; i < n_src; i++) {
        hp[i].x    = (float)src_bank[i].x;
        hp[i].y    = (float)src_bank[i].y;
        hp[i].z    = (float)src_bank[i].z;
        hp[i].E    = (float)src_bank[i].E;
        hp[i].wt   = (float)src_bank[i].wt;
        hp[i].alive = 1;

        /* Find cell on CPU (double precision geometry) */
        hp[i].cell = cg_find(prob, src_bank[i].x,
                             src_bank[i].y, src_bank[i].z);
        if (hp[i].cell < 0 ||
            prob->cell[hp[i].cell].mat < 0) {
            hp[i].alive = 0;
            hp[i].cell  = 0;
        }

        /* Sample isotropic direction on CPU */
        {
            double mu  = 2.0 * rn_dbl(&prob->rng) - 1.0;
            double phi = 2.0 * MO_PI * rn_dbl(&prob->rng);
            double st  = sqrt(1.0 - mu * mu);
            hp[i].u = (float)(st * cos(phi));
            hp[i].v = (float)(st * sin(phi));
            hp[i].w = (float)mu;
        }

        /* Seed xorwow RNG via SplitMix32 */
        {
            unsigned int seed = 42u ^ (unsigned int)i;
            hr[i].s0 = sm32(&seed);
            hr[i].s1 = sm32(&seed);
            hr[i].s2 = sm32(&seed);
            hr[i].s3 = sm32(&seed);
            hr[i].s4 = sm32(&seed);
            hr[i].d  = 0;
        }
    }

    /* ---- Allocate per-batch device memory ---- */
    size_t sz_parts = (size_t)n_src * sizeof(h_part_t);
    size_t sz_rngs  = (size_t)n_src * sizeof(h_rng_t);
    size_t sz_fbank = (size_t)fb_max * sizeof(h_site_t);
    size_t sz_nfb   = sizeof(int);

    /* Free previous per-batch allocs if size changed */
    if (ctx->d_parts && ctx->n_part_alloc != n_src) {
        bc_free(dev, ctx->d_parts);
        bc_free(dev, ctx->d_rngs);
        bc_free(dev, ctx->d_fbank);
        bc_free(dev, ctx->d_nfb);
        ctx->d_parts = NULL;
    }

    if (!ctx->d_parts) {
        ctx->d_parts = bc_alloc(dev, sz_parts);
        ctx->d_rngs  = bc_alloc(dev, sz_rngs);
        ctx->d_fbank = bc_alloc(dev, sz_fbank);
        ctx->d_nfb   = bc_alloc(dev, sz_nfb);
        ctx->n_part_alloc = n_src;
        if (!ctx->d_parts || !ctx->d_rngs ||
            !ctx->d_fbank || !ctx->d_nfb) {
            free(hp); free(hr);
            return -1;
        }
        bc_trak(dev, ctx->d_parts, sz_parts, "parts", AB_FL_RW);
        bc_trak(dev, ctx->d_rngs,  sz_rngs,  "rngs",  AB_FL_RW);
        bc_trak(dev, ctx->d_fbank, sz_fbank, "fbank", AB_FL_RW);
        bc_trak(dev, ctx->d_nfb,   sz_nfb,  "n_fbank", AB_FL_RW);
    }

    /* ---- Upload ---- */
    rc = bc_copy_h2d(dev, ctx->d_parts, hp, sz_parts);
    if (rc != BC_RT_OK) { free(hp); free(hr); return -1; }

    rc = bc_copy_h2d(dev, ctx->d_rngs, hr, sz_rngs);
    if (rc != BC_RT_OK) { free(hp); free(hr); return -1; }

    /* Zero fission bank counter */
    int zero = 0;
    rc = bc_copy_h2d(dev, ctx->d_nfb, &zero, sizeof(int));
    if (rc != BC_RT_OK) { free(hp); free(hr); return -1; }

    free(hp);
    free(hr);

    /* ---- SNAP buffer (allocated once, reused per batch) ---- */
    if (ctx->snap && !ctx->d_snap) {
        ctx->d_snap = bc_alloc(dev, 256);
        if (!ctx->d_snap) {
            fprintf(stderr, "gp_batch: snap alloc failed\n");
            ctx->snap = 0;
        }
    }

    /* ---- Build kernel arguments ---- */
    /* BarraCUDA kernarg layout: every param at 8-byte stride (param_idx*8).
     * After user params: hidden args (block_count_x/y/z, group_size_x/y/z).
     * Then snap pointer if snap mode.
     * Total = 26 user params × 8 + 24 hidden [+ 8 snap] bytes. */
    float k_eff_f = (float)prob->k_eff;

    unsigned char args[280];
    memset(args, 0, sizeof(args));
    int aoff = 0;

    /* All params packed at 8-byte stride — BarraCUDA reads param N
     * at offset N*8 via s_load_dword from kernarg base. */
    #define PACK_PTR(p) do { \
        void *_v = (p); \
        memcpy(args + aoff, &_v, 8); aoff += 8; \
    } while(0)
    #define PACK_I32(v) do { \
        int _i = (v); \
        memcpy(args + aoff, &_i, 4); aoff += 8; \
    } while(0)
    #define PACK_F32(v) do { \
        float _f = (v); \
        memcpy(args + aoff, &_f, 4); aoff += 8; \
    } while(0)

    PACK_PTR(ctx->d_parts);     /*  0: parts       */
    PACK_PTR(ctx->d_rngs);      /*  8: rngs        */
    PACK_PTR(ctx->d_fbank);     /* 16: fbank       */
    PACK_PTR(ctx->d_nfb);       /* 24: n_fbank     */
    PACK_PTR(ctx->d_surfs);     /* 32: surfs       */
    PACK_PTR(ctx->d_cells);     /* 40: cells       */
    PACK_PTR(ctx->d_csur);      /* 48: csur        */
    PACK_PTR(ctx->d_csen);      /* 56: csen        */
    PACK_PTR(ctx->d_mats);      /* 64: mats        */
    PACK_PTR(ctx->d_mnid);      /* 72: mnid        */
    PACK_PTR(ctx->d_mfrc);      /* 80: mfrc        */
    PACK_PTR(ctx->d_nhdr);      /* 88: nhdr        */
    PACK_PTR(ctx->d_nE);        /* 96: nE          */
    PACK_PTR(ctx->d_nst);       /*104: nst         */
    PACK_PTR(ctx->d_nsa);       /*112: nsa         */
    PACK_PTR(ctx->d_nsf);       /*120: nsf         */
    PACK_PTR(ctx->d_nse);       /*128: nse         */
    PACK_PTR(ctx->d_nnu);       /*136: nnu         */
    PACK_PTR(ctx->d_nres);      /*144: nres        */
    PACK_I32(n_src);             /*152: n_part      */
    PACK_I32(ctx->n_surf);      /*160: n_surf      */
    PACK_I32(ctx->n_cell);      /*168: n_cell      */
    PACK_I32(ctx->n_mat);       /*176: n_mat       */
    PACK_I32(ctx->n_nuc);       /*184: n_nuc       */
    PACK_F32(k_eff_f);          /*192: k_eff       */
    PACK_I32(fb_max);           /*200: fb_max      */
    /* aoff = 208 = 26 params × 8 */

    #undef PACK_PTR
    #undef PACK_I32
    #undef PACK_F32

    /* ---- Hidden dispatch args (BarraCUDA layout) ----
     * At offset 208: block_count_x/y/z (3 × u32) + group_size_x/y/z (3 × u16)
     * Total 18 bytes, padded to 24. bc_dispatch doesn't fill these,
     * so we do it here. blockDim/gridDim reads come from here. */
    int block = 256;
    int grid  = (n_src + block - 1) / block;

    {
        uint32_t bc[3] = { (uint32_t)grid, 1, 1 };
        uint16_t gs[3] = { (uint16_t)block, 1, 1 };
        memcpy(args + aoff,      bc, 12);   /* block_count x/y/z */
        memcpy(args + aoff + 12, gs,  6);   /* group_size  x/y/z */
        aoff += 24;
    }

    /* SNAP buffer pointer — packed after dispatch args.
     * Matches snap_koff in the instrumented kernel. If the kernel
     * wasn't compiled with --snap, the runtime clips to kernarg_size
     * and these bytes quietly vanish. No harm done. */
    if (ctx->snap && ctx->d_snap) {
        /* Zero the snap buffer before dispatch */
        unsigned char zb[256];
        memset(zb, 0, 256);
        bc_copy_h2d(dev, ctx->d_snap, zb, 256);

        void *sp = ctx->d_snap;
        memcpy(args + aoff, &sp, 8);
        aoff += 8;
    }

    /* ---- Dispatch ---- */
    rc = bc_dispatch(dev, kern,
                     (unsigned int)grid, 1, 1,
                     (unsigned int)block, 1, 1,
                     args, (unsigned int)aoff);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "gp_batch: dispatch failed (%d)\n", rc);
        return -1;
    }

    /* ---- SNAP readback ---- */
    if (ctx->snap && ctx->d_snap) {
        uint32_t snap_buf[64];
        memset(snap_buf, 0, sizeof(snap_buf));
        rc = bc_copy_d2h(dev, snap_buf, ctx->d_snap, 26 * 4);
        if (rc == BC_RT_OK) {
            printf("==== GPU SNAP DUMP ====\n");
            /* Param names for Moa's gp_kern — 26 user params */
            static const char *pn[] = {
                "parts", "rngs",  "fbank", "n_fbank",
                "surfs", "cells", "csur",  "csen",
                "mats",  "mnid",  "mfrc",  "nhdr",
                "nE",    "nst",   "nsa",   "nsf",
                "nse",   "nnu",   "nres",  "n_part",
                "n_surf","n_cell","n_mat", "n_nuc",
                "k_eff", "fb_max"
            };
            for (int si = 0; si < 26; si++) {
                float fval;
                memcpy(&fval, &snap_buf[si], 4);
                printf("  [%2d] %-8s = 0x%08X", si, pn[si], snap_buf[si]);
                /* Float interpretation for scalar params */
                if (si >= 19)
                    printf("  (%12.6g)", (double)fval);
                printf("\n");
            }
            printf("=======================\n");
        } else {
            fprintf(stderr, "gp_batch: snap readback failed\n");
        }
    }

    /* ---- Download fission bank ---- */
    int n_fb = 0;
    rc = bc_copy_d2h(dev, &n_fb, ctx->d_nfb, sizeof(int));
    if (rc != BC_RT_OK) return -1;

    if (n_fb > fb_max) n_fb = fb_max;

    if (n_fb > 0) {
        h_site_t *hfb = (h_site_t *)calloc((size_t)n_fb,
                                            sizeof(h_site_t));
        if (!hfb) return -1;

        rc = bc_copy_d2h(dev, hfb, ctx->d_fbank,
                         (size_t)n_fb * sizeof(h_site_t));
        if (rc != BC_RT_OK) { free(hfb); return -1; }

        /* SNAP: dump first fbank entry for diagnostics */
        if (ctx->snap && n_fb > 0) {
            uint32_t rx, ry, rz, rE;
            memcpy(&rx, &hfb[0].x, 4);
            memcpy(&ry, &hfb[0].y, 4);
            memcpy(&rz, &hfb[0].z, 4);
            memcpy(&rE, &hfb[0].E, 4);
            printf("  fbank[0]: x=%.6g (0x%08X) y=%.6g (0x%08X) "
                   "z=%.6g (0x%08X) E=%.6g (0x%08X)\n",
                   (double)hfb[0].x, rx, (double)hfb[0].y, ry,
                   (double)hfb[0].z, rz, (double)hfb[0].E, rE);
        }

        /* Convert float → double for CPU bank */
        int ncopy = n_fb;
        if (ncopy > MO_MAX_PART) ncopy = MO_MAX_PART;

        for (i = 0; i < ncopy; i++) {
            fis_bank[i].x  = (double)hfb[i].x;
            fis_bank[i].y  = (double)hfb[i].y;
            fis_bank[i].z  = (double)hfb[i].z;
            fis_bank[i].E  = (double)hfb[i].E;
            fis_bank[i].wt = 1.0;
        }
        free(hfb);

        *n_fis = ncopy;
    } else {
        *n_fis = 0;
    }

    return 0;
}

/* ---- gp_free ---- */

void
gp_free(gp_ctx_t *ctx)
{
    bc_device_t *dev = (bc_device_t *)ctx->dev;
    bc_kernel_t *kern = (bc_kernel_t *)ctx->kern;

    if (!dev) return;

    /* Free device memory */
    if (ctx->d_surfs)  bc_free(dev, ctx->d_surfs);
    if (ctx->d_cells)  bc_free(dev, ctx->d_cells);
    if (ctx->d_csur)   bc_free(dev, ctx->d_csur);
    if (ctx->d_csen)   bc_free(dev, ctx->d_csen);
    if (ctx->d_mats)   bc_free(dev, ctx->d_mats);
    if (ctx->d_mnid)   bc_free(dev, ctx->d_mnid);
    if (ctx->d_mfrc)   bc_free(dev, ctx->d_mfrc);
    if (ctx->d_nhdr)   bc_free(dev, ctx->d_nhdr);
    if (ctx->d_nE)     bc_free(dev, ctx->d_nE);
    if (ctx->d_nst)    bc_free(dev, ctx->d_nst);
    if (ctx->d_nsa)    bc_free(dev, ctx->d_nsa);
    if (ctx->d_nsf)    bc_free(dev, ctx->d_nsf);
    if (ctx->d_nse)    bc_free(dev, ctx->d_nse);
    if (ctx->d_nnu)    bc_free(dev, ctx->d_nnu);
    if (ctx->d_nres)   bc_free(dev, ctx->d_nres);
    if (ctx->d_parts)  bc_free(dev, ctx->d_parts);
    if (ctx->d_rngs)   bc_free(dev, ctx->d_rngs);
    if (ctx->d_fbank)  bc_free(dev, ctx->d_fbank);
    if (ctx->d_nfb)    bc_free(dev, ctx->d_nfb);
    if (ctx->d_snap)   bc_free(dev, ctx->d_snap);

    /* Unload kernel and shut down */
    if (ctx->loaded && kern)
        bc_unload_kernel(dev, kern);
    bc_device_shutdown(dev);

    free(kern);
    free(dev);

    memset(ctx, 0, sizeof(*ctx));

    printf("moa: GPU shutdown complete\n");
}
