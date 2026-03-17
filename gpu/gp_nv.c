/* gp_nv.c -- NVIDIA host-side GPU launcher
 *
 * The same boring-but-essential marshalling as gp_host.c, but
 * talking to NVIDIA via the CUDA Driver API instead of AMD via HSA.
 * Double → float, upload, dispatch, download, float → double.
 *
 * Uses BarraCUDA's nv_rt.h for GPU dispatch. Loads nvcuda.dll
 * at runtime — no CUDA SDK needed. Works on Windows and Linux,
 * because NVIDIA actually ships their driver everywhere. Novel.
 *
 * The kernel (tp_kern.cu → tp_kern.ptx) is JIT compiled by the
 * driver's PTX assembler. We write text, they do the rest.
 * It's like posting a letter and trusting the postal service.
 * Which, admittedly, requires a certain optimism about Royal Mail.
 */

#include "gp_nv.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>

/* ---- GPU struct mirrors (must match tp_kern.cu exactly) ---- */

typedef struct { float x,y,z,u,v,w,E,wt; int cell, alive; } h_part_t;
typedef struct { unsigned int s0,s1,s2,s3,s4,d;             } h_rng_t;
typedef struct { int type; float A,B,C,D,E,F,G,H,J,K;       } h_surf_t;
typedef struct { int n_half, mat; float rho; int sh_off;     } h_cell_t;
typedef struct { int n_nuc, mn_off, pad;                     } h_mat_t;
typedef struct { int ZA; float awr; int n_pts, lrf;
                 float SPI, AP, EL, EH;
                 int n_res, xs_off, res_off;
                 int n_mu, mu_off;
                 int n_lev, lev_off, pad;                     } h_nhdr_t;
typedef struct { float Er, AJ, Gn, Gg, Gfa, Gfb;            } h_res_t;
typedef struct { float x, y, z, E;                           } h_site_t;

/* ---- ABEND: Allocation Tracking ---- */

static void
gp_trak(gp_ctx_t *ctx, CUdevptr addr, size_t sz, const char *name)
{
    if (ctx->n_amap >= GP_MAX_ALLOC) return;
    gp_aent_t *e = &ctx->amap[ctx->n_amap++];
    e->addr = addr;
    e->size = sz;
    snprintf(e->name, sizeof(e->name), "%s", name);
}

/* ---- ABEND: Breadcrumb Labels ---- */

static const char *
gp_bstr(int chk)
{
    switch (chk) {
    case 0:  return "not started";
    case 1:  return "particle loaded";
    case 2:  return "loop top";
    case 3:  return "macro XS complete";
    case 4:  return "collision distance";
    case 5:  return "surface tracking done";
    case 6:  return "surface crossing";
    case 7:  return "collision processing";
    case 8:  return "scatter complete";
    case 9:  return "loop exit / writeback";
    case 21: return "cells[p.cell] access";
    case 22: return "mats[mi] access";
    case 23: return "nhdr[nid] access";
    case 24: return "XS grid arrays";
    case 25: return "XS lookup done";
    case 26: return "mats read ok";
    case 27: return "mnid[moff+ni]";
    default: return "unknown";
    }
}

/* ---- ABEND: Dump ----
 * When the kernel faults, produce a structured dump that would
 * make an IBM systems programmer shed a single tear of pride.
 * CUDA doesn't give us a fault address like HSA does, but we
 * can still print the allocation map and breadcrumb state. */

static void
gp_abnd(gp_ctx_t *ctx, int n_part, const char *reason)
{
    FILE *out = stderr;
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char ts[32];
    if (tm)
        snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d:%02d",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
    else
        snprintf(ts, sizeof(ts), "(unknown)");

    fprintf(out,
        " ============================================================\n"
        "  ABEND G0C5    %s\n"
        "  KERNEL gp_kern    GPU %s (sm_%d%d)    %s\n"
        " ============================================================\n\n",
        reason, ctx->dev.dev_name,
        ctx->dev.sm_major, ctx->dev.sm_minor, ts);

    /* ---- Memory Map ---- */
    if (ctx->n_amap > 0) {
        fprintf(out, " MEMORY MAP:\n");
        fprintf(out, "   %-18s %-10s %s\n", "BASE", "SIZE", "LABEL");
        int i;
        for (i = 0; i < ctx->n_amap; i++) {
            gp_aent_t *a = &ctx->amap[i];
            fprintf(out, "   %018" PRIX64 " %08" PRIX64 "   %s\n",
                    (uint64_t)a->addr, (uint64_t)a->size, a->name);
        }
        fprintf(out, "\n");
    }

    /* ---- Breadcrumb Recovery ----
     * If using mapped host memory, we can read the breadcrumbs directly
     * from host RAM — the GPU wrote there via PCIe, and the data survives
     * context corruption. If using device memory, try a D2H copy (which
     * will probably fail after CUDA_ERROR_ILLEGAL_ADDRESS, but hey). */
    if (n_part > 0 && (ctx->h_dbg || ctx->d_dbg)) {
        int cap = n_part;
        if (cap > 4096) cap = 4096;
        int *hdbg = NULL;
        int need_free = 0;

        if (ctx->h_dbg) {
            /* Mapped memory — just read it. No CUDA calls needed.
             * Like having the flight recorder built into the control tower
             * instead of the plane. Smart, that. */
            hdbg = ctx->h_dbg;
        } else {
            hdbg = (int *)calloc((size_t)cap, sizeof(int));
            if (hdbg) {
                int rc = nv_rt_d2h(&ctx->dev, hdbg, ctx->d_dbg,
                                   (size_t)cap * sizeof(int));
                if (rc != NV_RT_OK) {
                    fprintf(out, " BREADCRUMBS: readback FAILED "
                            "(context poisoned after fault)\n\n");
                    free(hdbg);
                    hdbg = NULL;
                }
                need_free = 1;
            }
        }

        if (hdbg) {
            /* Histogram breadcrumb values */
            int hist[32];
            int lo_chk = 99, hi_chk = 0;
            int lo_tid = 0,  hi_tid = 0;
            memset(hist, 0, sizeof(hist));
            int i;
            for (i = 0; i < cap; i++) {
                int v = hdbg[i];
                if (v >= 0 && v < 32) hist[v]++;
                if (v < lo_chk) { lo_chk = v; lo_tid = i; }
                if (v > hi_chk) { hi_chk = v; hi_tid = i; }
            }

            fprintf(out, " BREADCRUMBS (%d threads, %s):\n", cap,
                    ctx->h_dbg ? "mapped host memory" : "device readback");
            fprintf(out, "   Lowest:  thread %d -> checkpoint %d (%s)\n",
                    lo_tid, lo_chk, gp_bstr(lo_chk));
            fprintf(out, "   Highest: thread %d -> checkpoint %d (%s)\n",
                    hi_tid, hi_chk, gp_bstr(hi_chk));
            fprintf(out, "   Distribution:\n");
            int j;
            for (j = 0; j < 32; j++) {
                if (hist[j] > 0)
                    fprintf(out, "     chk %2d (%s): %d threads\n",
                            j, gp_bstr(j), hist[j]);
            }

            /* Show first few threads with the lowest checkpoint */
            fprintf(out, "   First threads at lowest checkpoint:\n");
            int shown = 0;
            for (i = 0; i < cap && shown < 8; i++) {
                if (hdbg[i] == lo_chk) {
                    fprintf(out, "     tid %d -> %d\n", i, hdbg[i]);
                    shown++;
                }
            }
            fprintf(out, "\n");

            if (need_free) free(hdbg);
        }
    }

    fprintf(out,
        " ============================================================\n"
        "  END OF DUMP    gp_kern\n"
        " ============================================================\n");
}

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

static CUdevptr
gp_up(nv_dev_t *dev, const void *src, size_t sz)
{
    CUdevptr d = nv_rt_alloc(dev, sz);
    if (!d) return 0;
    if (nv_rt_h2d(dev, d, src, sz) != NV_RT_OK) {
        nv_rt_free(dev, d);
        return 0;
    }
    return d;
}

/* ---- gp_init ---- */

int
gp_init(gp_ctx_t *ctx, const mo_prob_t *prob,
        const char *ptx_path)
{
    int i, j, rc;

    memset(ctx, 0, sizeof(*ctx));

    /* Init CUDA */
    fprintf(stderr, "gp_init: CUDA init...\n");
    fflush(stderr);
    rc = nv_rt_init(&ctx->dev);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gp_init: CUDA init failed (%d)\n", rc);
        return -1;
    }
    fprintf(stderr, "gp_init: CUDA ok, loading PTX...\n");
    fflush(stderr);

    /* Load PTX kernel — driver JIT compiles PTX to SASS.
     * For 5000-instruction kernels this can take a few seconds.
     * Patience is a virtue we don't have. */
    rc = nv_rt_load(&ctx->dev, ptx_path, "gp_kern", &ctx->kern);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gp_init: kernel load failed (%d)\n", rc);
        nv_rt_shut(&ctx->dev);
        return -1;
    }
    fprintf(stderr, "gp_init: kernel loaded, uploading data...\n");
    fflush(stderr);

    ctx->loaded = 1;
    ctx->n_surf = prob->n_surf;
    ctx->n_cell = prob->n_cell;
    ctx->n_mat  = prob->n_mat;
    ctx->n_nuc  = prob->n_nucl;

    /* ---- Upload surfaces (double -> float) ---- */
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
        ctx->d_surfs = gp_up(&ctx->dev, hs,
                             (size_t)prob->n_surf * sizeof(h_surf_t));
        free(hs);
        if (!ctx->d_surfs) goto fail;
        gp_trak(ctx, ctx->d_surfs,
                (size_t)prob->n_surf * sizeof(h_surf_t), "surfs");
    }

    /* ---- Flatten and upload cells + csur/csen ---- */
    {
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

        /* DIAG: dump cell layout for GPU */
        for (i = 0; i < prob->n_cell; i++)
            fprintf(stderr, "  cell[%d]: nh=%d mat=%d rho=%.4f off=%d\n",
                    i, hc[i].n_half, hc[i].mat, (double)hc[i].rho, hc[i].sh_off);
        for (i = 0; i < tot; i++)
            fprintf(stderr, "  csur[%d]=%d csen[%d]=%d\n",
                    i, hs_sur[i], i, hs_sen[i]);

        ctx->d_cells = gp_up(&ctx->dev, hc,
                             (size_t)prob->n_cell * sizeof(h_cell_t));
        ctx->d_csur  = gp_up(&ctx->dev, hs_sur,
                             (size_t)(tot + 1) * sizeof(int));
        ctx->d_csen  = gp_up(&ctx->dev, hs_sen,
                             (size_t)(tot + 1) * sizeof(int));
        free(hc); free(hs_sur); free(hs_sen);
        if (!ctx->d_cells || !ctx->d_csur || !ctx->d_csen)
            goto fail;
        gp_trak(ctx, ctx->d_cells,
                (size_t)prob->n_cell * sizeof(h_cell_t), "cells");
        gp_trak(ctx, ctx->d_csur,
                (size_t)(tot + 1) * sizeof(int), "csur");
        gp_trak(ctx, ctx->d_csen,
                (size_t)(tot + 1) * sizeof(int), "csen");
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

        ctx->d_mats = gp_up(&ctx->dev, hm,
                            (size_t)prob->n_mat * sizeof(h_mat_t));
        ctx->d_mnid = gp_up(&ctx->dev, mid,
                            (size_t)(tot + 1) * sizeof(int));
        ctx->d_mfrc = gp_up(&ctx->dev, mfc,
                            (size_t)(tot + 1) * sizeof(float));
        free(hm); free(mid); free(mfc);
        if (!ctx->d_mats || !ctx->d_mnid || !ctx->d_mfrc)
            goto fail;
        gp_trak(ctx, ctx->d_mats,
                (size_t)prob->n_mat * sizeof(h_mat_t), "mats");
        gp_trak(ctx, ctx->d_mnid,
                (size_t)(tot + 1) * sizeof(int), "mnid");
        gp_trak(ctx, ctx->d_mfrc,
                (size_t)(tot + 1) * sizeof(float), "mfrc");
    }

    /* ---- Flatten and upload nuclide data ---- */
    int tot_mu = 0, tot_lev = 0;
    {
        int tot_xs = 0, tot_res = 0;
        for (i = 0; i < prob->n_nucl; i++) {
            tot_xs  += prob->nucl[i].n_pts;
            tot_res += prob->nucl[i].n_res;
            tot_mu  += prob->nucl[i].n_mu;
            tot_lev += prob->nucl[i].n_lev;
        }
        ctx->n_xs  = tot_xs;
        ctx->n_res = tot_res;

        h_nhdr_t *hnh = (h_nhdr_t *)calloc((size_t)prob->n_nucl,
                                            sizeof(h_nhdr_t));
        if (!hnh) goto fail;

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

        /* mu-bar flattened arrays (energy grid + values) */
        float *fmE = NULL, *fmV = NULL;
        if (tot_mu > 0) {
            fmE = (float *)calloc((size_t)(tot_mu + 1), sizeof(float));
            fmV = (float *)calloc((size_t)(tot_mu + 1), sizeof(float));
            if (!fmE || !fmV) {
                free(hnh); free(fE); free(fst); free(fsa);
                free(fsf); free(fse); free(fnu); free(fres);
                free(fmE); free(fmV);
                goto fail;
            }
        }

        /* Discrete inelastic level Q values */
        float *flQ = NULL;
        if (tot_lev > 0) {
            flQ = (float *)calloc((size_t)(tot_lev + 1), sizeof(float));
            if (!flQ) {
                free(hnh); free(fE); free(fst); free(fsa);
                free(fsf); free(fse); free(fnu); free(fres);
                free(fmE); free(fmV);
                goto fail;
            }
        }

        int xoff = 0, roff = 0, moff = 0, loff = 0;
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
            hnh[i].n_mu    = n->n_mu;
            hnh[i].mu_off  = moff;
            hnh[i].n_lev   = n->n_lev;
            hnh[i].lev_off = loff;

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

            /* Flatten mu-bar angular distribution */
            if (fmE && n->n_mu > 0) {
                for (j = 0; j < n->n_mu; j++) {
                    fmE[moff + j] = (float)n->mu_E[j];
                    fmV[moff + j] = (float)n->mu_bar[j];
                }
            }
            moff += n->n_mu;

            /* Flatten discrete inelastic levels */
            if (flQ && n->n_lev > 0) {
                for (j = 0; j < n->n_lev; j++)
                    flQ[loff + j] = (float)n->lev_Q[j];
            }
            loff += n->n_lev;
        }

        ctx->d_nhdr = gp_up(&ctx->dev, hnh,
                            (size_t)prob->n_nucl * sizeof(h_nhdr_t));
        ctx->d_nE   = gp_up(&ctx->dev, fE,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nst  = gp_up(&ctx->dev, fst,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nsa  = gp_up(&ctx->dev, fsa,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nsf  = gp_up(&ctx->dev, fsf,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nse  = gp_up(&ctx->dev, fse,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nnu  = gp_up(&ctx->dev, fnu,
                            (size_t)(tot_xs + 1) * sizeof(float));
        ctx->d_nres = gp_up(&ctx->dev, fres,
                            (size_t)(tot_res + 1) * sizeof(h_res_t));

        /* Upload mu-bar arrays (delta-Wigner transport correction) */
        if (tot_mu > 0 && fmE && fmV) {
            ctx->d_muE = gp_up(&ctx->dev, fmE,
                               (size_t)(tot_mu + 1) * sizeof(float));
            ctx->d_muV = gp_up(&ctx->dev, fmV,
                               (size_t)(tot_mu + 1) * sizeof(float));
        } else {
            ctx->d_muE = 0;
            ctx->d_muV = 0;
        }

        /* Upload discrete inelastic level Q values */
        if (tot_lev > 0 && flQ) {
            ctx->d_levQ = gp_up(&ctx->dev, flQ,
                                (size_t)(tot_lev + 1) * sizeof(float));
        } else {
            ctx->d_levQ = 0;
        }

        free(hnh); free(fE); free(fst); free(fsa);
        free(fsf); free(fse); free(fnu); free(fres);
        free(fmE); free(fmV); free(flQ);

        if (!ctx->d_nhdr || !ctx->d_nE || !ctx->d_nst ||
            !ctx->d_nsa  || !ctx->d_nsf || !ctx->d_nse ||
            !ctx->d_nnu  || !ctx->d_nres)
            goto fail;
        if (tot_mu > 0 && (!ctx->d_muE || !ctx->d_muV))
            goto fail;
        if (tot_lev > 0 && !ctx->d_levQ)
            goto fail;
        gp_trak(ctx, ctx->d_nhdr,
                (size_t)prob->n_nucl * sizeof(h_nhdr_t), "nhdr");
        gp_trak(ctx, ctx->d_nE,
                (size_t)(tot_xs + 1) * sizeof(float), "nE");
        gp_trak(ctx, ctx->d_nst,
                (size_t)(tot_xs + 1) * sizeof(float), "nst");
        gp_trak(ctx, ctx->d_nsa,
                (size_t)(tot_xs + 1) * sizeof(float), "nsa");
        gp_trak(ctx, ctx->d_nsf,
                (size_t)(tot_xs + 1) * sizeof(float), "nsf");
        gp_trak(ctx, ctx->d_nse,
                (size_t)(tot_xs + 1) * sizeof(float), "nse");
        gp_trak(ctx, ctx->d_nnu,
                (size_t)(tot_xs + 1) * sizeof(float), "nnu");
        gp_trak(ctx, ctx->d_nres,
                (size_t)(tot_res + 1) * sizeof(h_res_t), "nres");
        if (ctx->d_muE) {
            gp_trak(ctx, ctx->d_muE,
                    (size_t)(tot_mu + 1) * sizeof(float), "muE");
            gp_trak(ctx, ctx->d_muV,
                    (size_t)(tot_mu + 1) * sizeof(float), "muV");
        }
        if (ctx->d_levQ) {
            gp_trak(ctx, ctx->d_levQ,
                    (size_t)(tot_lev + 1) * sizeof(float), "levQ");
        }
    }

    printf("moa: NVIDIA GPU initialised -- %d surfs, %d cells, %d nucs, "
           "%d xs pts, %d resonances, %d mu-bar pts, %d inel levels\n",
           ctx->n_surf, ctx->n_cell, ctx->n_nuc,
           ctx->n_xs, ctx->n_res, tot_mu, tot_lev);
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
    nv_dev_t *dev = &ctx->dev;
    int i, rc;

    if (!ctx->loaded || n_src <= 0) return -1;

    int fb_max = n_src * 3;
    if (fb_max > MO_MAX_PART) fb_max = MO_MAX_PART;

    /* ---- Build particle + RNG arrays on CPU ---- */
    h_part_t *hp = (h_part_t *)calloc((size_t)n_src, sizeof(h_part_t));
    h_rng_t  *hr = (h_rng_t *)calloc((size_t)n_src, sizeof(h_rng_t));
    if (!hp || !hr) { free(hp); free(hr); return -1; }

    int n_dead = 0;
    for (i = 0; i < n_src; i++) {
        hp[i].x    = (float)src_bank[i].x;
        hp[i].y    = (float)src_bank[i].y;
        hp[i].z    = (float)src_bank[i].z;
        hp[i].E    = (float)src_bank[i].E;
        hp[i].wt   = (float)src_bank[i].wt;
        hp[i].alive = 1;

        hp[i].cell = cg_find(prob, src_bank[i].x,
                             src_bank[i].y, src_bank[i].z);
        if (hp[i].cell < 0 ||
            prob->cell[hp[i].cell].mat < 0) {
            hp[i].alive = 0;
            hp[i].cell  = 0;
            n_dead++;
        }

        /* Isotropic direction on CPU */
        {
            double mu  = 2.0 * rn_dbl(&prob->rng) - 1.0;
            double phi = 2.0 * MO_PI * rn_dbl(&prob->rng);
            double st  = sqrt(1.0 - mu * mu);
            hp[i].u = (float)(st * cos(phi));
            hp[i].v = (float)(st * sin(phi));
            hp[i].w = (float)mu;
        }

        /* Seed xorwow — unique per batch via master RNG */
        {
            unsigned int seed = (unsigned int)(rn_dbl(&prob->rng)
                                * 4294967296.0) ^ (unsigned int)i;
            hr[i].s0 = sm32(&seed);
            hr[i].s1 = sm32(&seed);
            hr[i].s2 = sm32(&seed);
            hr[i].s3 = sm32(&seed);
            hr[i].s4 = sm32(&seed);
            hr[i].d  = 0;
        }
    }

    if (n_dead > 0)
        fprintf(stderr, "gp_batch: %d/%d source particles born DEAD"
                " (outside geometry)\n", n_dead, n_src);

    /* ---- Allocate per-batch device memory ---- */
    size_t sz_parts = (size_t)n_src * sizeof(h_part_t);
    size_t sz_rngs  = (size_t)n_src * sizeof(h_rng_t);
    size_t sz_fbank = (size_t)fb_max * sizeof(h_site_t);
    size_t sz_nfb   = sizeof(int);

    size_t sz_dbg   = (size_t)n_src * sizeof(int);

    /* Free previous per-batch allocs if size changed */
    if (ctx->d_parts && ctx->n_part_alloc != n_src) {
        nv_rt_free(dev, ctx->d_parts);
        nv_rt_free(dev, ctx->d_rngs);
        nv_rt_free(dev, ctx->d_fbank);
        nv_rt_free(dev, ctx->d_nfb);
        if (ctx->h_dbg) { nv_rt_mfre(dev, ctx->h_dbg); ctx->h_dbg = NULL; }
        else if (ctx->d_dbg) nv_rt_free(dev, ctx->d_dbg);
        ctx->d_parts = 0;
        ctx->d_dbg   = 0;
    }

    if (!ctx->d_parts) {
        ctx->d_parts = nv_rt_alloc(dev, sz_parts);
        ctx->d_rngs  = nv_rt_alloc(dev, sz_rngs);
        ctx->d_fbank = nv_rt_alloc(dev, sz_fbank);
        ctx->d_nfb   = nv_rt_alloc(dev, sz_nfb);
        ctx->n_part_alloc = n_src;
        if (!ctx->d_parts || !ctx->d_rngs ||
            !ctx->d_fbank || !ctx->d_nfb) {
            free(hp); free(hr);
            return -1;
        }
        /* Debug buffer: mapped host memory — survives GPU crashes.
         * Falls back to device memory if mapping unavailable. */
        {
            void *hp_dbg = NULL;
            CUdevptr dp_dbg = 0;
            int mrc = nv_rt_mmap(dev, &hp_dbg, &dp_dbg, sz_dbg);
            if (mrc == NV_RT_OK) {
                ctx->h_dbg = (int *)hp_dbg;
                ctx->d_dbg = dp_dbg;
            } else {
                /* Fallback: device memory (readback may fail after crash) */
                ctx->d_dbg = nv_rt_alloc(dev, sz_dbg);
                ctx->h_dbg = NULL;
            }
        }
        gp_trak(ctx, ctx->d_parts, sz_parts, "parts");
        gp_trak(ctx, ctx->d_rngs,  sz_rngs,  "rngs");
        gp_trak(ctx, ctx->d_fbank, sz_fbank, "fbank");
        gp_trak(ctx, ctx->d_nfb,   sz_nfb,   "n_fbank");
        if (ctx->d_dbg)
            gp_trak(ctx, ctx->d_dbg, sz_dbg, "dbg(mapped)");

        /* Block-hit instrumentation buffer (2048 blocks max) */
        ctx->n_bkhit = 2048;
        ctx->d_bkhit = nv_rt_alloc(dev,
            (size_t)ctx->n_bkhit * sizeof(int));
    }

    /* ---- Upload ---- */
    rc = nv_rt_h2d(dev, ctx->d_parts, hp, sz_parts);
    if (rc != NV_RT_OK) { free(hp); free(hr); return -1; }

    rc = nv_rt_h2d(dev, ctx->d_rngs, hr, sz_rngs);
    if (rc != NV_RT_OK) { free(hp); free(hr); return -1; }

    /* Zero fission bank counter */
    int zero = 0;
    rc = nv_rt_h2d(dev, ctx->d_nfb, &zero, sizeof(int));
    if (rc != NV_RT_OK) { free(hp); free(hr); return -1; }

    /* Zero debug breadcrumb buffer */
    if (ctx->h_dbg) {
        memset(ctx->h_dbg, 0, sz_dbg);
    } else if (ctx->d_dbg) {
        int *zdbg = (int *)calloc((size_t)n_src, sizeof(int));
        if (zdbg) {
            nv_rt_h2d(dev, ctx->d_dbg, zdbg, sz_dbg);
            free(zdbg);
        }
    }

    /* Zero block-hit counters */
    if (ctx->d_bkhit) {
        int *zbk = (int *)calloc((size_t)ctx->n_bkhit, sizeof(int));
        if (zbk) {
            nv_rt_h2d(dev, ctx->d_bkhit, zbk,
                      (size_t)ctx->n_bkhit * sizeof(int));
            free(zbk);
        }
    }

    free(hp);
    free(hr);

    /* ---- Build kernel arguments ----
     * CUDA Driver API: void **args — array of pointers-to-values.
     * For device pointers, &CUdevptr. For scalars, &int / &float.
     * So much saner than HSA's flat kernarg buffer. */
    float k_eff_f = (float)prob->k_eff;
    int n_part_i  = n_src;
    int n_surf_i  = ctx->n_surf;
    int n_cell_i  = ctx->n_cell;
    int n_mat_i   = ctx->n_mat;
    int n_nuc_i   = ctx->n_nuc;

    void *args[] = {
        &ctx->d_parts,    /*  0: parts   */
        &ctx->d_rngs,     /*  1: rngs    */
        &ctx->d_fbank,    /*  2: fbank   */
        &ctx->d_nfb,      /*  3: n_fbank */
        &ctx->d_surfs,    /*  4: surfs   */
        &ctx->d_cells,    /*  5: cells   */
        &ctx->d_csur,     /*  6: csur    */
        &ctx->d_csen,     /*  7: csen    */
        &ctx->d_mats,     /*  8: mats    */
        &ctx->d_mnid,     /*  9: mnid    */
        &ctx->d_mfrc,     /* 10: mfrc    */
        &ctx->d_nhdr,     /* 11: nhdr    */
        &ctx->d_nE,       /* 12: nE      */
        &ctx->d_nst,      /* 13: nst     */
        &ctx->d_nsa,      /* 14: nsa     */
        &ctx->d_nsf,      /* 15: nsf     */
        &ctx->d_nse,      /* 16: nse     */
        &ctx->d_nnu,      /* 17: nnu     */
        &ctx->d_nres,     /* 18: nres    */
        &n_part_i,        /* 19: n_part  */
        &n_surf_i,        /* 20: n_surf  */
        &n_cell_i,        /* 21: n_cell  */
        &n_mat_i,         /* 22: n_mat   */
        &n_nuc_i,         /* 23: n_nuc   */
        &k_eff_f,         /* 24: k_eff   */
        &fb_max,          /* 25: fb_max  */
        &ctx->d_muE,      /* 26: muE     */
        &ctx->d_muV,      /* 27: muV     */
        &ctx->d_levQ,     /* 28: levQ    */
        &ctx->d_dbg,      /* 29: dbg     */
        &ctx->d_bkhit     /* 30: bkhit   */
    };

    /* ---- Launch ---- */
    unsigned block = 256;
    unsigned grid  = ((unsigned)n_src + block - 1) / block;

    rc = nv_rt_launch(dev, &ctx->kern,
                      grid, 1, 1, block, 1, 1, 0, args);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gp_batch: launch failed (%d)\n", rc);
        return -1;
    }

    /* ---- Sync ---- */
    rc = nv_rt_sync(dev);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gp_batch: sync failed (%d)\n", rc);
        gp_abnd(ctx, n_src, "ILLEGAL MEMORY ACCESS");
        return -1;
    }

    /* ---- Debug: step/crossing/inf summary ---- */
    {
        int *dbgp = ctx->h_dbg;
        if (!dbgp) {
            dbgp = (int *)calloc((size_t)n_src, sizeof(int));
            if (dbgp)
                nv_rt_d2h(dev, dbgp, ctx->d_dbg,
                           (size_t)n_src * sizeof(int));
        }
        if (dbgp) {
            long s_sum = 0, x_sum = 0, inf_sum = 0, esc_sum = 0;
            int nz = 0, n_inf = 0, n_esc = 0;
            /* Last 12 slots are atomic counters */
            int lim = n_src - 12;
            if (lim < 0) lim = 0;
            for (i = 0; i < lim && i < 100000; i++) {
                int v = dbgp[i];
                int steps = v & 0xFFF;
                int cross = (v >> 12) & 0xF;
                int xinf  = (v >> 16) & 0xF;
                int xesc  = (v >> 20) & 0xF;
                s_sum += steps;
                x_sum += cross;
                inf_sum += xinf;
                esc_sum += xesc;
                if (steps == 0) nz++;
                if (xinf > 0) n_inf++;
                if (xesc > 0) n_esc++;
            }
            int nlim = lim > 0 ? lim : 1;
            fprintf(stderr, "dbg: steps=%.1f cross=%.1f"
                    " inf_inside=%ld(%d) esc=%ld(%d) dead=%d\n",
                    (double)s_sum / nlim,
                    (double)x_sum / nlim,
                    inf_sum, n_inf, esc_sum, n_esc, nz);
            /* XS diagnostic: slots 0-7 (tid=0, post-loop, race-free) */
            if (lim > 8) {
                fprintf(stderr, "  XS[tid0]: sig_t=%.6f sig_a=%.6f"
                        " sig_f=%.6f sig_el=%.6f nu=%.3f"
                        " E0=%d Ef=%d cell=%d\n",
                        dbgp[0] / 1e6, dbgp[1] / 1e6,
                        dbgp[2] / 1e6, dbgp[3] / 1e6,
                        dbgp[4] / 1e3, dbgp[5],
                        dbgp[6], dbgp[7]);
            }
            /* Atomic counters: last 3 slots */
            if (n_src >= 4) {
                int tot_coll = dbgp[n_src - 1];
                int tot_site = dbgp[n_src - 2];
                int tot_xing = dbgp[n_src - 3];
                fprintf(stderr, "  totals: coll=%d sites=%d xing=%d"
                    " (sites/part=%.3f coll/part=%.1f)\n",
                    tot_coll, tot_site, tot_xing,
                    (double)tot_site / (n_src > 0 ? n_src : 1),
                    (double)tot_coll / (n_src > 0 ? n_src : 1));
            }
            /* Scatter type counters: slots -4..-10 */
            if (n_src >= 11) {
                int n_fwd  = dbgp[n_src - 4];
                int n_comp = dbgp[n_src - 5];
                int n_inel = dbgp[n_src - 7];
                int mb_sum = dbgp[n_src - 6];
                int n_dlev = dbgp[n_src - 8];
                int n_leak = dbgp[n_src - 9];
                int n_phan = dbgp[n_src - 10];
                int n_el   = n_fwd + n_comp;
                fprintf(stderr, "  scatter: fwd=%d comp=%d inel=%d"
                    " (el_total=%d fwd_frac=%.3f avg_mubar=%.3f)\n",
                    n_fwd, n_comp, n_inel, n_el,
                    n_el > 0 ? (double)n_fwd / n_el : 0.0,
                    n_el > 0 ? (double)mb_sum / (1000.0 * n_el) : 0.0);
                int n_oob = (n_src >= 12) ? dbgp[n_src - 11] : 0;
                fprintf(stderr, "  xing: leak=%d phantom=%d"
                    " dlev=%d oob_coll=%d (xing_total=%d)\n",
                    n_leak, n_phan, n_dlev, n_oob,
                    dbgp[n_src - 3]);
            }
            if (dbgp != ctx->h_dbg) free(dbgp);
        }
    }

    /* DIAG: read back particles to check alive/dead/cell state */
    {
        h_part_t *pp = (h_part_t *)calloc((size_t)n_src, sizeof(h_part_t));
        if (pp) {
            nv_rt_d2h(dev, pp, ctx->d_parts, (size_t)n_src * sizeof(h_part_t));
            int nalv = 0, nded = 0, ncell[8] = {0};
            int nstep = 0;
            for (i = 0; i < n_src; i++) {
                if (pp[i].alive) nalv++; else nded++;
                if (pp[i].cell >= 0 && pp[i].cell < 8) ncell[pp[i].cell]++;
            }
            /* count particles that hit step limit (alive=1 still) */
            for (i = 0; i < n_src; i++)
                if (pp[i].alive) nstep++;
            fprintf(stderr, "  parts: alive=%d dead=%d step_lim=%d"
                    " cell0=%d cell1=%d cell2=%d\n",
                    nalv, nded, nstep, ncell[0], ncell[1], ncell[2]);
            /* Print first 3 dead particle positions */
            int shown = 0;
            for (i = 0; i < n_src && shown < 3; i++) {
                if (!pp[i].alive) {
                    float r2 = pp[i].x*pp[i].x + pp[i].y*pp[i].y + pp[i].z*pp[i].z;
                    fprintf(stderr, "  dead[%d]: r=%.3f cell=%d E=%.1f wt=%.4f\n",
                            i, (double)sqrtf(r2), pp[i].cell, (double)pp[i].E, (double)pp[i].wt);
                    shown++;
                }
            }
            free(pp);
        }
    }

    /* DIAG: block-hit counters — read back and print non-zero blocks */
    if (ctx->d_bkhit && ctx->n_bkhit > 0) {
        int *bk = (int *)calloc((size_t)ctx->n_bkhit, sizeof(int));
        if (bk) {
            nv_rt_d2h(dev, bk, ctx->d_bkhit,
                      (size_t)ctx->n_bkhit * sizeof(int));
            fprintf(stderr, "  bkhit: ");
            int nb = 0;
            for (i = 0; i < ctx->n_bkhit; i++) {
                if (bk[i] > 0) nb++;
            }
            fprintf(stderr, "%d blocks hit\n", nb);
            /* Full dump to file for analysis */
            {
                FILE *bfp = fopen("bkhit.txt", "w");
                if (bfp) {
                    for (i = 0; i < ctx->n_bkhit; i++) {
                        if (bk[i] > 0)
                            fprintf(bfp, "BB%-4d %12d\n", i, bk[i]);
                    }
                    fclose(bfp);
                    fprintf(stderr, "  bkhit: wrote bkhit.txt\n");
                }
            }
            free(bk);
        }
    }

    /* ---- Download fission bank ---- */
    int n_fb = 0;
    rc = nv_rt_d2h(dev, &n_fb, ctx->d_nfb, sizeof(int));
    if (rc != NV_RT_OK) return -1;

    if (n_fb > fb_max) n_fb = fb_max;
    fprintf(stderr, "  n_fbank=%d (fb_max=%d, n_src=%d, k_raw=%.4f)\n",
            n_fb, fb_max, n_src, (double)n_fb / (double)n_src);

    if (n_fb > 0) {
        h_site_t *hfb = (h_site_t *)calloc((size_t)n_fb,
                                            sizeof(h_site_t));
        if (!hfb) return -1;

        rc = nv_rt_d2h(dev, hfb, ctx->d_fbank,
                       (size_t)n_fb * sizeof(h_site_t));
        if (rc != NV_RT_OK) { free(hfb); return -1; }

        /* Convert float -> double for CPU bank */
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

        /* Diagnostic: check fission site validity */
        {
            int n_out = 0, n_elo = 0, n_nan = 0, n_prn = 0;
            double R2 = 8.7407 * 8.7407;
            for (i = 0; i < ncopy && i < 100000; i++) {
                double r2 = fis_bank[i].x * fis_bank[i].x
                          + fis_bank[i].y * fis_bank[i].y
                          + fis_bank[i].z * fis_bank[i].z;
                if (r2 > R2) {
                    n_out++;
                    if (n_prn < 5) {
                        fprintf(stderr, "  out[%d]: r=%.4f (%.3f,%.3f,%.3f)"
                                " E=%.0f\n", i, sqrt(r2),
                                fis_bank[i].x, fis_bank[i].y,
                                fis_bank[i].z, fis_bank[i].E);
                        n_prn++;
                    }
                }
                if (fis_bank[i].E < 1.0) n_elo++;
                if (fis_bank[i].x != fis_bank[i].x) n_nan++;
            }
            /* Energy statistics */
            double e_sum = 0.0, e_min = 1e30, e_max = 0.0;
            int n_egt10m = 0, n_elt1k = 0;
            for (i = 0; i < ncopy; i++) {
                double e = fis_bank[i].E;
                e_sum += e;
                if (e < e_min) e_min = e;
                if (e > e_max) e_max = e;
                if (e > 10.0e6) n_egt10m++;
                if (e < 1000.0) n_elt1k++;
            }
            fprintf(stderr, "fbank: %d sites, %d outside (R=8.7407),"
                    " %d E<1eV, %d nan\n",
                    ncopy, n_out, n_elo, n_nan);
            /* Radial distribution of fission sites */
            double r2_sum = 0.0;
            for (i = 0; i < ncopy; i++) {
                r2_sum += fis_bank[i].x * fis_bank[i].x
                        + fis_bank[i].y * fis_bank[i].y
                        + fis_bank[i].z * fis_bank[i].z;
            }
            double r_rms = sqrt(r2_sum / (ncopy > 0 ? ncopy : 1));
            fprintf(stderr, "  E: avg=%.0f min=%.0f max=%.0f"
                    " >10MeV=%d <1keV=%d"
                    " r_rms=%.3f\n",
                    e_sum / (ncopy > 0 ? ncopy : 1),
                    e_min, e_max, n_egt10m, n_elt1k, r_rms);
        }
    } else {
        *n_fis = 0;
    }

    return 0;
}

/* ---- gp_free ---- */

void
gp_free(gp_ctx_t *ctx)
{
    nv_dev_t *dev = &ctx->dev;

    /* Free device memory */
    if (ctx->d_surfs)  nv_rt_free(dev, ctx->d_surfs);
    if (ctx->d_cells)  nv_rt_free(dev, ctx->d_cells);
    if (ctx->d_csur)   nv_rt_free(dev, ctx->d_csur);
    if (ctx->d_csen)   nv_rt_free(dev, ctx->d_csen);
    if (ctx->d_mats)   nv_rt_free(dev, ctx->d_mats);
    if (ctx->d_mnid)   nv_rt_free(dev, ctx->d_mnid);
    if (ctx->d_mfrc)   nv_rt_free(dev, ctx->d_mfrc);
    if (ctx->d_nhdr)   nv_rt_free(dev, ctx->d_nhdr);
    if (ctx->d_nE)     nv_rt_free(dev, ctx->d_nE);
    if (ctx->d_nst)    nv_rt_free(dev, ctx->d_nst);
    if (ctx->d_nsa)    nv_rt_free(dev, ctx->d_nsa);
    if (ctx->d_nsf)    nv_rt_free(dev, ctx->d_nsf);
    if (ctx->d_nse)    nv_rt_free(dev, ctx->d_nse);
    if (ctx->d_nnu)    nv_rt_free(dev, ctx->d_nnu);
    if (ctx->d_nres)   nv_rt_free(dev, ctx->d_nres);
    if (ctx->d_muE)    nv_rt_free(dev, ctx->d_muE);
    if (ctx->d_muV)    nv_rt_free(dev, ctx->d_muV);
    if (ctx->d_levQ)   nv_rt_free(dev, ctx->d_levQ);
    if (ctx->d_parts)  nv_rt_free(dev, ctx->d_parts);
    if (ctx->d_rngs)   nv_rt_free(dev, ctx->d_rngs);
    if (ctx->d_fbank)  nv_rt_free(dev, ctx->d_fbank);
    if (ctx->d_nfb)    nv_rt_free(dev, ctx->d_nfb);
    if (ctx->h_dbg) {
        nv_rt_mfre(dev, ctx->h_dbg);
    } else if (ctx->d_dbg) {
        nv_rt_free(dev, ctx->d_dbg);
    }

    if (ctx->d_bkhit)  nv_rt_free(dev, ctx->d_bkhit);

    /* Unload kernel and shut down */
    if (ctx->loaded)
        nv_rt_unload(dev, &ctx->kern);
    nv_rt_shut(dev);

    memset(ctx, 0, sizeof(*ctx));

    printf("moa: NVIDIA GPU shutdown complete\n");
}
