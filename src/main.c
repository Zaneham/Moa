/* main.c -- Moa CLI driver
 *
 * The front door. Parse arguments, load the problem, run the
 * calculation, print results. Simple as.
 *
 * Usage: moa input.inp [--seed N] [--verbose]
 *
 * Named after the moa: 3.6 metres tall, 230 kg, couldn't fly,
 * extinct before anyone thought to write down what it tasted
 * like. But it was here first, and that counts for something.
 */

#define KAURI_IMPL
#include "../moa.h"

/* Arena backing store. 4 MB should be enough for scratch allocations.
 * If you need more, you're probably doing something wrong. */
#define MO_ARENA_SZ (4u * 1024u * 1024u)

#ifdef MOA_GPU
void *g_gpu_ctx = NULL;
#endif

static void
mo_use(void)
{
    printf("usage: moa <input.inp> [options]\n");
    printf("  --seed N     RNG seed (default: 42)\n");
    printf("  --verbose    print every batch\n");
    printf("  --help       this message\n");
}

int
main(int argc, char **argv)
{
    mo_prob_t *prob;
    const char *inp_path = NULL;
    uint64_t seed = 42;
    int verbose = 0;
    int i;

    /* Parse CLI args */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            mo_use();
            return 0;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (argv[i][0] != '-') {
            inp_path = argv[i];
        } else {
            fprintf(stderr, "moa: unknown option '%s'\n", argv[i]);
            mo_use();
            return 1;
        }
    }

    if (!inp_path) {
        fprintf(stderr, "moa: no input file specified\n");
        mo_use();
        return 1;
    }

    /* Allocate problem context on heap.
     * mo_prob_t is ~15 MB — stack allocation would make Windows
     * do that thing where it pretends nothing happened and then
     * crashes five functions later. */
    prob = (mo_prob_t *)calloc(1, sizeof(mo_prob_t));
    if (!prob) {
        fprintf(stderr, "moa: failed to allocate problem context\n");
        return 1;
    }

    /* Init Kauri arena */
    ka_init(&prob->arena, NULL, MO_ARENA_SZ, KA_CHAIN);

    /* Init RNG */
    rn_init(&prob->rng, seed);

    /* Load problem definition */
    prob->verbose = verbose;
    printf("moa: loading %s\n", inp_path);
    if (io_load(prob, inp_path) < 0) {
        fprintf(stderr, "moa: failed to load problem\n");
        ka_free(&prob->arena);
        free(prob);
        return 1;
    }

    printf("moa: %d surfaces, %d cells, %d nuclides, %d materials\n",
           prob->n_surf, prob->n_cell, prob->n_nucl, prob->n_mat);
    fflush(stdout);

    /* Init tallies */
    tl_init(&prob->tl_flux);
    tl_init(&prob->tl_fiss);
    te_init(&prob->etl_flux);

    /* Doppler broadening: if any cell has temp > 0, broaden nuclides.
     * Done once at init, modifies xs tables in-place. */
    {
        int c;
        double max_temp = 0.0;
        for (c = 0; c < prob->n_cell; c++) {
            if (prob->cell[c].temp > max_temp)
                max_temp = prob->cell[c].temp;
        }
        if (max_temp > 1.0) {
            int n;
            for (n = 0; n < prob->n_nucl; n++) {
                dp_brod(&prob->nucl[n], max_temp);
            }
        }
    }

    /* Init GPU if requested */
#ifdef MOA_GPU
    g_gpu_ctx = NULL;
    if (prob->use_gpu) {
        fprintf(stderr, "moa: GPU requested, allocating ctx...\n");
        fflush(stderr);
        gp_ctx_t *gctx = (gp_ctx_t *)calloc(1, sizeof(gp_ctx_t));
        if (gctx && gp_init(gctx, prob, prob->hsaco_path) == 0) {
            /* SNAP: set MOA_SNAP=1 env var to dump GPU param values */
            if (getenv("MOA_SNAP")) {
                gctx->snap = 1;
                printf("moa: SNAP mode enabled — GPU params will be dumped\n");
            }
            g_gpu_ctx = gctx;
        } else {
            fprintf(stderr, "moa: GPU init failed, falling back to CPU\n");
            prob->use_gpu = 0;
            free(gctx);
        }
    }
#endif

    /* Run transport */
    if (prob->mode == MO_MODE_FIXD) {
        tf_run(prob);
    } else {
        tc_run(prob);
    }

    /* Shutdown GPU */
#ifdef MOA_GPU
    if (g_gpu_ctx) {
        gp_free((gp_ctx_t *)g_gpu_ctx);
        free(g_gpu_ctx);
        g_gpu_ctx = NULL;
    }
#endif

    /* Cleanup */
    if (prob->mesh) {
        free(prob->mesh);
        prob->mesh = NULL;
    }
    ka_free(&prob->arena);
    free(prob);

    printf("moa: done.\n");
    return 0;
}
