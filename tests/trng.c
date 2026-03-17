/* trng.c -- RNG tests
 * Statistical quality checks for xoshiro256**. If these fail,
 * every Monte Carlo result downstream is suspect, which makes
 * this the most important test file that nobody reads. */

#define KAURI_IMPL
#include "../moa.h"
#include "tharns.h"

/* ---- Determinism test ----
 * Same seed must produce same sequence. Always. No exceptions.
 * Non-deterministic RNG in a Monte Carlo code is like a compass
 * that points at "interesting." */

static void t_rng_det(void)
{
    mo_rng_t r1, r2;
    int i;

    rn_init(&r1, 12345);
    rn_init(&r2, 12345);

    for (i = 0; i < 1000; i++) {
        uint64_t a = rn_next(&r1);
        uint64_t b = rn_next(&r2);
        CHECK(a == b);
    }
    PASS();
}
TH_REG("rng", t_rng_det)

/* ---- Different seeds produce different sequences ---- */

static void t_rng_diff(void)
{
    mo_rng_t r1, r2;
    rn_init(&r1, 1);
    rn_init(&r2, 2);

    /* They should diverge immediately */
    CHECK(rn_next(&r1) != rn_next(&r2));
    PASS();
}
TH_REG("rng", t_rng_diff)

/* ---- Uniform distribution [0, 1) ----
 * Generate many doubles, check they're in range and roughly
 * uniform via chi-squared on 10 bins. */

static void t_rng_unif(void)
{
    mo_rng_t rng;
    int bins[10] = {0};
    int i, N = 100000;
    double chi2 = 0.0;
    double expect = (double)N / 10.0;

    rn_init(&rng, 42);

    for (i = 0; i < N; i++) {
        double x = rn_dbl(&rng);
        CHECK(x >= 0.0);
        CHECK(x < 1.0);

        int b = (int)(x * 10.0);
        if (b >= 10) b = 9;
        bins[b]++;
    }

    /* Chi-squared: should be < 16.92 for 9 dof at 95% confidence */
    for (i = 0; i < 10; i++) {
        double diff = (double)bins[i] - expect;
        chi2 += diff * diff / expect;
    }
    CHECK(chi2 < 25.0);  /* generous bound for 100k samples */
    PASS();
}
TH_REG("rng", t_rng_unif)

/* ---- Jump advances state ----
 * After jumping, the sequence should differ from the original. */

static void t_rng_jump(void)
{
    mo_rng_t r1, r2;
    rn_init(&r1, 42);
    r2 = r1;

    rn_jump(&r2);

    /* Post-jump should differ from pre-jump */
    CHECK(rn_next(&r1) != rn_next(&r2));
    PASS();
}
TH_REG("rng", t_rng_jump)

/* ---- No zero state ----
 * xoshiro256** should never enter all-zero state from any
 * normal seed. If it does, it stays there forever. */

static void t_rng_nzero(void)
{
    mo_rng_t rng;
    int i;

    rn_init(&rng, 0);  /* even seed=0 should work */

    for (i = 0; i < 100; i++) {
        uint64_t v = rn_next(&rng);
        (void)v;
        /* At least one state word should be nonzero */
        CHECK(rng.s[0] || rng.s[1] || rng.s[2] || rng.s[3]);
    }
    PASS();
}
TH_REG("rng", t_rng_nzero)
