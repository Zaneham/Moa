/* rng.c -- xoshiro256** PRNG
 *
 * The beating heart of Monte Carlo: a random number generator that
 * isn't actually random at all. Deterministic chaos, carefully
 * curated. Like a British queue that only looks orderly.
 *
 * Algorithm: xoshiro256** by Blackman & Vigna (2018).
 * Period: 2^256 - 1. Enough for every neutron in every reactor
 * that will ever exist, plus a few spares.
 *
 * Reference:
 *   Blackman, D., & Vigna, S. (2021). Scrambled linear
 *   pseudorandom number generators. ACM Transactions on
 *   Mathematical Software, 47(4), Article 36.
 */

#include "../moa.h"

/* ---- splitmix64 for seeding ----
 * A simpler generator used only to bootstrap xoshiro's 256-bit
 * state from a single 64-bit seed. Like using a match to light
 * a reactor. Proportionate. */

static uint64_t
sm_next(uint64_t *st)
{
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t
rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

/* ---- Public API ---- */

void
rn_init(mo_rng_t *r, uint64_t seed)
{
    /* Feed the seed through splitmix64 to fill all four words.
     * A zero seed is technically valid but spiritually suspect. */
    r->s[0] = sm_next(&seed);
    r->s[1] = sm_next(&seed);
    r->s[2] = sm_next(&seed);
    r->s[3] = sm_next(&seed);
}

uint64_t
rn_next(mo_rng_t *r)
{
    /* xoshiro256**: the starstar variant. Multiply-rotate-multiply
     * on s[1]*5, then rotate left 7, then *9. The magic numbers
     * were chosen by people who understand Galois fields. We just
     * trust them and move on. */
    uint64_t *s = r->s;
    uint64_t  res = rotl64(s[1] * 5, 7) * 9;
    uint64_t  t   = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3]  = rotl64(s[3], 45);

    return res;
}

double
rn_dbl(mo_rng_t *r)
{
    /* Convert to [0, 1) by taking the upper 53 bits and dividing.
     * IEEE 754 double has 53 bits of mantissa, so this gives us
     * maximal resolution. No wasted entropy. */
    return (double)(rn_next(r) >> 11) * 0x1.0p-53;
}

void
rn_jump(mo_rng_t *r)
{
    /* Jump 2^128 states ahead. For splitting the sequence across
     * GPU threads: thread N uses jump(N) from the master seed.
     * Each thread gets 2^128 values before overlap, which is
     * more than enough unless you're simulating the heat death
     * of the universe. In which case, use jump2x256. */
    static const uint64_t jtab[4] = {
        0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL,
        0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i, b;

    for (i = 0; i < 4; i++) {
        for (b = 0; b < 64; b++) {
            if (jtab[i] & (1ULL << b)) {
                s0 ^= r->s[0];
                s1 ^= r->s[1];
                s2 ^= r->s[2];
                s3 ^= r->s[3];
            }
            rn_next(r);
        }
    }
    r->s[0] = s0;
    r->s[1] = s1;
    r->s[2] = s2;
    r->s[3] = s3;
}
