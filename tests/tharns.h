/* tharns.h -- Moa test harness
 * Borrowed from BarraCUDA with the grace of a seagull
 * stealing chips. Same pattern, different bird. */
#ifndef THARNS_H
#define THARNS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef void (*tfunc_t)(void);

typedef struct {
    const char *tname;
    const char *tcats;
    tfunc_t     func;
} tcase_t;

#define TH_MAXTS 256

extern tcase_t th_list[];
extern int th_cnt;
extern int npass, nfail, nskip;

/* Self-registration via constructor. gcc/clang only.
 * MSVC users: we've been over this. */
#define TH_REG(cat, fn) \
    __attribute__((constructor)) static void reg_##fn(void) { \
        if (th_cnt < TH_MAXTS) \
            th_list[th_cnt++] = (tcase_t){#fn, cat, fn}; \
    }

/* ---- Assertions ---- */

#define CHECK(x) do { if (!(x)) { \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    nfail++; return; } } while(0)

#define CHEQ(a, b)   CHECK((a) == (b))
#define CHNE(a, b)   CHECK((a) != (b))
#define CHSTR(a, b)  CHECK(strcmp((a),(b)) == 0)
#define PASS()       do { npass++; } while(0)
#define SKIP(r)      do { nskip++; printf("  SKIP: %s\n", r); return; } while(0)

/* Float comparison with tolerance */
#define CHFLT(a, b, tol) do { \
    double _a = (double)(a), _b = (double)(b), _t = (double)(tol); \
    if (fabs(_a - _b) > _t) { \
        printf("  FAIL %s:%d: %.10g != %.10g (tol=%.2g)\n", \
               __FILE__, __LINE__, _a, _b, _t); \
        nfail++; return; } } while(0)

#endif /* THARNS_H */
