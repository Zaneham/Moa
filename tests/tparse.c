/* tparse.c -- ENDF parser tests
 * Testing a parser for a format designed before we landed on the
 * moon. The format has aged about as well as you'd expect. */

#include "../moa.h"
#include "tharns.h"

/* ---- FORTRAN float parsing ----
 * The crucial bit: parsing ENDF's unhinged E-notation. */

static void t_nd_fval_basic(void)
{
    /* Standard notation */
    CHFLT(nd_fval(" 1.23456+03", 11), 1234.56, 0.01);
    CHFLT(nd_fval("-2.34000-01", 11), -0.234, 0.001);
    CHFLT(nd_fval(" 0.00000+00", 11), 0.0, 1e-10);
    PASS();
}
TH_REG("parse", t_nd_fval_basic)

static void t_nd_fval_exp(void)
{
    /* With explicit E */
    CHFLT(nd_fval(" 1.234E+003", 11), 1234.0, 0.1);
    CHFLT(nd_fval("-5.678E-002", 11), -0.05678, 0.0001);
    PASS();
}
TH_REG("parse", t_nd_fval_exp)

static void t_nd_fval_d(void)
{
    /* FORTRAN D-notation */
    CHFLT(nd_fval(" 1.234D+003", 11), 1234.0, 0.1);
    PASS();
}
TH_REG("parse", t_nd_fval_d)

static void t_nd_fval_neg(void)
{
    /* Negative exponent with no E */
    CHFLT(nd_fval("+1.23456-03", 11), 0.00123456, 1e-8);
    PASS();
}
TH_REG("parse", t_nd_fval_neg)

static void t_nd_fval_zero(void)
{
    /* Blank/zero field */
    CHFLT(nd_fval("           ", 11), 0.0, 1e-15);
    PASS();
}
TH_REG("parse", t_nd_fval_zero)

/* ---- Integer parsing ---- */

static void t_nd_ival(void)
{
    CHEQ(nd_ival("  92235"), 92235);
    CHEQ(nd_ival("    -42"), -42);
    CHEQ(nd_ival("      0"), 0);
    CHEQ(nd_ival("     +7"), 7);
    PASS();
}
TH_REG("parse", t_nd_ival)
