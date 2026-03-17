/* tgeom.c -- CSG geometry tests
 * Testing whether our neutrons know which side of a sphere they're on.
 * Surprisingly non-trivial for objects that don't have eyes. */

#include "../moa.h"
#include "tharns.h"

/* ---- Sphere evaluation ---- */

static void t_cg_sph_eval(void)
{
    mo_surf_t s;
    cg_sph(&s, 0.0, 0.0, 0.0, 5.0);

    /* Origin: inside (negative) */
    CHECK(cg_eval(&s, 0.0, 0.0, 0.0) < 0.0);

    /* On surface: ≈ 0 */
    CHFLT(cg_eval(&s, 5.0, 0.0, 0.0), 0.0, 1e-10);

    /* Outside (positive) */
    CHECK(cg_eval(&s, 10.0, 0.0, 0.0) > 0.0);
    PASS();
}
TH_REG("geom", t_cg_sph_eval)

/* ---- Sphere ray intersection ---- */

static void t_cg_sph_dist(void)
{
    mo_surf_t s;
    double d;

    cg_sph(&s, 0.0, 0.0, 0.0, 5.0);

    /* Ray from origin along +x: should hit at d=5 */
    d = cg_dist(&s, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    CHFLT(d, 5.0, 1e-6);

    /* Ray from outside pointing at sphere center */
    d = cg_dist(&s, 10.0, 0.0, 0.0, -1.0, 0.0, 0.0);
    CHFLT(d, 5.0, 1e-6);

    /* Ray going away from sphere: should be HUGE_VAL */
    d = cg_dist(&s, 10.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    CHECK(d > 1e10);
    PASS();
}
TH_REG("geom", t_cg_sph_dist)

/* ---- Plane evaluation ---- */

static void t_cg_plane(void)
{
    mo_surf_t s;

    cg_plane(&s, 0, 3.0);  /* PX at x=3 */

    /* x < 3: negative */
    CHECK(cg_eval(&s, 2.0, 0.0, 0.0) < 0.0);
    /* x = 3: zero */
    CHFLT(cg_eval(&s, 3.0, 0.0, 0.0), 0.0, 1e-10);
    /* x > 3: positive */
    CHECK(cg_eval(&s, 4.0, 0.0, 0.0) > 0.0);
    PASS();
}
TH_REG("geom", t_cg_plane)

/* ---- Cell finding ---- */

static void t_cg_find(void)
{
    static mo_prob_t prob;
    memset(&prob, 0, sizeof(prob));

    /* Sphere of radius 5 at origin */
    cg_sph(&prob.surf[0], 0.0, 0.0, 0.0, 5.0);
    prob.n_surf = 1;

    /* Cell 0: inside sphere (sense -1 on surface 0) */
    prob.cell[0].n_half = 1;
    prob.cell[0].surf[0] = 0;
    prob.cell[0].sense[0] = -1;
    prob.cell[0].mat = 0;
    prob.cell[0].rho = 0.04794;

    /* Cell 1: outside sphere (sense +1 on surface 0) */
    prob.cell[1].n_half = 1;
    prob.cell[1].surf[0] = 0;
    prob.cell[1].sense[0] = 1;
    prob.cell[1].mat = -1;  /* void */

    prob.n_cell = 2;

    CHEQ(cg_find(&prob, 0.0, 0.0, 0.0), 0);   /* origin → cell 0 */
    CHEQ(cg_find(&prob, 3.0, 0.0, 0.0), 0);   /* inside → cell 0 */
    CHEQ(cg_find(&prob, 10.0, 0.0, 0.0), 1);  /* outside → cell 1 */
    PASS();
}
TH_REG("geom", t_cg_find)

/* ---- Surface crossing ---- */

static void t_cg_cross(void)
{
    static mo_prob_t prob;
    mo_part_t p;
    int s_hit;
    double d;

    memset(&prob, 0, sizeof(prob));

    /* Same sphere geometry as above */
    cg_sph(&prob.surf[0], 0.0, 0.0, 0.0, 5.0);
    prob.n_surf = 1;

    prob.cell[0].n_half = 1;
    prob.cell[0].surf[0] = 0;
    prob.cell[0].sense[0] = -1;
    prob.cell[0].mat = 0;
    prob.cell[0].rho = 0.04794;

    prob.cell[1].n_half = 1;
    prob.cell[1].surf[0] = 0;
    prob.cell[1].sense[0] = 1;
    prob.cell[1].mat = -1;

    prob.n_cell = 2;

    /* Particle at origin, heading +x */
    memset(&p, 0, sizeof(p));
    p.x = 0.0; p.y = 0.0; p.z = 0.0;
    p.u = 1.0; p.v = 0.0; p.w = 0.0;
    p.cell = 0;
    p.alive = 1;

    /* Should find surface at d=5 */
    d = cg_next(&prob, &p, &s_hit);
    CHFLT(d, 5.0, 1e-6);
    CHEQ(s_hit, 0);

    /* Move to surface and cross */
    tp_move(&p, d);
    int ok = cg_cross(&prob, &p, s_hit);

    /* Should have crossed into void cell and died */
    CHEQ(ok, 0);
    CHEQ(p.alive, 0);
    PASS();
}
TH_REG("geom", t_cg_cross)

/* ---- Cylinder evaluation ---- */

static void t_cg_cyl(void)
{
    mo_surf_t s;
    cg_cyl(&s, 2, 3.0);  /* CZ: cylinder along z, radius 3 */

    /* At (0,0,0): r=0, inside (negative) */
    CHECK(cg_eval(&s, 0.0, 0.0, 0.0) < 0.0);

    /* At (3,0,0): on surface */
    CHFLT(cg_eval(&s, 3.0, 0.0, 0.0), 0.0, 1e-10);

    /* At (4,0,0): outside */
    CHECK(cg_eval(&s, 4.0, 0.0, 0.0) > 0.0);
    PASS();
}
TH_REG("geom", t_cg_cyl)
