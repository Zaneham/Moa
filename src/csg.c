/* csg.c -- Constructive Solid Geometry engine
 *
 * Ray tracing for people who care about physics, not pixels.
 * We trace neutrons through surfaces defined by quadric equations,
 * which is exactly as glamorous as it sounds.
 *
 * Every surface is stored as a general quadric:
 *   Ax² + By² + Cz² + Dxy + Eyz + Fxz + Gx + Hy + Jz + K = 0
 *
 * Cells are boolean intersections of halfspaces. A halfspace is
 * "the side of a surface where the quadric evaluates negative"
 * (sense = -1) or positive (sense = +1).
 *
 * This is how MCNP does it. We're not going to argue with
 * fifty years of nuclear engineering tradition.
 */

#include "../moa.h"

/* ---- Surface constructors ----
 * Build specific surface types from their natural parameters.
 * The general quadric stores everything, even though a plane
 * only needs two of the ten coefficients. */

void
cg_plane(mo_surf_t *s, int axis, double d)
{
    memset(s, 0, sizeof(*s));
    /* Plane: Gx + Hy + Jz + K = 0, so e.g. PX at x=d: G=1, K=-d */
    switch (axis) {
    case 0: s->type = MO_PX; s->G = 1.0; s->K = -d; break;
    case 1: s->type = MO_PY; s->H = 1.0; s->K = -d; break;
    case 2: s->type = MO_PZ; s->J = 1.0; s->K = -d; break;
    }
}

void
cg_cyl(mo_surf_t *s, int axis, double r)
{
    memset(s, 0, sizeof(*s));
    /* Cylinder along axis: two quadratic terms = 1, K = -r² */
    switch (axis) {
    case 0: s->type = MO_CX; s->B = 1.0; s->C = 1.0; break;
    case 1: s->type = MO_CY; s->A = 1.0; s->C = 1.0; break;
    case 2: s->type = MO_CZ; s->A = 1.0; s->B = 1.0; break;
    }
    s->K = -(r * r);
}

void
cg_sph(mo_surf_t *s, double x0, double y0, double z0, double r)
{
    memset(s, 0, sizeof(*s));
    s->type = MO_SPH;
    /* (x-x0)² + (y-y0)² + (z-z0)² - r² = 0
     * Expand: x² + y² + z² - 2x0·x - 2y0·y - 2z0·z + (x0²+y0²+z0²-r²) = 0 */
    s->A = 1.0;
    s->B = 1.0;
    s->C = 1.0;
    s->G = -2.0 * x0;
    s->H = -2.0 * y0;
    s->J = -2.0 * z0;
    s->K = x0 * x0 + y0 * y0 + z0 * z0 - r * r;
}

/* ---- Evaluate surface equation at a point ----
 * Returns the signed value of the quadric. Negative = inside
 * for the convention where sense=-1 means "the negative side." */

double
cg_eval(const mo_surf_t *s, double x, double y, double z)
{
    return s->A * x * x + s->B * y * y + s->C * z * z
         + s->D * x * y + s->E * y * z + s->F * x * z
         + s->G * x     + s->H * y     + s->J * z
         + s->K;
}

/* ---- Ray-surface intersection distance ----
 * Given a ray (x,y,z) + t·(u,v,w), find the smallest positive t
 * where the ray intersects the surface.
 *
 * For a general quadric, this gives a quadratic in t:
 *   at² + bt + c = 0
 * where:
 *   a = A·u² + B·v² + C·w² + D·u·v + E·v·w + F·u·w
 *   b = 2(A·x·u + B·y·v + C·z·w) + D(x·v+y·u) + E(y·w+z·v)
 *     + F(x·w+z·u) + G·u + H·v + J·w
 *   c = eval(x,y,z)
 *
 * Returns HUGE_VAL if no intersection (ray parallel, or only
 * intersects behind us). */

double
cg_dist(const mo_surf_t *s, double x, double y, double z,
        double u, double v, double w)
{
    double a, b, c, disc, sq, t1, t2;

    a = s->A * u * u + s->B * v * v + s->C * w * w
      + s->D * u * v + s->E * v * w + s->F * u * w;

    b = 2.0 * (s->A * x * u + s->B * y * v + s->C * z * w)
      + s->D * (x * v + y * u) + s->E * (y * w + z * v)
      + s->F * (x * w + z * u)
      + s->G * u + s->H * v + s->J * w;

    c = cg_eval(s, x, y, z);

    /* Linear case: surface is a plane (a ≈ 0) */
    if (fabs(a) < 1.0e-20) {
        if (fabs(b) < 1.0e-20) return HUGE_VAL;  /* parallel */
        t1 = -c / b;
        return (t1 > MO_EPS) ? t1 : HUGE_VAL;
    }

    /* Quadratic case */
    disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return HUGE_VAL;  /* miss. better luck next neutron */

    sq = sqrt(disc);
    t1 = (-b - sq) / (2.0 * a);
    t2 = (-b + sq) / (2.0 * a);

    /* Take smallest positive root above epsilon.
     * Epsilon nudge prevents re-intersection with the surface
     * we just crossed, which would be embarrassing. */
    if (t1 > MO_EPS) return t1;
    if (t2 > MO_EPS) return t2;
    return HUGE_VAL;
}

/* ---- Find cell containing a point ----
 * Test every cell's halfspace list. A point is in a cell if
 * it satisfies ALL halfspace constraints. O(n_cell * n_half)
 * which is fine for < 100 cells and hilarious for 100,000. */

int
cg_find(const mo_prob_t *prob, double x, double y, double z)
{
    int i, j;

    for (i = 0; i < prob->n_cell; i++) {
        const mo_cell_t *c = &prob->cell[i];
        int inside = 1;

        /* Skip uninitialised slots (no halfspaces defined) */
        if (c->n_half == 0) continue;

        for (j = 0; j < c->n_half; j++) {
            int si = c->surf[j];
            if (KA_CHK(si, prob->n_surf)) { inside = 0; break; }

            double val = cg_eval(&prob->surf[si], x, y, z);
            /* sense = -1: want negative side. sense = +1: want positive */
            if (c->sense[j] < 0 && val > 0.0) { inside = 0; break; }
            if (c->sense[j] > 0 && val < 0.0) { inside = 0; break; }
        }
        if (inside) return i;
    }
    return -1;  /* void / outside all cells. the neutron has left the chat */
}

/* ---- Find nearest surface crossing ----
 * Cast the ray from the particle's position/direction, check
 * distance to every surface of the current cell. Return the
 * shortest positive distance and the surface index.
 *
 * We only check surfaces of the current cell because a well-defined
 * geometry guarantees the nearest crossing is one of them. */

double
cg_next(const mo_prob_t *prob, const mo_part_t *p, int *s_hit)
{
    double d_min = HUGE_VAL;
    const mo_cell_t *c;
    int j;

    *s_hit = -1;

    if (KA_CHK(p->cell, prob->n_cell)) return HUGE_VAL;
    c = &prob->cell[p->cell];

    for (j = 0; j < c->n_half; j++) {
        int si = c->surf[j];
        if (KA_CHK(si, prob->n_surf)) continue;

        double d = cg_dist(&prob->surf[si], p->x, p->y, p->z,
                           p->u, p->v, p->w);
        if (d < d_min) {
            d_min = d;
            *s_hit = si;
        }
    }
    return d_min;
}

/* ---- Cross a surface ----
 * Move the particle epsilon past the surface, then find
 * which cell it's now in. If no cell claims it, the
 * neutron has leaked out of the geometry.
 *
 * Returns: 1 if particle is in a valid cell, 0 if leaked. */

int
cg_cross(const mo_prob_t *prob, mo_part_t *p, int s_hit)
{
    (void)s_hit;  /* we don't actually need the surface index for now */

    /* The particle has already been moved to the surface.
     * Nudge it epsilon further along its direction to get
     * unambiguously across. */
    p->x += MO_EPS * p->u;
    p->y += MO_EPS * p->v;
    p->z += MO_EPS * p->w;

    int new_cell = cg_find(prob, p->x, p->y, p->z);
    if (new_cell < 0) {
        /* Leaked. The neutron has escaped the geometry.
         * In a real reactor this would be bad. In a simulation
         * it's just a tally contribution we didn't get. */
        p->alive = 0;
        return 0;
    }

    p->cell = new_cell;

    /* If new cell is void (mat < 0), particle is lost */
    if (prob->cell[new_cell].mat < 0) {
        p->alive = 0;
        return 0;
    }

    return 1;
}
