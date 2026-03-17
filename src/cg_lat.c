/* cg_lat.c -- Lattice geometry
 *
 * The nuclear engineer's answer to "I don't want to define 289
 * identical fuel pins by hand": fill a grid with repeated
 * universes. Each lattice element maps to a universe, which
 * contains its own cells and surfaces in local coordinates.
 *
 * Without lattice cards, a 17×17 PWR assembly needs ~1,500
 * surfaces and ~1,200 cells. With: ~10 surfaces, ~5 cells,
 * and a grid that says "same thing, 289 times." Copy-paste
 * but with mathematical rigour. Sort of.
 *
 * Supports rectangular and hexagonal lattices, though hex
 * is future work (the geometry transforms make everyone cry).
 *
 * Reference:
 *   Romano, P. K., & Forget, B. (2013). The OpenMC Monte Carlo
 *   particle transport code. Annals of Nuclear Energy, 51,
 *   274-281. Section 2.3: lattice geometry. */

#include "../moa.h"

/* ---- Find lattice element indices ----
 * Given global coordinates (x,y,z), compute which lattice
 * element (i,j,k) the point falls in. For rectangular lattices
 * this is just floor((x - orig) / pitch). For hex, it's floor
 * plus some trigonometry that we'll implement when hexagons
 * stop having six sides (i.e., never, but also next sprint). */

int
cl_find(const mo_prob_t *prob, const mo_lat_t *lat,
        double x, double y, double z, int *li, int *lj, int *lk)
{
    (void)prob;

    if (lat->type == 0) {
        /* Rectangular lattice */
        if (lat->pitch[0] < 1.0e-20) return -1;
        if (lat->pitch[1] < 1.0e-20) return -1;

        *li = (int)floor((x - lat->orig[0]) / lat->pitch[0]);
        *lj = (int)floor((y - lat->orig[1]) / lat->pitch[1]);

        if (lat->nz > 1 && lat->pitch[2] > 1.0e-20)
            *lk = (int)floor((z - lat->orig[2]) / lat->pitch[2]);
        else
            *lk = 0;

        /* Clamp to lattice bounds */
        if (*li < 0 || *li >= lat->nx) return -1;
        if (*lj < 0 || *lj >= lat->ny) return -1;
        if (*lk < 0 || *lk >= lat->nz) return -1;

        int idx = (*li) * lat->ny * lat->nz + (*lj) * lat->nz + (*lk);
        if (KA_CHK(idx, MO_MAX_LELEM)) return -1;

        return lat->fill[idx];
    }

    /* Hex: not yet. Return -1, which means "I don't know where
     * this neutron is" — the geometry equivalent of shrugging. */
    return -1;
}

/* ---- Global to local coordinates ----
 * Translate from global (problem) coordinates to local
 * (universe) coordinates within lattice element (i,j,k).
 * The local origin is at the centre of the element. */

void
cl_locl(const mo_lat_t *lat, int i, int j, int k,
        double x, double y, double z,
        double *lx, double *ly, double *lz)
{
    /* Local = global - element_centre */
    *lx = x - (lat->orig[0] + ((double)i + 0.5) * lat->pitch[0]);
    *ly = y - (lat->orig[1] + ((double)j + 0.5) * lat->pitch[1]);

    if (lat->nz > 1 && lat->pitch[2] > 1.0e-20)
        *lz = z - (lat->orig[2] + ((double)k + 0.5) * lat->pitch[2]);
    else
        *lz = z;
}

/* ---- Local to global coordinates ----
 * Inverse of cl_locl. */

void
cl_glob(const mo_lat_t *lat, int i, int j, int k,
        double lx, double ly, double lz,
        double *x, double *y, double *z)
{
    *x = lx + lat->orig[0] + ((double)i + 0.5) * lat->pitch[0];
    *y = ly + lat->orig[1] + ((double)j + 0.5) * lat->pitch[1];

    if (lat->nz > 1 && lat->pitch[2] > 1.0e-20)
        *z = lz + lat->orig[2] + ((double)k + 0.5) * lat->pitch[2];
    else
        *z = lz;
}

/* ---- Nearest lattice boundary crossing ----
 * For rectangular lattices, find the distance to the nearest
 * lattice cell boundary along the particle's flight path.
 * This is separate from the CSG surface crossing — it's the
 * boundary of the repeated unit, not the surfaces within it.
 *
 * Returns the distance to the nearest lattice boundary.
 * If the particle would cross a lattice boundary before
 * hitting any surface within the cell, we need to transition
 * to the next lattice element. */

double
cl_next(const mo_prob_t *prob, const mo_lat_t *lat,
        const mo_part_t *p, int *s_hit)
{
    double d_min = HUGE_VAL;
    double xlo, xhi, ylo, yhi;
    double d;

    (void)prob;
    (void)s_hit;

    if (lat->type != 0) return HUGE_VAL;

    /* Element boundaries in global coordinates */
    xlo = lat->orig[0] + (double)p->li * lat->pitch[0];
    xhi = xlo + lat->pitch[0];
    ylo = lat->orig[1] + (double)p->lj * lat->pitch[1];
    yhi = ylo + lat->pitch[1];

    /* Distance to each face, same axis-aligned plane math as cg_dist
     * but simpler because these are always axis-aligned planes */
    if (p->u > 1.0e-20) {
        d = (xhi - p->x) / p->u;
        if (d > MO_EPS && d < d_min) d_min = d;
    } else if (p->u < -1.0e-20) {
        d = (xlo - p->x) / p->u;
        if (d > MO_EPS && d < d_min) d_min = d;
    }

    if (p->v > 1.0e-20) {
        d = (yhi - p->y) / p->v;
        if (d > MO_EPS && d < d_min) d_min = d;
    } else if (p->v < -1.0e-20) {
        d = (ylo - p->y) / p->v;
        if (d > MO_EPS && d < d_min) d_min = d;
    }

    /* Z boundaries if 3D lattice */
    if (lat->nz > 1 && lat->pitch[2] > 1.0e-20) {
        double zlo = lat->orig[2] + (double)p->lk * lat->pitch[2];
        double zhi = zlo + lat->pitch[2];

        if (p->w > 1.0e-20) {
            d = (zhi - p->z) / p->w;
            if (d > MO_EPS && d < d_min) d_min = d;
        } else if (p->w < -1.0e-20) {
            d = (zlo - p->z) / p->w;
            if (d > MO_EPS && d < d_min) d_min = d;
        }
    }

    return d_min;
}
