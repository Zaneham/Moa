/* tl_mesh.c -- Mesh tallies
 *
 * Overlay a Cartesian grid on the geometry and score track-length
 * flux in each voxel. The result is a 3D flux map that makes for
 * lovely heatmaps and occasionally useful physics.
 *
 * 64³ = 262,144 voxels × 3 arrays × 8 bytes ≈ 6 MB. Heap-allocated
 * because mo_prob_t is already a hefty creature and doesn't need
 * another 6 MB of existential baggage.
 *
 * Output is gnuplot/matplotlib-ready text: x y z flux sigma.
 * Because if your data can't be plotted in five lines of Python,
 * does it even exist?
 *
 * Reference:
 *   Lux, I., & Koblinger, L. (1991). Monte Carlo particle
 *   transport methods. CRC Press. Chapter 7. */

#include "../moa.h"

/* ---- Init ---- */

void
tm_init(mo_mtly_t *t, const double lo[3], const double hi[3])
{
    memset(t, 0, sizeof(*t));

    t->lo[0] = lo[0]; t->lo[1] = lo[1]; t->lo[2] = lo[2];
    t->hi[0] = hi[0]; t->hi[1] = hi[1]; t->hi[2] = hi[2];
    t->nx = MO_MESH_NX;
    t->ny = MO_MESH_NY;
    t->nz = MO_MESH_NZ;
    t->active = 1;
}

/* ---- Score a track segment ----
 * Find the voxel containing (x,y,z), accumulate wt*dist.
 * We score at the midpoint of each track segment, which is
 * adequate when the mesh is coarser than the mean free path.
 * For fine meshes you'd need to clip tracks to voxel boundaries,
 * but that's a problem for someone with more ambition and
 * fewer deadlines. */

void
tm_scor(mo_mtly_t *t, double x, double y, double z,
        double wt, double dist)
{
    int ix, iy, iz, idx;
    double dx, dy, dz;

    if (!t->active) return;

    dx = t->hi[0] - t->lo[0];
    dy = t->hi[1] - t->lo[1];
    dz = t->hi[2] - t->lo[2];

    if (dx < 1.0e-30 || dy < 1.0e-30 || dz < 1.0e-30) return;

    ix = (int)((x - t->lo[0]) / dx * (double)t->nx);
    iy = (int)((y - t->lo[1]) / dy * (double)t->ny);
    iz = (int)((z - t->lo[2]) / dz * (double)t->nz);

    /* Outside mesh — quietly ignore. The neutron has wandered
     * off the map, which is its prerogative. */
    if (ix < 0 || ix >= t->nx) return;
    if (iy < 0 || iy >= t->ny) return;
    if (iz < 0 || iz >= t->nz) return;

    idx = ix * t->ny * t->nz + iy * t->nz + iz;
    if (KA_CHK(idx, MO_MESH_BIN)) return;

    t->bat[idx] += wt * dist;
}

/* ---- Normalise batch ---- */

void
tm_norm(mo_mtly_t *t, int n_part)
{
    int i;
    double inv;

    if (!t->active || n_part <= 0) return;
    inv = 1.0 / (double)n_part;

    for (i = 0; i < MO_MESH_BIN; i++) {
        double val = t->bat[i] * inv;
        t->sum[i] += val;
        t->sq[i]  += val * val;
        t->bat[i]  = 0.0;
    }
    t->n_bat++;
}

/* ---- Statistics for one voxel ---- */

void
tm_stat(const mo_mtly_t *t, int ix, int iy, int iz, int n_bat,
        double *mean, double *sdev)
{
    int idx;

    *mean = 0.0;
    *sdev = 0.0;

    if (ix < 0 || ix >= t->nx) return;
    if (iy < 0 || iy >= t->ny) return;
    if (iz < 0 || iz >= t->nz) return;

    idx = ix * t->ny * t->nz + iy * t->nz + iz;
    if (KA_CHK(idx, MO_MESH_BIN)) return;
    if (n_bat <= 0) return;

    *mean = t->sum[idx] / (double)n_bat;

    if (n_bat > 1) {
        double var = (t->sq[idx] / (double)n_bat
                    - (*mean) * (*mean))
                   / (double)(n_bat - 1);
        *sdev = (var > 0.0) ? sqrt(var) : 0.0;
    }
}

/* ---- Dump to text file ----
 * Format: x y z flux sigma
 * One line per non-empty voxel. Coordinates are voxel centres.
 * Pipe through `gnuplot -e "splot 'mesh.dat' using 1:2:3:4 with points palette"`
 * and pretend you knew what you were doing all along. */

void
tm_dump(const mo_mtly_t *t, int n_bat, const char *path)
{
    FILE *f;
    int ix, iy, iz;
    double dx, dy, dz;

    if (!t->active || n_bat <= 0) return;

    f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "moa: cannot open mesh dump file: %s\n", path);
        return;
    }

    dx = (t->hi[0] - t->lo[0]) / (double)t->nx;
    dy = (t->hi[1] - t->lo[1]) / (double)t->ny;
    dz = (t->hi[2] - t->lo[2]) / (double)t->nz;

    fprintf(f, "# x y z flux sigma\n");

    for (ix = 0; ix < t->nx; ix++) {
        for (iy = 0; iy < t->ny; iy++) {
            for (iz = 0; iz < t->nz; iz++) {
                double mean, sdev;
                tm_stat(t, ix, iy, iz, n_bat, &mean, &sdev);
                if (mean < 1.0e-30) continue;

                double cx = t->lo[0] + ((double)ix + 0.5) * dx;
                double cy = t->lo[1] + ((double)iy + 0.5) * dy;
                double cz = t->lo[2] + ((double)iz + 0.5) * dz;

                fprintf(f, "%10.4f %10.4f %10.4f %14.6e %12.4e\n",
                        cx, cy, cz, mean, sdev);
            }
        }
    }

    fclose(f);
    printf("moa: mesh tally dumped to %s\n", path);
}
