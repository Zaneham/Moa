/* io_input.c -- Problem definition parser
 *
 * A simple line-based text format for defining reactor geometry.
 * Not XML, not JSON, not YAML. Just lines. Because nuclear
 * engineers have been using card-image formats since before
 * your parents were born, and old habits die harder than
 * subcritical assemblies.
 *
 * Format:
 *   SURFACE id type params...
 *   CELL id MAT matid sense1 sense2 ...
 *   CELL id VOID sense1 sense2 ...
 *   MATERIAL id ZA fraction [ZA fraction ...]
 *   ENDF ZA path
 *   KCODE npart nbatch nskip
 *
 * Lines starting with # are comments. Blank lines are ignored.
 * Everything is space-delimited. Welcome to 1965. */

#include "../moa.h"

#define IO_MAXLN 512
#define IO_MAXTK 32

/* ---- Tokeniser ----
 * Split a line into whitespace-separated tokens.
 * Returns number of tokens. Modifies the input string
 * (inserts NULs). Not thread-safe. Not sorry. */

static int
io_tok(char *line, char *toks[], int max_toks)
{
    int n = 0;
    char *p = line;

    while (*p && n < max_toks) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0' || *p == '#') break;  /* comment or EOL */

        toks[n++] = p;

        /* Find end of token */
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
        if (*p) *p++ = '\0';
    }
    return n;
}

/* ---- Parse SURFACE card ----
 * SURFACE id SPH x0 y0 z0 r
 * SURFACE id PX d
 * SURFACE id PY d
 * SURFACE id PZ d
 * SURFACE id CX r
 * SURFACE id CY r
 * SURFACE id CZ r */

static int
io_surf(mo_prob_t *prob, char *toks[], int ntok)
{
    int id;
    mo_surf_t *s;

    if (ntok < 3) return -1;
    id = atoi(toks[1]);
    if (id < 0 || id >= MO_MAX_SURF) {
        fprintf(stderr, "moa: surface id %d out of range\n", id);
        return -1;
    }

    /* Extend surface pool to include this id */
    if (id >= prob->n_surf) prob->n_surf = id + 1;
    s = &prob->surf[id];

    if (strcmp(toks[2], "SPH") == 0 && ntok >= 7) {
        cg_sph(s, atof(toks[3]), atof(toks[4]),
               atof(toks[5]), atof(toks[6]));
    } else if (strcmp(toks[2], "PX") == 0 && ntok >= 4) {
        cg_plane(s, 0, atof(toks[3]));
    } else if (strcmp(toks[2], "PY") == 0 && ntok >= 4) {
        cg_plane(s, 1, atof(toks[3]));
    } else if (strcmp(toks[2], "PZ") == 0 && ntok >= 4) {
        cg_plane(s, 2, atof(toks[3]));
    } else if (strcmp(toks[2], "CX") == 0 && ntok >= 4) {
        cg_cyl(s, 0, atof(toks[3]));
    } else if (strcmp(toks[2], "CY") == 0 && ntok >= 4) {
        cg_cyl(s, 1, atof(toks[3]));
    } else if (strcmp(toks[2], "CZ") == 0 && ntok >= 4) {
        cg_cyl(s, 2, atof(toks[3]));
    } else {
        fprintf(stderr, "moa: unknown surface type '%s'\n", toks[2]);
        return -1;
    }
    return 0;
}

/* ---- Parse CELL card ----
 * CELL id MAT matid sense1 sense2 ... [RHO density]
 * CELL id VOID sense1 sense2 ...
 *
 * Senses: +N means positive side of surface N,
 *         -N means negative side of surface N. */

static int
io_cell(mo_prob_t *prob, char *toks[], int ntok)
{
    int id, i, tok_start;
    mo_cell_t *c;

    if (ntok < 3) return -1;
    id = atoi(toks[1]);
    if (id < 0 || id >= MO_MAX_CELL) {
        fprintf(stderr, "moa: cell id %d out of range\n", id);
        return -1;
    }

    if (id >= prob->n_cell) prob->n_cell = id + 1;
    c = &prob->cell[id];
    memset(c, 0, sizeof(*c));
    c->lat_id = -1;

    if (strcmp(toks[2], "VOID") == 0) {
        c->mat = -1;
        c->rho = 0.0;
        tok_start = 3;
    } else if (strcmp(toks[2], "MAT") == 0 && ntok >= 4) {
        c->mat = atoi(toks[3]);
        c->rho = 0.04794;  /* default density, overridden by RHO */
        tok_start = 4;
    } else {
        fprintf(stderr, "moa: cell must have MAT or VOID\n");
        return -1;
    }

    /* Parse halfspace senses and optional RHO */
    for (i = tok_start; i < ntok && c->n_half < MO_MAX_HALF; i++) {
        if (strcmp(toks[i], "RHO") == 0 && i + 1 < ntok) {
            c->rho = atof(toks[i + 1]);
            i++;
            continue;
        }
        if (strcmp(toks[i], "TEMP") == 0 && i + 1 < ntok) {
            c->temp = atof(toks[i + 1]);
            i++;
            continue;
        }

        int sense_val = atoi(toks[i]);
        if (sense_val == 0) continue;

        int si = (sense_val < 0) ? -sense_val : sense_val;
        c->surf[c->n_half]  = si;
        c->sense[c->n_half] = (sense_val < 0) ? -1 : 1;
        c->n_half++;
    }
    return 0;
}

/* ---- Parse MATERIAL card ----
 * MATERIAL id ZA1 frac1 [ZA2 frac2 ...] */

static int
io_mat(mo_prob_t *prob, char *toks[], int ntok)
{
    int id, i;
    mo_mat_t *m;

    if (ntok < 4) return -1;
    id = atoi(toks[1]);
    if (id < 0 || id >= MO_MAX_MAT) {
        fprintf(stderr, "moa: material id %d out of range\n", id);
        return -1;
    }

    if (id >= prob->n_mat) prob->n_mat = id + 1;
    m = &prob->mat[id];
    memset(m, 0, sizeof(*m));

    /* Init SAB links to none */
    {
        int j;
        for (j = 0; j < MO_MAX_MNUC; j++) m->sab_id[j] = -1;
    }

    /* Parse ZA/fraction pairs */
    for (i = 2; i + 1 < ntok && m->n_nuc < MO_MAX_MNUC; i += 2) {
        int za = atoi(toks[i]);
        double frac = atof(toks[i + 1]);

        /* Find nuclide index by ZA match */
        int j, found = -1;
        for (j = 0; j < prob->n_nucl; j++) {
            if (prob->nucl[j].ZA == za) { found = j; break; }
        }

        if (found < 0) {
            /* Nuclide not loaded yet — will be resolved when ENDF card
             * is processed. Store ZA as negative placeholder. */
            found = prob->n_nucl;
            if (found < MO_MAX_NUC) {
                prob->nucl[found].ZA = za;
                prob->n_nucl++;
            } else {
                fprintf(stderr, "moa: too many nuclides\n");
                continue;
            }
        }

        m->nuc_id[m->n_nuc] = found;
        m->frac[m->n_nuc]   = frac;
        m->n_nuc++;
    }
    return 0;
}

/* ---- Parse ENDF card ----
 * ENDF ZA path */

static int
io_endf(mo_prob_t *prob, char *toks[], int ntok)
{
    int za, i;

    if (ntok < 3) return -1;
    za = atoi(toks[1]);

    /* Find nuclide slot with this ZA */
    for (i = 0; i < prob->n_nucl; i++) {
        if (prob->nucl[i].ZA == za) {
            return nd_load(&prob->nucl[i], toks[2], za);
        }
    }

    /* Not referenced by any material yet — load anyway */
    if (prob->n_nucl < MO_MAX_NUC) {
        int idx = prob->n_nucl++;
        return nd_load(&prob->nucl[idx], toks[2], za);
    }

    fprintf(stderr, "moa: too many nuclides for ENDF %d\n", za);
    return -1;
}

/* ---- Parse KCODE card ----
 * KCODE npart nbatch nskip */

static int
io_kcode(mo_prob_t *prob, char *toks[], int ntok)
{
    if (ntok < 4) return -1;
    prob->n_part  = atoi(toks[1]);
    prob->n_batch = atoi(toks[2]);
    prob->n_skip  = atoi(toks[3]);

    if (prob->n_part > MO_MAX_PART) {
        fprintf(stderr, "moa: n_part %d exceeds MO_MAX_PART (%d), clamping\n",
                prob->n_part, MO_MAX_PART);
        prob->n_part = MO_MAX_PART;
    }
    return 0;
}

/* ---- Parse ETALLY card ----
 * ETALLY n_bins emin emax [cell1 cell2 ...]
 * ETALLY CUSTOM n edge1 edge2 ... cell1 cell2 ...
 * Energy-binned flux tally. The physicist's security blanket. */

static int
io_etly(mo_prob_t *prob, char *toks[], int ntok)
{
    int i;

    if (ntok < 4) return -1;

    if (strcmp(toks[1], "CUSTOM") == 0) {
        int n = atoi(toks[2]);
        if (n <= 0 || n > MO_MAX_EBIN) return -1;
        if (ntok < 3 + n + 1) return -1;  /* need n+1 edges */

        prob->etl_flux.n_bin = n;
        prob->etl_flux.active = 1;
        for (i = 0; i <= n && i + 3 < ntok; i++) {
            prob->etl_flux.edge[i] = atof(toks[3 + i]);
        }

        /* Remaining tokens are cell IDs */
        prob->etl_flux.n_reg = 0;
        for (i = 3 + n + 1; i < ntok; i++) {
            if (prob->etl_flux.n_reg < MO_MAX_REG) {
                prob->etl_flux.reg[prob->etl_flux.n_reg++] = atoi(toks[i]);
            }
        }
    } else {
        int n_bins = atoi(toks[1]);
        double emin = atof(toks[2]);
        double emax = atof(toks[3]);

        te_set(&prob->etl_flux, n_bins, emin, emax);

        /* Optional cell list */
        prob->etl_flux.n_reg = 0;
        for (i = 4; i < ntok; i++) {
            if (prob->etl_flux.n_reg < MO_MAX_REG) {
                prob->etl_flux.reg[prob->etl_flux.n_reg++] = atoi(toks[i]);
            }
        }
    }

    /* Default: if no cells specified, tally all cells */
    if (prob->etl_flux.n_reg == 0) {
        for (i = 0; i < MO_MAX_REG; i++) {
            prob->etl_flux.reg[i] = i;
        }
        prob->etl_flux.n_reg = MO_MAX_REG;
    }

    return 0;
}

/* ---- Parse MESH card ----
 * MESH xlo xhi ylo yhi zlo zhi
 * Overlay a mesh tally. Grid resolution is compile-time fixed
 * at 64³. If you want higher resolution, change MO_MESH_N*
 * and bring more RAM. */

static int
io_mesh(mo_prob_t *prob, char *toks[], int ntok)
{
    double lo[3], hi[3];

    if (ntok < 7) return -1;

    lo[0] = atof(toks[1]); hi[0] = atof(toks[2]);
    lo[1] = atof(toks[3]); hi[1] = atof(toks[4]);
    lo[2] = atof(toks[5]); hi[2] = atof(toks[6]);

    /* Mesh is heap-allocated in main.c — set a flag here */
    if (!prob->mesh) {
        prob->mesh = (mo_mtly_t *)calloc(1, sizeof(mo_mtly_t));
        if (!prob->mesh) {
            fprintf(stderr, "moa: failed to allocate mesh tally\n");
            return -1;
        }
    }
    tm_init(prob->mesh, lo, hi);

    return 0;
}

/* ---- Parse FIXED card ----
 * FIXED POINT x y z E n_part n_batch
 * FIXED VOLUME cell E n_part n_batch
 * Presence of FIXED sets mode to fixed-source. */

static int
io_fixd(mo_prob_t *prob, char *toks[], int ntok)
{
    if (ntok < 2) return -1;

    prob->mode = MO_MODE_FIXD;

    if (strcmp(toks[1], "POINT") == 0 && ntok >= 8) {
        prob->src.type   = MO_SRC_PT;
        prob->src.pos[0] = atof(toks[2]);
        prob->src.pos[1] = atof(toks[3]);
        prob->src.pos[2] = atof(toks[4]);
        prob->src.E      = atof(toks[5]);
        prob->n_part     = atoi(toks[6]);
        prob->n_batch    = atoi(toks[7]);
        if (ntok >= 9) prob->n_skip = atoi(toks[8]);
    } else if (strcmp(toks[1], "VOLUME") == 0 && ntok >= 6) {
        prob->src.type = MO_SRC_VOL;
        prob->src.cell = atoi(toks[2]);
        prob->src.E    = atof(toks[3]);
        prob->n_part   = atoi(toks[4]);
        prob->n_batch  = atoi(toks[5]);
        if (ntok >= 7) prob->n_skip = atoi(toks[6]);
    } else {
        fprintf(stderr, "moa: bad FIXED card format\n");
        return -1;
    }

    if (prob->n_part > MO_MAX_PART) prob->n_part = MO_MAX_PART;
    return 0;
}

/* ---- Parse SAB card ----
 * SAB ZA path
 * Load S(α,β) thermal scattering table for a nuclide. */

static int
io_sab(mo_prob_t *prob, char *toks[], int ntok)
{
    int za, i;

    if (ntok < 3) return -1;
    za = atoi(toks[1]);

    if (prob->n_sab >= MO_MAX_SAB) {
        fprintf(stderr, "moa: too many SAB tables\n");
        return -1;
    }

    int idx = prob->n_sab;
    if (sb_load(&prob->sab[idx], toks[2], za) < 0)
        return -1;

    /* Link SAB table to nuclides in materials */
    for (i = 0; i < prob->n_mat; i++) {
        mo_mat_t *m = &prob->mat[i];
        int j;
        for (j = 0; j < m->n_nuc; j++) {
            int nid = m->nuc_id[j];
            if (nid >= 0 && nid < prob->n_nucl
                && prob->nucl[nid].ZA == za) {
                m->sab_id[j] = idx;
            }
        }
    }

    prob->n_sab++;
    return 0;
}

/* ---- Parse LATTICE card ----
 * LATTICE id RECT nx ny nz px py pz ox oy oz
 * Define a rectangular lattice grid. */

static int
io_lat(mo_prob_t *prob, char *toks[], int ntok)
{
    int id;
    mo_lat_t *l;

    if (ntok < 12) return -1;
    id = atoi(toks[1]);
    if (id < 0 || id >= MO_MAX_LAT) return -1;

    if (id >= prob->n_lat) prob->n_lat = id + 1;
    l = &prob->lat[id];
    memset(l, 0, sizeof(*l));

    if (strcmp(toks[2], "RECT") == 0) {
        l->type = 0;
        l->nx = atoi(toks[3]);
        l->ny = atoi(toks[4]);
        l->nz = atoi(toks[5]);
        l->pitch[0] = atof(toks[6]);
        l->pitch[1] = atof(toks[7]);
        l->pitch[2] = atof(toks[8]);
        l->orig[0]  = atof(toks[9]);
        l->orig[1]  = atof(toks[10]);
        l->orig[2]  = atof(toks[11]);
    } else {
        fprintf(stderr, "moa: unknown lattice type '%s'\n", toks[2]);
        return -1;
    }
    return 0;
}

/* ---- Parse FILL card ----
 * FILL lat_id u1 u2 u3 ...
 * Fill lattice elements with universe IDs. */

static int
io_fill(mo_prob_t *prob, char *toks[], int ntok)
{
    int lid, i;
    mo_lat_t *l;

    if (ntok < 3) return -1;
    lid = atoi(toks[1]);
    if (KA_CHK(lid, prob->n_lat)) return -1;

    l = &prob->lat[lid];
    int max_elem = l->nx * l->ny * l->nz;
    if (max_elem > MO_MAX_LELEM) max_elem = MO_MAX_LELEM;

    for (i = 2; i < ntok && (i - 2) < max_elem; i++) {
        l->fill[i - 2] = atoi(toks[i]);
    }
    return 0;
}

/* ---- Parse UNIVERSE card ----
 * UNIVERSE id cell_lo cell_hi
 * Define a universe as a range of cell IDs. */

static int
io_univ(mo_prob_t *prob, char *toks[], int ntok)
{
    int id;

    if (ntok < 4) return -1;
    id = atoi(toks[1]);
    if (id < 0 || id >= MO_MAX_UNIV) return -1;

    if (id >= prob->n_univ) prob->n_univ = id + 1;
    prob->univ[id].cell_lo = atoi(toks[2]);
    prob->univ[id].cell_hi = atoi(toks[3]);

    return 0;
}

/* ---- Load problem from input file ---- */

int
io_load(mo_prob_t *prob, const char *path)
{
    FILE *f;
    char  line[IO_MAXLN];
    char *toks[IO_MAXTK];
    int   ntok, lnum = 0;

    /* Pre-init all cells to void so unused slots (from 1-based IDs)
     * don't accidentally match particles. Cell 0 with n_half=0 and
     * mat=0 is a trap that catches everything and routes to a
     * nonexistent material. Ask me how I know. */
    {
        int i;
        for (i = 0; i < MO_MAX_CELL; i++) {
            prob->cell[i].mat = -1;
            prob->cell[i].lat_id = -1;
        }
    }

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "moa: cannot open input file: %s\n", path);
        return -1;
    }

    while (fgets(line, IO_MAXLN, f)) {
        lnum++;
        ntok = io_tok(line, toks, IO_MAXTK);
        if (ntok == 0) continue;  /* blank or comment */

        int rc = 0;
        if      (strcmp(toks[0], "SURFACE")  == 0) rc = io_surf(prob, toks, ntok);
        else if (strcmp(toks[0], "CELL")     == 0) rc = io_cell(prob, toks, ntok);
        else if (strcmp(toks[0], "MATERIAL") == 0) rc = io_mat(prob, toks, ntok);
        else if (strcmp(toks[0], "ENDF")     == 0) rc = io_endf(prob, toks, ntok);
        else if (strcmp(toks[0], "KCODE")    == 0) rc = io_kcode(prob, toks, ntok);
        else if (strcmp(toks[0], "ETALLY")   == 0) rc = io_etly(prob, toks, ntok);
        else if (strcmp(toks[0], "MESH")     == 0) rc = io_mesh(prob, toks, ntok);
        else if (strcmp(toks[0], "FIXED")    == 0) rc = io_fixd(prob, toks, ntok);
        else if (strcmp(toks[0], "SAB")      == 0) rc = io_sab(prob, toks, ntok);
        else if (strcmp(toks[0], "LATTICE")  == 0) rc = io_lat(prob, toks, ntok);
        else if (strcmp(toks[0], "FILL")     == 0) rc = io_fill(prob, toks, ntok);
        else if (strcmp(toks[0], "UNIVERSE") == 0) rc = io_univ(prob, toks, ntok);
        else if (strcmp(toks[0], "THREADS")  == 0) {
            if (ntok >= 2) prob->n_thrd = atoi(toks[1]);
        }
        else if (strcmp(toks[0], "GPU") == 0) {
            prob->use_gpu = 1;
            if (ntok >= 2) {
                strncpy(prob->hsaco_path, toks[1],
                        sizeof(prob->hsaco_path) - 1);
            } else {
#ifdef MOA_GPU_NV
                strncpy(prob->hsaco_path, "gpu/tp_kern.ptx",
                        sizeof(prob->hsaco_path) - 1);
#else
                strncpy(prob->hsaco_path, "gpu/tp_kern.hsaco",
                        sizeof(prob->hsaco_path) - 1);
#endif
            }
        }
        else {
            fprintf(stderr, "moa: unknown card '%s' at line %d\n",
                    toks[0], lnum);
        }

        if (rc < 0) {
            fprintf(stderr, "moa: error at line %d\n", lnum);
        }
    }

    fclose(f);

    /* Validate */
    if (prob->n_surf == 0) {
        fprintf(stderr, "moa: no surfaces defined\n");
        return -1;
    }
    if (prob->n_cell == 0) {
        fprintf(stderr, "moa: no cells defined\n");
        return -1;
    }
    if (prob->n_part <= 0) {
        prob->n_part  = 1000;
        prob->n_batch = 50;
        prob->n_skip  = 10;
    }
    if (prob->n_thrd <= 0) prob->n_thrd = 1;

    return 0;
}
