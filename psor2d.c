/*
 * AMS 530 Project 3, Problem 3.5
 * Pricing of 2-Asset American Options by Projected SOR
 *
 * Solves the two-asset Black-Scholes PDE for uncorrelated assets
 *
 *   V_t + 0.5*s1^2*S1^2*V_S1S1 + 0.5*s2^2*S2^2*V_S2S2
 *       + r*S1*V_S1 + r*S2*V_S2 - r*V = 0
 *
 * backward in time from the terminal payoff.  Writing tau = T - t and
 * discretising in tau with backward Euler gives one sparse linear system
 * per time step,
 *
 *   (I - dtau*L) V^{n+1} = V^n,
 *
 * whose matrix carries exactly the five-point stencil structure of the
 * Poisson problem in 3.4.  Each system is solved by projected SOR with
 * red-black ordering: after every update the early-exercise constraint
 *
 *   V_ij <- max(V_ij, Phi_ij)
 *
 * is imposed, which is what turns the linear solve into the linear
 * complementarity problem an American option satisfies.
 *
 * Payoff:  Phi(S1,S2) = max(K1-S1,0) + max(K2-S2,0)
 *
 * This payoff is separable, so the EUROPEAN price is exactly the sum of
 * two one-dimensional Black-Scholes puts.  Running with -e (projection
 * disabled) therefore has a closed-form answer to check against, which
 * is the validation the assignment asks for.  Puts are used rather than
 * calls because an American call on a non-dividend-paying asset never
 * exercises early, so the projection would never bind.
 *
 * Parallelism: 2-D block decomposition over a Cartesian communicator,
 * one-cell halo exchange, red-black sweeps so that every point of a
 * colour is independent.  The red-black ordering is global, so the
 * iterates are independent of P and the iteration counts printed by
 * different core counts should agree.
 *
 * Build:  mpicc -O3 -march=native -std=c99 -o psor2d psor2d.c -lm
 * Run:    mpirun -np P ./psor2d -M 256 -n 50 [-e] [-v]
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Strict ISO C does not expose these, so define them if absent. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

/* ------------------------------------------------------------------ */
/* Problem parameters                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    int    M;        /* interior points per dimension                   */
    int    nt;       /* time steps                                      */
    int    american; /* 1 = project (American), 0 = European            */
    int    verbose;
    double T, K1, K2, r, sig1, sig2, Smax1, Smax2;
    double omega, tol;
    int    maxit;
} Params;

static void defaults(Params *p)
{
    p->M        = 256;
    p->nt       = 50;
    p->american = 1;
    p->verbose  = 0;
    p->T        = 1.0;
    p->K1       = 100.0;
    p->K2       = 100.0;
    p->r        = 0.05;
    p->sig1     = 0.30;
    p->sig2     = 0.25;
    p->Smax1    = 300.0;
    p->Smax2    = 300.0;
    p->omega    = -1.0;      /* negative => use theoretical optimum     */
    p->tol      = 1.0e-7;
    p->maxit    = 20000;
}

/* ------------------------------------------------------------------ */
/* Closed-form European put, used for boundary data and verification    */
/* ------------------------------------------------------------------ */

static double ncdf(double x)
{
    return 0.5 * erfc(-x * M_SQRT1_2);
}

static double bs_put(double S, double K, double r, double sig, double tau)
{
    double d1, d2, sq;

    if (tau <= 0.0) return (K - S > 0.0) ? K - S : 0.0;
    if (S <= 0.0)   return K * exp(-r * tau);

    sq = sig * sqrt(tau);
    d1 = (log(S / K) + (r + 0.5 * sig * sig) * tau) / sq;
    d2 = d1 - sq;
    return K * exp(-r * tau) * ncdf(-d2) - S * ncdf(-d1);
}

/* Terminal payoff at a grid point. */
static double payoff_at(const Params *p, double S1, double S2)
{
    double a = p->K1 - S1, b = p->K2 - S2;
    if (a < 0.0) a = 0.0;
    if (b < 0.0) b = 0.0;
    return a + b;
}

/*
 * Dirichlet data on the outer boundary.  For the European problem the
 * separable payoff makes the sum of two closed-form puts exact, so the
 * boundary is exact and the interior error is pure discretisation error.
 * For the American problem the same profile is taken, then projected
 * against the payoff so that the deep-in-the-money edge S->0 carries the
 * immediate-exercise value K rather than the discounted K*exp(-r*tau).
 */
static double boundary_at(const Params *p, double S1, double S2, double tau)
{
    double euro = bs_put(S1, p->K1, p->r, p->sig1, tau)
                + bs_put(S2, p->K2, p->r, p->sig2, tau);
    if (p->american) {
        double ex = payoff_at(p, S1, S2);
        return (euro > ex) ? euro : ex;
    }
    return euro;
}

/* ------------------------------------------------------------------ */
/* Decomposition bookkeeping                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    MPI_Comm comm;
    int rank, nproc;
    int dims[2], coords[2];
    int north, south, east, west;   /* neighbour ranks, MPI_PROC_NULL at edge */
    int nx, ny;                     /* local interior extent                  */
    int i0, j0;                     /* global index of local (1,1)            */
    int ld;                         /* leading dimension = ny + 2             */
} Grid;

/* Split n items over k parts, part q gets base or base+1. */
static void split(int n, int k, int q, int *cnt, int *off)
{
    int base = n / k, rem = n % k;
    *cnt = base + (q < rem ? 1 : 0);
    *off = q * base + (q < rem ? q : rem);
}

static void grid_setup(Grid *g, int M)
{
    int periods[2] = {0, 0};
    int c, o;

    MPI_Comm_size(MPI_COMM_WORLD, &g->nproc);

    g->dims[0] = g->dims[1] = 0;
    MPI_Dims_create(g->nproc, 2, g->dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, g->dims, periods, 0, &g->comm);
    MPI_Comm_rank(g->comm, &g->rank);
    MPI_Cart_coords(g->comm, g->rank, 2, g->coords);

    /* dimension 0 is the S1 (i) direction, dimension 1 is S2 (j) */
    MPI_Cart_shift(g->comm, 0, 1, &g->west,  &g->east);
    MPI_Cart_shift(g->comm, 1, 1, &g->south, &g->north);

    split(M, g->dims[0], g->coords[0], &c, &o); g->nx = c; g->i0 = o + 1;
    split(M, g->dims[1], g->coords[1], &c, &o); g->ny = c; g->j0 = o + 1;

    g->ld = g->ny + 2;
}

#define IDX(g, i, j) ((size_t)(i) * (size_t)(g)->ld + (size_t)(j))

/* ------------------------------------------------------------------ */
/* Halo exchange                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    double *sl, *sr, *rl, *rr;   /* buffers for the strided j-direction */
} Halo;

static double comm_time = 0.0;

static void halo_exchange(const Grid *g, double *V, Halo *h)
{
    MPI_Request req[8];
    int nreq = 0, i;
    double t0 = MPI_Wtime();

    /* i-direction: rows are contiguous, send rows 1 and nx */
    MPI_Irecv(&V[IDX(g, 0,           0)], g->ld, MPI_DOUBLE,
              g->west, 100, g->comm, &req[nreq++]);
    MPI_Irecv(&V[IDX(g, g->nx + 1,   0)], g->ld, MPI_DOUBLE,
              g->east, 101, g->comm, &req[nreq++]);
    MPI_Isend(&V[IDX(g, 1,           0)], g->ld, MPI_DOUBLE,
              g->west, 101, g->comm, &req[nreq++]);
    MPI_Isend(&V[IDX(g, g->nx,       0)], g->ld, MPI_DOUBLE,
              g->east, 100, g->comm, &req[nreq++]);

    /* j-direction: columns are strided, pack first */
    for (i = 0; i <= g->nx + 1; i++) {
        h->sl[i] = V[IDX(g, i, 1)];
        h->sr[i] = V[IDX(g, i, g->ny)];
    }
    MPI_Irecv(h->rl, g->nx + 2, MPI_DOUBLE, g->south, 102, g->comm, &req[nreq++]);
    MPI_Irecv(h->rr, g->nx + 2, MPI_DOUBLE, g->north, 103, g->comm, &req[nreq++]);
    MPI_Isend(h->sl, g->nx + 2, MPI_DOUBLE, g->south, 103, g->comm, &req[nreq++]);
    MPI_Isend(h->sr, g->nx + 2, MPI_DOUBLE, g->north, 102, g->comm, &req[nreq++]);

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    if (g->south != MPI_PROC_NULL)
        for (i = 0; i <= g->nx + 1; i++) V[IDX(g, i, 0)] = h->rl[i];
    if (g->north != MPI_PROC_NULL)
        for (i = 0; i <= g->nx + 1; i++) V[IDX(g, i, g->ny + 1)] = h->rr[i];

    comm_time += MPI_Wtime() - t0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    Params p;
    Grid   g;
    Halo   h;
    double *V, *rhs, *Phi, *aw, *ae, *as, *an, *dg;
    double h1, h2, dtau, t_start, t_end, t_tot, t_comm;
    double probe_local, probe, l2_local, l2, err_local, errmax;
    double probe_euro_local = 0.0, probe_s1 = 0.0, probe_s2 = 0.0;
    double eloc[3] = {0.0, 0.0, 0.0}, emax3[3];
    long   total_iters = 0;
    int    n, i, j, it, color, k;
    size_t sz;

    MPI_Init(&argc, &argv);
    defaults(&p);

    for (k = 1; k < argc; k++) {
        if      (!strcmp(argv[k], "-M") && k + 1 < argc) p.M     = atoi(argv[++k]);
        else if (!strcmp(argv[k], "-n") && k + 1 < argc) p.nt    = atoi(argv[++k]);
        else if (!strcmp(argv[k], "-w") && k + 1 < argc) p.omega = atof(argv[++k]);
        else if (!strcmp(argv[k], "-t") && k + 1 < argc) p.tol   = atof(argv[++k]);
        else if (!strcmp(argv[k], "-e")) p.american = 0;
        else if (!strcmp(argv[k], "-v")) p.verbose  = 1;
    }

    grid_setup(&g, p.M);

    h1   = p.Smax1 / (p.M + 1);
    h2   = p.Smax2 / (p.M + 1);
    dtau = p.T / p.nt;

    /* Theoretical optimum for a five-point stencil on an M x M mesh. */
    if (p.omega <= 0.0)
        p.omega = 2.0 / (1.0 + sin(M_PI / (p.M + 1)));

    sz = (size_t)(g.nx + 2) * (size_t)(g.ny + 2);
    V   = malloc(sz * sizeof(double));
    rhs = malloc(sz * sizeof(double));
    Phi = malloc(sz * sizeof(double));
    aw  = malloc((g.nx + 2) * sizeof(double));
    ae  = malloc((g.nx + 2) * sizeof(double));
    as  = malloc((g.ny + 2) * sizeof(double));
    an  = malloc((g.ny + 2) * sizeof(double));
    dg  = malloc(sz * sizeof(double));
    h.sl = malloc((g.nx + 2) * sizeof(double));
    h.sr = malloc((g.nx + 2) * sizeof(double));
    h.rl = malloc((g.nx + 2) * sizeof(double));
    h.rr = malloc((g.nx + 2) * sizeof(double));
    if (!V || !rhs || !Phi || !aw || !ae || !as || !an || !dg ||
        !h.sl || !h.sr || !h.rl || !h.rr) {
        fprintf(stderr, "rank %d: allocation failed\n", g.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Stencil coefficients depend only on the row (i) or column (j). */
    for (i = 0; i <= g.nx + 1; i++) {
        double S1 = (g.i0 + i - 1) * h1;
        double a  = 0.5 * p.sig1 * p.sig1 * S1 * S1 / (h1 * h1);
        double b  = p.r * S1 / (2.0 * h1);
        aw[i] = dtau * (a - b);
        ae[i] = dtau * (a + b);
    }
    for (j = 0; j <= g.ny + 1; j++) {
        double S2 = (g.j0 + j - 1) * h2;
        double c  = 0.5 * p.sig2 * p.sig2 * S2 * S2 / (h2 * h2);
        double d  = p.r * S2 / (2.0 * h2);
        as[j] = dtau * (c - d);
        an[j] = dtau * (c + d);
    }
    for (i = 1; i <= g.nx; i++)
        for (j = 1; j <= g.ny; j++)
            dg[IDX(&g, i, j)] = 1.0 + 0.5 * (aw[i] + ae[i] + as[j] + an[j]) * 2.0
                              + dtau * p.r;

    /* Terminal condition: V = payoff at tau = 0. */
    for (i = 0; i <= g.nx + 1; i++) {
        double S1 = (g.i0 + i - 1) * h1;
        for (j = 0; j <= g.ny + 1; j++) {
            double S2 = (g.j0 + j - 1) * h2;
            Phi[IDX(&g, i, j)] = payoff_at(&p, S1, S2);
            V[IDX(&g, i, j)]   = Phi[IDX(&g, i, j)];
        }
    }

    MPI_Barrier(g.comm);
    comm_time = 0.0;
    t_start   = MPI_Wtime();

    /* ---------------- time marching ---------------- */
    for (n = 1; n <= p.nt; n++) {
        double tau = n * dtau;

        memcpy(rhs, V, sz * sizeof(double));

        /* Impose Dirichlet data on the physical boundary. */
        for (i = 0; i <= g.nx + 1; i++) {
            int gi = g.i0 + i - 1;
            double S1 = gi * h1;
            for (j = 0; j <= g.ny + 1; j++) {
                int gj = g.j0 + j - 1;
                if (gi == 0 || gi == p.M + 1 || gj == 0 || gj == p.M + 1)
                    V[IDX(&g, i, j)] = boundary_at(&p, S1, gj * h2, tau);
            }
        }

        for (it = 0; it < p.maxit; it++) {
            double dmax_local = 0.0, dmax;

            for (color = 0; color < 2; color++) {
                halo_exchange(&g, V, &h);

                for (i = 1; i <= g.nx; i++) {
                    int gi = g.i0 + i - 1;
                    if (gi == 0 || gi == p.M + 1) continue;
                    for (j = 1; j <= g.ny; j++) {
                        int gj = g.j0 + j - 1;
                        size_t q;
                        double gs, vn, dv;

                        if (gj == 0 || gj == p.M + 1) continue;
                        if (((gi + gj) & 1) != color) continue;

                        q  = IDX(&g, i, j);
                        gs = (rhs[q]
                              + aw[i] * V[q - g.ld]
                              + ae[i] * V[q + g.ld]
                              + as[j] * V[q - 1]
                              + an[j] * V[q + 1]) / dg[q];

                        vn = V[q] + p.omega * (gs - V[q]);
                        if (p.american && vn < Phi[q]) vn = Phi[q];

                        dv = fabs(vn - V[q]);
                        if (dv > dmax_local) dmax_local = dv;
                        V[q] = vn;
                    }
                }
            }

            {
                double tc = MPI_Wtime();
                MPI_Allreduce(&dmax_local, &dmax, 1, MPI_DOUBLE, MPI_MAX, g.comm);
                comm_time += MPI_Wtime() - tc;
            }

            total_iters++;
            if (dmax < p.tol) break;
        }

        if (g.rank == 0 && p.verbose)
            printf("  step %3d  tau=%.4f  iters=%d\n", n, tau, it + 1);
    }

    MPI_Barrier(g.comm);
    t_end = MPI_Wtime();

    /* ---------------- verification ---------------- */

    /*
     * The probe is the grid node closest to (K1,K2).  Its exact European
     * value is evaluated at that node, not at the strike, so the two
     * numbers compare like with like.  For an American run the probe
     * should exceed it: that gap is the early-exercise premium.
     */
    l2_local = 0.0; err_local = 0.0; probe_local = 0.0; probe_euro_local = 0.0;
    {
        int ip = (int)floor(p.K1 / h1 + 0.5);
        int jp = (int)floor(p.K2 / h2 + 0.5);
        if (ip < 1)    ip = 1;
        if (ip > p.M)  ip = p.M;
        if (jp < 1)    jp = 1;
        if (jp > p.M)  jp = p.M;

        for (i = 1; i <= g.nx; i++) {
            int gi = g.i0 + i - 1;
            double S1 = gi * h1;
            for (j = 1; j <= g.ny; j++) {
                int gj = g.j0 + j - 1;
                double S2 = gj * h2;
                double v  = V[IDX(&g, i, j)];
                double e;

                l2_local += v * v;

                e = fabs(v - (bs_put(S1, p.K1, p.r, p.sig1, p.T)
                            + bs_put(S2, p.K2, p.r, p.sig2, p.T)));
                if (e > err_local) { err_local = e; eloc[1] = S1; eloc[2] = S2; }

                if (gi == ip && gj == jp) {
                    probe_local      = v;
                    probe_euro_local = bs_put(S1, p.K1, p.r, p.sig1, p.T)
                                     + bs_put(S2, p.K2, p.r, p.sig2, p.T);
                    probe_s1 = S1;
                    probe_s2 = S2;
                }
            }
        }
    }

    {   /* locate the worst point: MAXLOC on the rank, then broadcast it */
        struct { double val; int rank; } ein, eout;
        ein.val = err_local; ein.rank = g.rank;
        MPI_Allreduce(&ein, &eout, 1, MPI_DOUBLE_INT, MPI_MAXLOC, g.comm);
        eloc[0] = err_local; eloc[1] = eloc[1]; eloc[2] = eloc[2];
        MPI_Bcast(eloc, 3, MPI_DOUBLE, eout.rank, g.comm);
        (void)emax3;
    }
    MPI_Reduce(&l2_local,         &l2,     1, MPI_DOUBLE, MPI_SUM, 0, g.comm);
    MPI_Reduce(&err_local,        &errmax, 1, MPI_DOUBLE, MPI_MAX, 0, g.comm);
    MPI_Allreduce(MPI_IN_PLACE, &probe_local,      1, MPI_DOUBLE, MPI_MAX, g.comm);
    MPI_Allreduce(MPI_IN_PLACE, &probe_euro_local, 1, MPI_DOUBLE, MPI_MAX, g.comm);
    MPI_Allreduce(MPI_IN_PLACE, &probe_s1,         1, MPI_DOUBLE, MPI_MAX, g.comm);
    MPI_Allreduce(MPI_IN_PLACE, &probe_s2,         1, MPI_DOUBLE, MPI_MAX, g.comm);
    probe = probe_local;

    /* Slowest rank's wall time, with its own communication share. */
    {
        struct { double val; int rank; } in, out;
        double loc[2];
        in.val = t_end - t_start; in.rank = g.rank;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, g.comm);
        loc[0] = t_end - t_start;
        loc[1] = comm_time;
        MPI_Bcast(loc, 2, MPI_DOUBLE, out.rank, g.comm);
        t_tot  = loc[0];
        t_comm = loc[1];
    }

    if (g.rank == 0) {
        printf("RESULT style=%s P=%d dims=%dx%d M=%d nt=%d omega=%.6f "
               "time=%.6f comm=%.6f iters=%ld l2=%.10e "
               "probeS=(%.4f,%.4f) probe=%.10f probe_euro_exact=%.10f "
               "premium=%.10f maxerr_vs_euro=%.6e errat=(%.3f,%.3f)\n",
               p.american ? "american" : "european",
               g.nproc, g.dims[0], g.dims[1], p.M, p.nt, p.omega,
               t_tot, t_comm, total_iters, sqrt(l2),
               probe_s1, probe_s2, probe, probe_euro_local,
               probe - probe_euro_local, errmax, eloc[1], eloc[2]);
        fflush(stdout);
    }

    free(V); free(rhs); free(Phi); free(aw); free(ae); free(as); free(an);
    free(dg); free(h.sl); free(h.sr); free(h.rl); free(h.rr);
    MPI_Comm_free(&g.comm);
    MPI_Finalize();
    return 0;
}
