/* Include benchmark-specific header. */
#include "2mm.h"

int NUM_THREADS;

//-------------------------- Timer ---------------------------------------------
double bench_t_start, bench_t_end;
static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void bench_timer_start()
{
    bench_t_start = rtclock ();
}
void bench_timer_stop()
{
    bench_t_end = rtclock ();
}
void bench_timer_print()
{
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}
//------------------------------------------------------------------------------

static
void print_array(int ni, int nl,
                 double D[ ni][nl])
{
    int i, j;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "D");
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
            fprintf (stderr, "%0.2lf ", D[i][j]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "D");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


static
void init_array(int ni, int nj, int nk, int nl,
                double *alpha, double *beta,
                double A [ni] [nk],
                double B [nk] [nj],
                double C [nj] [nl],
                double D [ni] [nl])
{

    int i, j;
    *alpha = 1.5;
    *beta = 1.2;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (double) ((i * j + 1) % ni) / ni;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (double) (i * (j + 1) % nj) / nj;
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = (double) ((i * (j + 3) + 1) % nl) / nl;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (double) (i * (j + 2) % nk) / nk;
}

static
void kernel_2mm(int ni, int nj, int nk, int nl,
                double alpha,
                double beta,
                double tmp[ ni][nj],
                double A[ ni][nk],
                double B[ nk][nj],
                double C[ nj][nl],
                double D[ ni][nl])
{
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
        {
            tmp[i][j] = 0.0;
            for (int k = 0; k < nk; ++k)
                tmp[i][j] += alpha * A[i][k] * B[k][j];
        }
    #pragma omp barrier
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nl; j++)
        {
            D[i][j] *= beta;
            for (int k = 0; k < nj; ++k)
                D[i][j] += tmp[i][k] * C[k][j];
        }
}

int main(int argc, char** argv)
{

    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    double alpha;
    double beta;

    int nums[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    int nis[5] = {16, 40, 180, 800, 1600};
    int njs[5] = {18, 50, 190, 900, 1800};
    int nks[5] = {22, 70, 210, 1100, 2200};
    int nls[5] = {24, 80, 220, 1200, 2400};
    char *names[5] = {"MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"};

    for (int t = 0; t < 5; t++) {
        ni = nis[t];
        nk = nks[t];
        nj = njs[t];
        nl = nls[t];

        printf("----------------NEW DATASET----------------\n");
        printf("ni = %d, nk = %d, nj = %d, nl = %d\n%s_DATASET:\n\n", ni, nk, nj, nl, names[t]);

        double (*tmp)[ni][nj]; tmp = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
        double (*A)[ni][nk]; A = (double(*)[ni][nk])malloc ((ni) * (nk) * sizeof(double));
        double (*B)[nk][nj]; B = (double(*)[nk][nj])malloc ((nk) * (nj) * sizeof(double));
        double (*C)[nj][nl]; C = (double(*)[nj][nl])malloc ((nj) * (nl) * sizeof(double));
        double (*D)[ni][nl]; D = (double(*)[ni][nl])malloc ((ni) * (nl) * sizeof(double));

        init_array (ni, nj, nk, nl, &alpha, &beta,
                    *A,
                    *B,
                    *C,
                    *D);

        for (int i = 0; i < 8; i++) {
            NUM_THREADS = nums[i];

            //bench_timer_start();
            double start = omp_get_wtime();
            kernel_2mm (ni, nj, nk, nl,
                        alpha, beta,
                        *tmp,
                        *A,
                        *B,
                        *C,
                        *D);

            //bench_timer_stop();
            double end = omp_get_wtime();
            printf("Number of threads = %d\n", NUM_THREADS);
            //bench_timer_print();
            printf("Time = %.6f", end - start);
            printf("\n");
        }

        free((void*)tmp);
        free((void*)A);
        free((void*)B);
        free((void*)C);
        free((void*)D);

        printf("\n");
    }

    return 0;
}