#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>
#include <time.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];

    }
}

double run_serial(int m, int n)
{
    double *a, *b, *c0, *c1, *c2, *c3;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c0 = calloc(sizeof(*c0),  m);
    

    for (int i = 0; i < m; i++) {
for (int j = 0; j < n; j++)
a[i * n + j] = i + j;

}
for (int j = 0; j < n; j++)
b[j] = j;

    for (int j = 0; j < n; j++)
        b[j] = j;
    double t = omp_get_wtime();
    matrix_vector_product(a, b, c0, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c0);
    return t;
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel num_threads(4)
    {
        int nthreads = omp_get_num_threads();//кол-во потоков
        int threadid = omp_get_thread_num();//текущий поток
        int items_per_thread = m / nthreads;//сколько строк на поток
        int lb = threadid * items_per_thread;//выбираем первую строку для потока
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        //если текущий поток последний то берем последнюю строку ИНАЧЕ последний входящий в окно
        
        for (int i = lb; i <= ub; i++) {//
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];//

        }

    }
}

double run_parallel(int m, int n)
{
    double *a, *b, *c;
    // Allocate memory for 2-d array a[m, n]
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = calloc(sizeof(*c), m);

    #pragma omp parallel for num_threads(4)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i*n + j] = i + j;
            c[i] = 0.0;
        }
    }
    
    for (int j = 0; j < n; j++)
        b[j] = j;
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime()- t;
    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
    return t;
}

int main(int argc, char **argv)
{
    double a, b;
    int m = 20000;
    int n = m;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    a = run_serial(m, n);
    b = run_parallel(m, n);
    printf("s: %lf\n", a/b);

    return 0;
}