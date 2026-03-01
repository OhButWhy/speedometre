#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <vector>
using namespace std;


void easy_iterational_method(int m, int p) {
    const double EPS   = 1e-5;
    

    // заполняем матрицу A (построчно в одном массиве)
    vector<double> A(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A[i * m + j] = (i == j) ? 2.0 : 1.0;

    // вектор правых частей b[i] = m+1
    vector<double> b(m, m + 1.0);

    // начальное приближение x = 0
    vector<double> x(m, 0.0);
    vector<double> x_new(m, 0.0);

    omp_set_num_threads(p);

    int iter = 0;
    double diff = 0.0;

    while (sqrt(diff) > EPS) {
        diff = 0.0;

        // одна итерация Якоби
        #pragma omp parallel for schedule(static) reduction(+:diff)
        for (int i = 0; i < m; i++) {
            double sigma = 0.0;
            for (int j = 0; j < m; j++) {
                if (j != i)
                    sigma += A[i * m + j] * x[j];
            }
            x_new[i] = (b[i] - sigma) / A[i * m + i]; // A[i][i]
            double d = x_new[i] - x[i];
            diff += d * d;
        }

        // обновляем x
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            x[i] = x_new[i];

        iter++;
    } 

    printf("Iterations: %d, ||dx|| = %.2e\n", iter, sqrt(diff));
    printf("Solution (first 10 elements):\n");
    int print_n = (m < 10) ? m : 10;
    for (int i = 0; i < print_n; i++)
        printf("  x[%d] = %.6f\n", i, x[i]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]); // размер системы
    int p = atoi(argv[2]); // количество потоков
    easy_iterational_method(m, p);
    return 0;
}