#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <time.h>
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

    // omp_set_num_threads(p);

    int iter = 0;
    double criteria = 1.0;
    double t = 0.01;

    while (sqrt(criteria) > EPS) {
        if (iter > 1000) {
            printf("Too many iterations, stopping.\n");
            break;
        }

        if (criteria > 1000){
            t *= -1.0;
        }
        criteria = 0.0;


 
        // #pragma omp parallel for schedule(static) reduction(+:criteria)
        for (int i = 0; i < m; i++) {
            double sigma = 0.0;
            for (int j = 0; j < m; j++) {
                if (j != i)
                    sigma += A[i * m + j] * x[j];
            }
            x_new[i] = x[i] -t * (sigma - b[i]); //(b[i] - sigma) / A[i * m + i]; // A[i][i]
            if (b[i] != 0.0) {
                double d = (sigma - b[i]) / b[i]; // (b[i] - sigma) / A[i * m + i];
                criteria += d * d;
            } else{
                criteria += (sigma - b[i]) * (sigma - b[i]);
            }
        }

        // обновляем x
        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            x[i] = x_new[i];

        iter++;
    } 

    // printf("Iterations: %d, ||dx|| = %.2e\n", iter, sqrt(criteria));
    // printf("Solution (first 10 elements):\n");
    // int print_n = (m < 10) ? m : 10;
    // for (int i = 0; i < print_n; i++)
    //     printf("  x[%d] = %.6f\n", i, x[i]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        // fprintf(stderr, "Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]); // размер системы
    int p = atoi(argv[2]); // количество потоков
    // double t = omp_get_wtime();
    easy_iterational_method(m, p);
    // t = omp_get_wtime()- t;
    cout<< "iuhyg"<< endl;
    return 0;
}