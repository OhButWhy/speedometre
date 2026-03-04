#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <time.h>
using namespace std;




void easy_iterational_method_two(int m) {
    const double EPS = 1e-5;
    vector<double> b(m, m + 1.0);
    
    vector<double> x(m, 0.0);
    vector<double> x_new(m, 0.0);
    
    vector<double> A(m * m);
    
    int iter = 0;
    double criteria = 1.0;
    double t = 0.01;
    bool should_stop = false;
    

    #pragma omp parallel num_threads(4)
    {
        int nthreads = omp_get_num_threads(); // кол-во потоков
        int threadid = omp_get_thread_num();  // текущий поток
        int items_per_thread = m / nthreads;  // сколько строк на поток
        int lb = threadid * items_per_thread; // первая строка для потока
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        

        for (int i = lb; i <= ub; i++){
            for (int j = 0; j < m; j++)
                A[i * m + j] = (i == j) ? 2.0 : 1.0;
            
        }
        
        #pragma omp barrier
        
        while (should_stop == false) {
            
            double local_criteria = 0.0;
            for (int i = lb; i <= ub; i++) {
                double sigma = 0.0;
                for (int j = 0; j < m; j++) {
                    if (j != i){
                        sigma += A[i * m + j] * x[j];
                    }
                }
                x_new[i] = x[i] - t * (sigma - b[i]);
                
                if (b[i] != 0.0) {
                    double d = (sigma - b[i]) / b[i];
                    local_criteria += d * d;
                } else {
                    local_criteria += (sigma - b[i]) * (sigma - b[i]);
                }
            }
            
            #pragma omp atomic
            criteria += local_criteria;
            
            #pragma omp barrier

            for (int i = lb; i <= ub; i++)
                x[i] = x_new[i];
            
            #pragma omp barrier
            // std::cout<< "dfghj3" << endl;

            #pragma omp single
            {
                if (sqrt(criteria) <= EPS || iter > 20) {
                    should_stop = true;
                } else {
                    if (criteria > 1000) {
                        t *= -1.0;
                    }
                    criteria = 0.0;
                    iter++;
                }
                
            }
            #pragma omp barrier
        }
        
    }
}


int main() { 
    int m = 40;
    double t = omp_get_wtime();    
    easy_iterational_method_two(m);
    t = omp_get_wtime()- t;

    cout<< "iuhyg:"<< t << endl;
    return 0;
}