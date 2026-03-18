#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <time.h>


void easy_iterational_method_one(int m, int p, double initial_t, int chunk_size) {
    const double EPS   = 1e-5;

    omp_set_num_threads(p);
    

  
    std::vector<double> A(m * m);
    #pragma omp parallel for schedule(dynamic, chunk_size) 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A[i * m + j] = (i == j) ? 2.0 : 1.0;


    std::vector<double> b(m, m + 1.0);

    std::vector<double> x(m, 0.0);
    std::vector<double> x_new(m, 0.0);

    

    int iter = 0;
    double criteria = 1.0;
    double t = initial_t;

    while (sqrt(criteria) > EPS) {
        if (iter > 1000) {
            break;
        }

        if (criteria > 1000){
            t *= -1.0;
        }
        criteria = 0.0;


 
        #pragma omp parallel for schedule(dynamic, chunk_size) reduction(+ : criteria) 
        for (int i = 0; i < m; i++) {
            double sigma = 0.0;
            for (int j = 0; j < m; j++) {
                if (j != i)
                    sigma += A[i * m + j] * x[j];
            }
            x_new[i] = x[i] -t * (sigma - b[i]); 
            if (b[i] != 0.0) {
                double d = (sigma - b[i]) / b[i]; 
                criteria += d * d;
            } else{
                criteria += (sigma - b[i]) * (sigma - b[i]) / 0.0001;
            }
        }

        #pragma omp parallel for schedule(dynamic, chunk_size) 
        for (int i = 0; i < m; i++)
            x[i] = x_new[i];

        iter++;
    } 


}

void easy_iterational_method_two(int m, int p, double initial_t) {
    const double EPS = 1e-5;
    std::vector<double> b(m, m + 1.0);
    
    std::vector<double> x(m, 0.0);
    std::vector<double> x_new(m, 0.0);
    
    std::vector<double> A(m * m);
    
    int iter = 0;
    double criteria = 1.0;
    double t = initial_t;
    
    bool should_stop = false;
    

    #pragma omp parallel num_threads(p)
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
        
        bool my_stop = false;
        while (!my_stop) {
            #pragma omp single
            {
                if (sqrt(criteria) <= EPS || iter > 1000) {
                    should_stop = true;
                } else {
                    if (criteria > 1000) {
                        t *= -1.0;
                    }
                    criteria = 0.0;
                    iter++;
                }
            }

            my_stop = should_stop;

            if (!my_stop) {
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

                #pragma omp barrier

                #pragma omp atomic
                criteria += local_criteria;

                #pragma omp barrier

                for (int i = lb; i <= ub; i++)
                    x[i] = x_new[i];

                #pragma omp barrier
            }
        }
        
    }

}

int main(int argc, char **argv) { //int argc, char **argv
    if (argc < 4) {
        return 1;
    }
    int m = atoi(argv[1]); // размер системы
    int p = atoi(argv[2]); // количество потоков
    int chunk_size = atoi(argv[3]); // размер чанка для первого способа
    if (chunk_size <= 0) {
        return 1;
    }
    double step_t = 1.0 / (m + 1.0);

    double elapsed_one = omp_get_wtime();    
    easy_iterational_method_one(m, p, step_t, chunk_size);
    elapsed_one = omp_get_wtime()- elapsed_one;
    std::cout<< "separate parallel sections time:"<< elapsed_one << std::endl;

    // double elapsed_two = omp_get_wtime();    
    // easy_iterational_method_two(m, p, step_t);
    // elapsed_two = omp_get_wtime()- elapsed_two;

    // std::cout<< "one parallel section time:"<< elapsed_two << std::endl;
    return 0;
}