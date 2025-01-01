#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <vector>

// Clase para medir tiempos
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }
};

namespace integral_method {
    // Versión secuencial del método de la integral
    double calculate_pi_sequential(long long n) {
        double step = 1.0 / n;
        double sum = 0.0;
        
        for (long long i = 0; i < n; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        
        return step * sum;
    }

    // Versión paralela con OpenMP
    double calculate_pi_parallel(long long n) {
        double step = 1.0 / n;
        double sum = 0.0;
        
        #pragma omp parallel for reduction(+:sum)
        for (long long i = 0; i < n; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        
        return step * sum;
    }
}

namespace monte_carlo {
    // Versión secuencial del método de Monte Carlo
    double calculate_pi_sequential(long long n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        
        long long inside_circle = 0;
        
        for (long long i = 0; i < n; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if (x*x + y*y <= 1.0) {
                inside_circle++;
            }
        }
        
        return 4.0 * inside_circle / n;
    }

    // Versión paralela con OpenMP
    double calculate_pi_parallel(long long n) {
        long long inside_circle = 0;
        
        #pragma omp parallel reduction(+:inside_circle)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);
            
            #pragma omp for
            for (long long i = 0; i < n; i++) {
                double x = dis(gen);
                double y = dis(gen);
                if (x*x + y*y <= 1.0) {
                    inside_circle++;
                }
            }
        }
        
        return 4.0 * inside_circle / n;
    }
}

// Función para realizar pruebas con diferente número de hilos
void run_tests(long long n) {
    const int max_threads = 6;
    std::vector<int> thread_counts = {1, 2, 3, 4, 5, 6};
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\nPruebas con n = " << n << " iteraciones\n";
    std::cout << "============================================\n";
    
    // Medición secuencial para el método de la integral
    Timer timer;
    double pi_integral = integral_method::calculate_pi_sequential(n);
    double time_integral_seq = timer.elapsed();
    std::cout << "Método Integral (Secuencial):\n";
    std::cout << "PI = " << pi_integral << "\n";
    std::cout << "Tiempo: " << time_integral_seq << " segundos\n\n";
    
    // Medición secuencial para Monte Carlo
    timer = Timer();
    double pi_monte_carlo = monte_carlo::calculate_pi_sequential(n);
    double time_monte_carlo_seq = timer.elapsed();
    std::cout << "Método Monte Carlo (Secuencial):\n";
    std::cout << "PI = " << pi_monte_carlo << "\n";
    std::cout << "Tiempo: " << time_monte_carlo_seq << " segundos\n\n";
    
    // Pruebas paralelas para diferentes números de hilos
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        std::cout << "\nPruebas con " << threads << " hilos:\n";
        std::cout << "------------------------\n";
        
        // Método de la integral
        timer = Timer();
        pi_integral = integral_method::calculate_pi_parallel(n);
        double time_integral = timer.elapsed();
        double speedup_integral = time_integral_seq / time_integral;
        
        std::cout << "Método Integral:\n";
        std::cout << "PI = " << pi_integral << "\n";
        std::cout << "Tiempo: " << time_integral << " segundos\n";
        std::cout << "Speedup: " << speedup_integral << "\n";
        std::cout << "Eficiencia: " << speedup_integral/threads << "\n\n";
        
        // Método de Monte Carlo
        timer = Timer();
        pi_monte_carlo = monte_carlo::calculate_pi_parallel(n);
        double time_monte_carlo = timer.elapsed();
        double speedup_monte_carlo = time_monte_carlo_seq / time_monte_carlo;
        
        std::cout << "Método Monte Carlo:\n";
        std::cout << "PI = " << pi_monte_carlo << "\n";
        std::cout << "Tiempo: " << time_monte_carlo << " segundos\n";
        std::cout << "Speedup: " << speedup_monte_carlo << "\n";
        std::cout << "Eficiencia: " << speedup_monte_carlo/threads << "\n";
    }
}

int main() {
    const long long n = 1000000000; // Mil millones de iteraciones
    run_tests(n);
    return 0;
}