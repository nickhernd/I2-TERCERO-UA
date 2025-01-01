#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <future>
#include <omp.h>
#include <iomanip>

// Funciones de utilidad para medir tiempo
class Timer {
private:
    std::chrono::steady_clock::time_point start;
public:
    Timer() : start(std::chrono::steady_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }
};

// Versión 1: Implementación secuencial
namespace sequential {
    double average(const std::vector<double>& v) {
        double sum = 0.0;
        for(size_t i = 0; i < v.size(); i++) {
            sum += v[i];
        }
        return sum/v.size();
    }

    double maximum(const std::vector<double>& v) {
        double max = v[0];
        for(size_t i = 1; i < v.size(); i++) {
            if (v[i] > max) max = v[i];
        }
        return max;
    }

    double minimum(const std::vector<double>& v) {
        double min = v[0];
        for(size_t i = 1; i < v.size(); i++) {
            if (v[i] < min) min = v[i];
        }
        return min;
    }
}

// Versión 2: Implementación con reduction
namespace reduction {
    double average(const std::vector<double>& v) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for(size_t i = 0; i < v.size(); i++) {
            sum += v[i];
        }
        return sum/v.size();
    }

    double maximum(const std::vector<double>& v) {
        double max = v[0];
        #pragma omp parallel for reduction(max:max)
        for(size_t i = 1; i < v.size(); i++) {
            if (v[i] > max) max = v[i];
        }
        return max;
    }

    double minimum(const std::vector<double>& v) {
        double min = v[0];
        #pragma omp parallel for reduction(min:min)
        for(size_t i = 1; i < v.size(); i++) {
            if (v[i] < min) min = v[i];
        }
        return min;
    }
}

// Versión 3: Implementación con sections
namespace sections {
    void compute_all(const std::vector<double>& v, double& min, double& max, double& avg) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                min = sequential::minimum(v);
            }
            
            #pragma omp section
            {
                max = sequential::maximum(v);
            }
            
            #pragma omp section
            {
                avg = sequential::average(v);
            }
        }
    }
}

// Versión 4: Implementación con async
namespace async {
    void compute_all(const std::vector<double>& v, double& min, double& max, double& avg) {
        auto future_min = std::async(std::launch::async, sequential::minimum, std::ref(v));
        auto future_max = std::async(std::launch::async, sequential::maximum, std::ref(v));
        auto future_avg = std::async(std::launch::async, sequential::average, std::ref(v));
        
        min = future_min.get();
        max = future_max.get();
        avg = future_avg.get();
    }
}

// Versión 5: Combinación de reduction y sections
namespace reduction_sections {
    void compute_all(const std::vector<double>& v, double& min, double& max, double& avg) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                min = reduction::minimum(v);
            }
            
            #pragma omp section
            {
                max = reduction::maximum(v);
            }
            
            #pragma omp section
            {
                avg = reduction::average(v);
            }
        }
    }
}

// Versión 6: Combinación de reduction y async
namespace reduction_async {
    void compute_all(const std::vector<double>& v, double& min, double& max, double& avg) {
        auto future_min = std::async(std::launch::async, reduction::minimum, std::ref(v));
        auto future_max = std::async(std::launch::async, reduction::maximum, std::ref(v));
        auto future_avg = std::async(std::launch::async, reduction::average, std::ref(v));
        
        min = future_min.get();
        max = future_max.get();
        avg = future_avg.get();
    }
}

// Función principal para pruebas
int main() {
    // Configuración inicial
    const int size = 100000000;
    std::random_device os_seed;
    const int seed = 1;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<> distribute(0, 1000);
    
    // Generar datos
    std::vector<double> v(size);
    for(int i = 0; i < size; i++) {
        v[i] = distribute(generator);
    }
    
    double min, max, avg;
    Timer timer;
    double elapsed;
    
    // Prueba 1: Secuencial
    std::cout << "\nPrueba secuencial:" << std::endl;
    timer = Timer();
    min = sequential::minimum(v);
    max = sequential::maximum(v);
    avg = sequential::average(v);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << std::fixed << std::setprecision(3) << elapsed << "s" << std::endl;
    std::cout << "Min: " << min << " Max: " << max << " Avg: " << avg << std::endl;
    double sequential_time = elapsed;
    
    // Prueba 2: Reduction
    std::cout << "\nPrueba reduction:" << std::endl;
    timer = Timer();
    min = reduction::minimum(v);
    max = reduction::maximum(v);
    avg = reduction::average(v);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << elapsed << "s" << std::endl;
    std::cout << "Speedup: " << sequential_time/elapsed << std::endl;
    
    // Prueba 3: Sections
    std::cout << "\nPrueba sections:" << std::endl;
    timer = Timer();
    sections::compute_all(v, min, max, avg);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << elapsed << "s" << std::endl;
    std::cout << "Speedup: " << sequential_time/elapsed << std::endl;
    
    // Prueba 4: Async
    std::cout << "\nPrueba async:" << std::endl;
    timer = Timer();
    async::compute_all(v, min, max, avg);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << elapsed << "s" << std::endl;
    std::cout << "Speedup: " << sequential_time/elapsed << std::endl;
    
    // Prueba 5: Reduction + Sections
    std::cout << "\nPrueba reduction + sections:" << std::endl;
    timer = Timer();
    reduction_sections::compute_all(v, min, max, avg);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << elapsed << "s" << std::endl;
    std::cout << "Speedup: " << sequential_time/elapsed << std::endl;
    
    // Prueba 6: Reduction + Async
    std::cout << "\nPrueba reduction + async:" << std::endl;
    timer = Timer();
    reduction_async::compute_all(v, min, max, avg);
    elapsed = timer.elapsed();
    std::cout << "Tiempo: " << elapsed << "s" << std::endl;
    std::cout << "Speedup: " << sequential_time/elapsed << std::endl;
    
    return 0;
}