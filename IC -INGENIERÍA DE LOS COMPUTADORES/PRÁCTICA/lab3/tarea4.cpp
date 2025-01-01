#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>

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

// Parte 1: Demostración de condición de carrera
void test_race_condition() {
    std::cout << "\n=== Test de Condición de Carrera ===\n";
    
    // Versión con race condition
    for(int test = 0; test < 5; test++) {
        int max = 0;
        int min = 1000;
        
        #pragma omp parallel for
        for (int i = 1000; i >= 0; i--) {
            if (i > max) max = i;
            if (i < min) min = i;
        }
        
        std::cout << "Test " << test + 1 << " - Sin critical - "
                  << "Max: " << max << " Min: " << min << std::endl;
    }
    
    // Versión con critical
    for(int test = 0; test < 5; test++) {
        int max = 0;
        int min = 1000;
        
        #pragma omp parallel for
        for (int i = 1000; i >= 0; i--) {
            #pragma omp critical
            {
                if (i > max) max = i;
                if (i < min) min = i;
            }
        }
        
        std::cout << "Test " << test + 1 << " - Con critical - "
                  << "Max: " << max << " Min: " << min << std::endl;
    }
}

// Parte 2: Análisis de inicialización de vectores
void test_vector_initialization(int size) {
    std::cout << "\n=== Test de Inicialización de Vectores ===\n";
    std::cout << "Tamaño del vector: " << size << std::endl;
    
    // Método 1: push_back secuencial
    {
        Timer timer;
        std::vector<float> v1;
        for (int i = 0; i < size; i++) {
            v1.push_back(i);
        }
        std::cout << "Método 1 (push_back secuencial): " 
                  << timer.elapsed() << " segundos\n";
    }
    
    // Método 2: reserva previa y push_back secuencial
    {
        Timer timer;
        std::vector<float> v1;
        v1.reserve(size);
        for (int i = 0; i < size; i++) {
            v1.push_back(i);
        }
        std::cout << "Método 2 (reserve + push_back secuencial): " 
                  << timer.elapsed() << " segundos\n";
    }
    
    // Método 3: constructor con tamaño y acceso directo secuencial
    {
        Timer timer;
        std::vector<float> v2(size);
        for (int i = 0; i < size; i++) {
            v2[i] = i;
        }
        std::cout << "Método 3 (constructor con size + [] secuencial): " 
                  << timer.elapsed() << " segundos\n";
    }
    
    // Método 4: push_back paralelo (incorrecto - demuestra el problema)
    {
        Timer timer;
        std::vector<float> v1;
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            #pragma omp critical
            v1.push_back(i);
        }
        std::cout << "Método 4 (push_back paralelo con critical - NO SEGURO): " 
                  << timer.elapsed() << " segundos\n";
        std::cout << "Tamaño final del vector: " << v1.size() 
                  << " (puede ser incorrecto)\n";
    }
    
    // Método 5: acceso directo paralelo
    {
        Timer timer;
        std::vector<float> v2(size);
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            v2[i] = i;
        }
        std::cout << "Método 5 (constructor con size + [] paralelo): " 
                  << timer.elapsed() << " segundos\n";
    }
}

int main() {
    // Test de condiciones de carrera
    test_race_condition();
    
    // Test de inicialización de vectores
    test_vector_initialization(10000000);
    
    return 0;
}