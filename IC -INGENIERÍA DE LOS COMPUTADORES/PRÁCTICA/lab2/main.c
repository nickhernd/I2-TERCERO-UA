#include <stdio.h>
#include <stdlib.h>
#include "processing.h"

const int LEN = 100;  // Longitud del vector
const int GRUPO = 1;  // Número de grupo

int main(int argc, char** argv) {
    int rank, size, chunk_size, remainder;
    int *vector, *local_vector;
    int local_start, local_end, local_size;
    double start_time, end_time;
    
    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Obtener el rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Obtener el número total de procesos
    
    // El proceso 0 inicializa el vector
    if (rank == 0) {
        int firma = init_lab2(GRUPO);
        vector = (int *) malloc(sizeof(int) * LEN);
        for (int i = 0; i < LEN; i++) { vector[i] = i + 1; }
        printf("Firma: 0x%08X\n", firma);
    }
    
    // Calcular el tamaño de los chunks para cada proceso
    chunk_size = LEN / size;
    remainder = LEN % size;
    
    // Determinar el tamaño y los índices locales para cada proceso
    if (rank < remainder) {
        local_size = chunk_size + 1;
        local_start = rank * (chunk_size + 1);
    } else {
        local_size = chunk_size;
        local_start = rank * chunk_size + remainder;
    }
    local_end = local_start + local_size - 1;
    
    // Asignar memoria para el vector local
    local_vector = (int *) malloc(sizeof(int) * local_size);
    
    // Distribuir el vector entre todos los procesos
    MPI_Scatter(vector, local_size, MPI_INT, local_vector, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Medir el tiempo de inicio
    start_time = MPI_Wtime();
    
    // Ejecutar las cuatro funciones de procesamiento en el vector local
    process_block1(GRUPO, local_vector, local_size, 0, local_size-1);
    process_block2(GRUPO, local_vector, local_size, 0, local_size-1);
    process_block3(GRUPO, local_vector, local_size, 0, local_size-1);
    process_block4(GRUPO, local_vector, local_size, 0, local_size-1);
    
    // Medir el tiempo de fin
    end_time = MPI_Wtime();
    
    // Recoger los resultados de todos los procesos
    MPI_Gather(local_vector, local_size, MPI_INT, vector, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // El proceso 0 imprime el tiempo de ejecución y libera la memoria
    if (rank == 0) {
        printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
        free(vector);
    }
    
    // Liberar la memoria local y finalizar MPI
    free(local_vector);
    MPI_Finalize();
    return 0;
}