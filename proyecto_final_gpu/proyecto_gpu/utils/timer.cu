#include <cuda_runtime.h>
#include <stdio.h>


// Inicia timer
void iniciar_timer(cudaEvent_t *start,
                   cudaEvent_t *stop)
{
    cudaEventCreate(start);
    cudaEventCreate(stop);

    cudaEventRecord(*start);
}


// Detiene timer y devuelve tiempo
float detener_timer(cudaEvent_t start,
                    cudaEvent_t stop)
{
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(&ms,
                         start,
                         stop);

    return ms;
}


// Imprime tiempo
void imprimir_tiempo(const char *nombre,
                     float ms)
{
    printf("%s: %.3f ms\n",
           nombre,
           ms);
}