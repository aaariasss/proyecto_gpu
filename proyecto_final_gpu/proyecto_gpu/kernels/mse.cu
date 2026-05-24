#include <cuda_runtime.h>
#include <math.h>

__global__ void calcular_rmse(float *entrada,
                              float *referencia,
                              float *sumas,
                              int H,
                              int W)
{
    __shared__ float cache[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = H * W;

    // Cada thread calcula su diferencia al cuadrado
    if (idx < total)
    {
        float diff = entrada[idx] - referencia[idx];
        cache[tid] = diff * diff;
    }
    else
    {
        cache[tid] = 0.0f;
    }

    __syncthreads();

    // Reduccion en arbol dentro del bloque
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] += cache[tid + s];

        __syncthreads();
    }

    // El thread 0 escribe la suma parcial de este bloque
    // La division y la raiz las hace el CPU despues de sumar todos los bloques
    if (tid == 0)
    {
        sumas[blockIdx.x] = cache[0];
    }
}