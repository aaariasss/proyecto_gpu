#include <cuda_runtime.h>
#include <math.h>

// Kernel A: reducción para encontrar el máximo por imagen
__global__ void reducir_max(float *entrada,
                             float *maximos,
                             int H,
                             int W)
{
    int b = blockIdx.x;          // imagen en el batch
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    int idx_base = b * H * W;

    float max_val = 0.0f;

    // cada thread recorre parte de la imagen
    for (int i = tid; i < H * W; i += blockDim.x)
    {
        float val = entrada[idx_base + i];
        if (val > max_val)
            max_val = val;
    }

    sdata[tid] = max_val;
    __syncthreads();

    // reducción en shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        maximos[b] = sdata[0];
    }
}


// Kernel B: normalización final
__global__ void normalizar(float *entrada,
                           float *salida,
                           float *maximos,
                           int B,
                           int H,
                           int W)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (fila < H && col < W)
    {
        int idx = b * H * W + fila * W + col;

        float max_val = maximos[b];

        if (max_val > 0.0f)
        {
            salida[idx] = entrada[idx] / max_val;
        }
        else
        {
            salida[idx] = 0.0f;
        }
    }
}