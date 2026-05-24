#include <cuda_runtime.h>

__global__ void escala_grises(float *entrada,
                              float *salida,
                              int B,
                              int H,
                              int W)
{
    // Coordenadas globales del pixel
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard para evitar salirnos de la imagen
    if (fila < H && col < W)
    {
        // Indice lineal del pixel
        int i = fila * W + col;

        // Procesar cada imagen del batch
        for (int b = 0; b < B; b++)
        {
            // Offset de la imagen b
            int base = b * 3 * H * W;

            // Leer canales RGB
            float R = entrada[base + 0 * H * W + i];
            float G = entrada[base + 1 * H * W + i];
            float Bc = entrada[base + 2 * H * W + i];

            // Convertir a gris
            salida[b * H * W + i] =
                0.2989f * R +
                0.5870f * G +
                0.1140f * Bc;
        }
    }
}