#include <cuda_runtime.h>
#include <math.h>

__global__ void sobel(float *entrada,
                      float *salida,
                      int B,
                      int H,
                      int W)
{
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard general
    if (fila < H && col < W)
    {
        // Procesar cada imagen del batch
        for (int b = 0; b < B; b++)
        {
            int base = b * H * W;

            int i = fila * W + col;

            // Bordes de la imagen = 0
            if (fila == 0 || fila == H-1 ||
                col  == 0 || col  == W-1)
            {
                salida[base + i] = 0.0f;
            }
            else
            {
                // Vecinos
                float arriba_izq = entrada[base + (fila-1)*W + (col-1)];
                float arriba     = entrada[base + (fila-1)*W + col];
                float arriba_der = entrada[base + (fila-1)*W + (col+1)];

                float izquierda  = entrada[base + fila*W + (col-1)];
                float derecha    = entrada[base + fila*W + (col+1)];

                float abajo_izq  = entrada[base + (fila+1)*W + (col-1)];
                float abajo      = entrada[base + (fila+1)*W + col];
                float abajo_der  = entrada[base + (fila+1)*W + (col+1)];

                // Sobel X
                float Gx =
                    -1.0f * arriba_izq +
                     1.0f * arriba_der +
                    -2.0f * izquierda +
                     2.0f * derecha +
                    -1.0f * abajo_izq +
                     1.0f * abajo_der;

                // Sobel Y
                float Gy =
                    -1.0f * arriba_izq +
                    -2.0f * arriba +
                    -1.0f * arriba_der +
                     1.0f * abajo_izq +
                     2.0f * abajo +
                     1.0f * abajo_der;

                // Magnitud del gradiente
                float magnitud = sqrtf(Gx * Gx + Gy * Gy);

                salida[base + i] = magnitud;
            }
        }
    }
}