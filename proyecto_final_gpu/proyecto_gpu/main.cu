#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


// Kernels
__global__ void escala_grises(float *entrada,
                              float *salida,
                              int B,
                              int H,
                              int W);

__global__ void sobel(float *entrada,
                      float *salida,
                      int B,
                      int H,
                      int W);

__global__ void reducir_max(float *entrada,
                             float *maximos,
                             int H,
                             int W);

__global__ void normalizar(float *entrada,
                           float *salida,
                           float *maximos,
                           int B,
                           int H,
                           int W);

__global__ void calcular_rmse(float *entrada,
                              float *referencia,
                              float *sumas,
                              int H,
                              int W);


// Utils imagen
float* cargar_imagen_rgb(const char *nombre,
                         int *H,
                         int *W);

void guardar_png_gris(const char *nombre,
                      float *datos,
                      int H,
                      int W);

void guardar_png_rgb(const char *nombre,
                     float *datos,
                     int H,
                     int W);


// Utils timer
void iniciar_timer(cudaEvent_t *start,
                   cudaEvent_t *stop);

float detener_timer(cudaEvent_t start,
                    cudaEvent_t stop);

void imprimir_tiempo(const char *nombre,
                     float ms);


// Macro CUDA_CHECK
#define CUDA_CHECK(call)                                   \
do {                                                       \
    cudaError_t err = call;                                \
    if (err != cudaSuccess)                                \
    {                                                      \
        printf("CUDA ERROR: %s\n",                         \
               cudaGetErrorString(err));                   \
        exit(1);                                           \
    }                                                      \
} while(0)



int main()
{
    // Batch size
    int B = 8;

    int H, W;

    // Cargar primera imagen para obtener dimensiones
    float *img0 =
        cargar_imagen_rgb("imagenes/img0.png",
                          &H,
                          &W);

    if (img0 == NULL)
    {
        return 1;
    }

    int total_rgb  = B * 3 * H * W;
    int total_gris = B * H * W;


    // Memoria CPU batch RGB
    float *h_rgb =
        (float*)malloc(total_rgb * sizeof(float));

    // Copiar imagen 0
    for (int i = 0; i < 3 * H * W; i++)
    {
        h_rgb[i] = img0[i];
    }

    free(img0);


    // Cargar resto de imagenes
    for (int b = 1; b < B; b++)
    {
        char nombre[100];

        sprintf(nombre,
                "imagenes/img%d.png",
                b);

        int h2, w2;

        float *temp =
            cargar_imagen_rgb(nombre,
                              &h2,
                              &w2);

        if (temp == NULL)
        {
            return 1;
        }

        for (int i = 0; i < 3 * H * W; i++)
        {
            h_rgb[b * 3 * H * W + i] = temp[i];
        }

        free(temp);
    }


    // Guardar imagen original
    guardar_png_rgb("resultados/imagen_00_original.png",
                     h_rgb,
                     H,
                     W);


    // Memoria GPU
    float *d_rgb;
    float *d_gris;
    float *d_bordes;
    float *d_normalizada;

    float *d_referencia;

    float *d_maximos;
    float *d_sumas;


    CUDA_CHECK(
        cudaMalloc(&d_rgb,
                   total_rgb * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_gris,
                   total_gris * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_bordes,
                   total_gris * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_normalizada,
                   total_gris * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_referencia,
                   H * W * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_maximos,
                   B * sizeof(float)));

    int num_bloques_rmse = (H * W + 255) / 256;

    CUDA_CHECK(
        cudaMalloc(&d_sumas,
                   B * num_bloques_rmse * sizeof(float)));


    // Timer total
    cudaEvent_t total_start, total_stop;

    iniciar_timer(&total_start,
                   &total_stop);


    // Transferencia H -> D
    cudaEvent_t h2d_start, h2d_stop;

    iniciar_timer(&h2d_start,
                   &h2d_stop);

    CUDA_CHECK(
        cudaMemcpy(d_rgb,
                   h_rgb,
                   total_rgb * sizeof(float),
                   cudaMemcpyHostToDevice));

    float tiempo_h2d =
        detener_timer(h2d_start,
                       h2d_stop);


    // Configuracion grid
    dim3 bloque(16,16);

    dim3 grid(
        (W + bloque.x - 1) / bloque.x,
        (H + bloque.y - 1) / bloque.y
    );


    // Kernel 1 - grises
    cudaEvent_t k1_start, k1_stop;

    iniciar_timer(&k1_start,
                   &k1_stop);

    escala_grises<<<grid,bloque>>>(
        d_rgb,
        d_gris,
        B,
        H,
        W
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    float tiempo_k1 =
        detener_timer(k1_start,
                      k1_stop);


    // Kernel 2 - Sobel
    cudaEvent_t k2_start, k2_stop;

    iniciar_timer(&k2_start,
                   &k2_stop);

    sobel<<<grid,bloque>>>(
        d_gris,
        d_bordes,
        B,
        H,
        W
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    float tiempo_k2 =
        detener_timer(k2_start,
                      k2_stop);


    // Kernel 3 - normalizacion
    cudaEvent_t k3_start, k3_stop;

    iniciar_timer(&k3_start,
                   &k3_stop);

    // Paso A: encontrar el maximo de cada imagen en GPU
    reducir_max<<<B, 256, 256 * sizeof(float)>>>(
        d_bordes,
        d_maximos,
        H,
        W
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Paso B: dividir cada pixel entre el maximo de su imagen
    dim3 grid3(
        (W + bloque.x - 1) / bloque.x,
        (H + bloque.y - 1) / bloque.y,
        B
    );

    normalizar<<<grid3, bloque>>>(
        d_bordes,
        d_normalizada,
        d_maximos,
        B,
        H,
        W
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    float tiempo_k3 =
        detener_timer(k3_start,
                      k3_stop);


    // Referencia = primera imagen normalizada
    CUDA_CHECK(
        cudaMemcpy(
                    d_referencia,
                    d_normalizada, 
                    H * W * sizeof(float),
                    cudaMemcpyDeviceToDevice
    ));


    // Kernel 4 - RMSE
    cudaEvent_t k4_start, k4_stop;

    iniciar_timer(&k4_start,
                   &k4_stop);

    // Paso A: suma parcial por bloque, por imagen
    for (int b = 0; b < B; b++)
    {
        calcular_rmse<<<num_bloques_rmse, 256>>>(
            d_normalizada + b * H * W,
            d_referencia,
            d_sumas + b * num_bloques_rmse,
            H,
            W
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Paso B: reduccion final en CPU (suma los bloques de cada imagen)
    float *h_sumas =
        (float*)malloc(B * num_bloques_rmse * sizeof(float));

    CUDA_CHECK(
        cudaMemcpy(h_sumas,
                   d_sumas,
                   B * num_bloques_rmse * sizeof(float),
                   cudaMemcpyDeviceToHost));

    float *h_rmse_calc =
        (float*)malloc(B * sizeof(float));

    int total_px = H * W;

    for (int b = 0; b < B; b++)
    {
        float suma = 0.0f;

        for (int blq = 0; blq < num_bloques_rmse; blq++)
        {
            suma += h_sumas[b * num_bloques_rmse + blq];
        }

        h_rmse_calc[b] = sqrtf(suma / total_px);
    }

    free(h_sumas);

    float tiempo_k4 =
        detener_timer(k4_start,
                      k4_stop);


    // Transferencia D -> H
    cudaEvent_t d2h_start, d2h_stop;

    iniciar_timer(&d2h_start,
                   &d2h_stop);

    float *h_gris =
        (float*)malloc(total_gris * sizeof(float));

    float *h_bordes =
        (float*)malloc(total_gris * sizeof(float));

    float *h_normalizada =
        (float*)malloc(total_gris * sizeof(float));

    CUDA_CHECK(
        cudaMemcpy(h_gris,
                   d_gris,
                   total_gris * sizeof(float),
                   cudaMemcpyDeviceToHost));

    CUDA_CHECK(
        cudaMemcpy(h_bordes,
                   d_bordes,
                   total_gris * sizeof(float),
                   cudaMemcpyDeviceToHost));

    CUDA_CHECK(
        cudaMemcpy(h_normalizada,
                   d_normalizada,
                   total_gris * sizeof(float),
                   cudaMemcpyDeviceToHost));

    float tiempo_d2h =
        detener_timer(d2h_start,
                      d2h_stop);


    // Timer total
    float tiempo_total =
        detener_timer(total_start,
                      total_stop);


    // Guardar resultados
    guardar_png_gris(
        "resultados/imagen_00_grises.png",
        h_gris,
        H,
        W
    );

    guardar_png_gris(
        "resultados/imagen_00_bordes.png",
        h_bordes,
        H,
        W
    );

    guardar_png_gris(
        "resultados/imagen_00_normalizada.png",
        h_normalizada,
        H,
        W
    );


    // Guardar RMSE
    FILE *f =
        fopen("resultados/rmse_por_imagen.txt",
              "w");

    for (int b = 0; b < B; b++)
    {
        fprintf(f,
                "Imagen %d RMSE: %f\n",
                b,
                h_rmse_calc[b]);

        printf("Imagen %d RMSE: %f\n",
               b,
               h_rmse_calc[b]);
    }

    fclose(f);


    // CPU equivalente para calcular speedup
    clock_t cpu_inicio = clock();

    // Paso 1 CPU: escala de grises
    float *cpu_gris = (float*)malloc(total_gris * sizeof(float));
    for (int b = 0; b < B; b++)
    {
        int base = b * 3 * H * W;
        for (int i = 0; i < H * W; i++)
        {
            cpu_gris[b * H * W + i] =
                0.2989f * h_rgb[base + 0 * H * W + i] +
                0.5870f * h_rgb[base + 1 * H * W + i] +
                0.1140f * h_rgb[base + 2 * H * W + i];
        }
    }

    // Paso 2 CPU: Sobel
    float *cpu_bordes = (float*)calloc(total_gris, sizeof(float));
    for (int b = 0; b < B; b++)
    {
        int base = b * H * W;
        for (int fila = 1; fila < H - 1; fila++)
        {
            for (int col = 1; col < W - 1; col++)
            {
                float ai = cpu_gris[base + (fila-1)*W + (col-1)];
                float a  = cpu_gris[base + (fila-1)*W + col];
                float ad = cpu_gris[base + (fila-1)*W + (col+1)];
                float iz = cpu_gris[base + fila*W + (col-1)];
                float de = cpu_gris[base + fila*W + (col+1)];
                float bi = cpu_gris[base + (fila+1)*W + (col-1)];
                float bj = cpu_gris[base + (fila+1)*W + col];
                float bd = cpu_gris[base + (fila+1)*W + (col+1)];
                float Gx = -ai + ad - 2.0f*iz + 2.0f*de - bi + bd;
                float Gy = -ai - 2.0f*a - ad + bi + 2.0f*bj + bd;
                cpu_bordes[base + fila*W + col] = sqrtf(Gx*Gx + Gy*Gy);
            }
        }
    }

    // Paso 3 CPU: normalizacion
    float *cpu_norm = (float*)malloc(total_gris * sizeof(float));
    for (int b = 0; b < B; b++)
    {
        int base = b * H * W;
        float max_val = 0.0f;
        for (int i = 0; i < H * W; i++)
            if (cpu_bordes[base + i] > max_val)
                max_val = cpu_bordes[base + i];
        for (int i = 0; i < H * W; i++)
            cpu_norm[base + i] = (max_val > 0.0f) ?
                cpu_bordes[base + i] / max_val : 0.0f;
    }

    // Paso 4 CPU: RMSE
    float *cpu_rmse = (float*)malloc(B * sizeof(float));
    float *ref_cpu  = cpu_norm;
    for (int b = 0; b < B; b++)
    {
        float suma = 0.0f;
        for (int i = 0; i < H * W; i++)
        {
            float d = cpu_norm[b * H * W + i] - ref_cpu[i];
            suma += d * d;
        }
        cpu_rmse[b] = sqrtf(suma / (H * W));
    }

    clock_t cpu_fin = clock();
    float tiempo_cpu_ms =
        1000.0f * (float)(cpu_fin - cpu_inicio) / CLOCKS_PER_SEC;

    float speedup = tiempo_cpu_ms / tiempo_total;

    free(cpu_gris);
    free(cpu_bordes);
    free(cpu_norm);
    free(cpu_rmse);


    // Imprimir tiempos
    printf("\n");

    imprimir_tiempo("Transferencia H->D",
                    tiempo_h2d);

    imprimir_tiempo("Kernel grises",
                    tiempo_k1);

    imprimir_tiempo("Kernel Sobel",
                    tiempo_k2);

    imprimir_tiempo("Kernel normalizacion",
                    tiempo_k3);

    imprimir_tiempo("Kernel RMSE",
                    tiempo_k4);

    imprimir_tiempo("Transferencia D->H",
                    tiempo_d2h);

    imprimir_tiempo("Pipeline total",
                    tiempo_total);

    imprimir_tiempo("CPU equivalente",
                    tiempo_cpu_ms);

    printf("Speedup GPU vs CPU: %.2fx\n",
           speedup);


    // Liberar memoria CPU
    free(h_rgb);
    free(h_gris);
    free(h_bordes);
    free(h_normalizada);
    free(h_rmse_calc);


    // Liberar memoria GPU
    cudaFree(d_rgb);
    cudaFree(d_gris);
    cudaFree(d_bordes);
    cudaFree(d_normalizada);

    cudaFree(d_referencia);

    cudaFree(d_maximos);
    cudaFree(d_sumas);


    return 0;
}