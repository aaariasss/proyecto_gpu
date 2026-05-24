#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// CARGAR IMAGEN RGB

float* cargar_imagen_rgb(const char *nombre,
                         int *H,
                         int *W)
{
    int canales;

    // Cargar imagen
    unsigned char *img =
        stbi_load(nombre,
                  W,
                  H,
                  &canales,
                  3);

    if (!img)
    {
        printf("Error cargando imagen: %s\n", nombre);
        return NULL;
    }

    int total = (*H) * (*W);

    // Reservar memoria float
    float *salida =
        (float*)malloc(3 * total * sizeof(float));

    // Convertir unsigned char -> float
    for (int i = 0; i < total; i++)
    {
        salida[0 * total + i] =
            img[3*i + 0] / 255.0f;

        salida[1 * total + i] =
            img[3*i + 1] / 255.0f;

        salida[2 * total + i] =
            img[3*i + 2] / 255.0f;
    }

    stbi_image_free(img);

    return salida;
}

// GUARDAR IMAGEN EN GRISES

void guardar_png_gris(const char *nombre,
                      float *datos,
                      int H,
                      int W)
{
    int total = H * W;

    // Buffer unsigned char
    unsigned char *img =
        (unsigned char*)malloc(total);

    // Convertir float -> uchar
    for (int i = 0; i < total; i++)
    {
        float v = datos[i];

        // Clamp
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;

        img[i] = (unsigned char)(v * 255.0f);
    }

    // Guardar PNG
    stbi_write_png(nombre,
                   W,
                   H,
                   1,
                   img,
                   W);

    free(img);
}

// GUARDAR IMAGEN RGB

void guardar_png_rgb(const char *nombre,
                     float *datos,
                     int H,
                     int W)
{
    int total = H * W;

    unsigned char *img =
        (unsigned char*)malloc(3 * total);

    // Convertir float -> uchar
    for (int i = 0; i < total; i++)
    {
        float R = datos[0 * total + i];
        float G = datos[1 * total + i];
        float B = datos[2 * total + i];

        // Clamp
        if (R < 0.0f) R = 0.0f;
        if (R > 1.0f) R = 1.0f;

        if (G < 0.0f) G = 0.0f;
        if (G > 1.0f) G = 1.0f;

        if (B < 0.0f) B = 0.0f;
        if (B > 1.0f) B = 1.0f;

        img[3*i + 0] =
            (unsigned char)(R * 255.0f);

        img[3*i + 1] =
            (unsigned char)(G * 255.0f);

        img[3*i + 2] =
            (unsigned char)(B * 255.0f);
    }

    stbi_write_png(nombre,
                   W,
                   H,
                   3,
                   img,
                   W * 3);

    free(img);
}