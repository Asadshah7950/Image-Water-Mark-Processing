#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "../include/watermark.h"

Image* load_image(const char *filename) {
    Image *img = (Image*)malloc(sizeof(Image));
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 3);
    if (!img->data) {
        fprintf(stderr, "Error loading:  %s\\n", filename);
        free(img);
        return NULL;
    }
    img->channels = 3;
    printf("✓ Loaded:  %dx%d\\n", img->width, img->height);
    return img;
}

void save_image(const char *filename, Image *img) {
    stbi_write_png(filename, img->width, img->height, img->channels,
                   img->data, img->width * img->channels);
    printf("✓ Saved: %s\\n", filename);
}

void free_image(Image *img) {
    if (img) {
        if (img->data) stbi_image_free(img->data);
        free(img);
    }
}

Watermark* load_logo(const char *filename, int target_width, int target_height) {
    int orig_width, orig_height, orig_channels;
    unsigned char *orig_data = stbi_load(filename, &orig_width, &orig_height, &orig_channels, 1);

    if (!orig_data) {
        fprintf(stderr, "Error loading logo: %s\\n", filename);
        return NULL;
    }

    Watermark *wm = (Watermark*)malloc(sizeof(Watermark));
    wm->width = target_width;
    wm->height = target_height;
    wm->data = (unsigned char*)malloc(target_width * target_height);

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            int src_x = (x * orig_width) / target_width;
            int src_y = (y * orig_height) / target_height;
            unsigned char pixel = orig_data[src_y * orig_width + src_x];
            wm->data[y * target_width + x] = (pixel > 128) ? 255 : 0;
        }
    }
    stbi_image_free(orig_data);
    printf("✓ Logo loaded: %dx%d\\n", target_width, target_height);
    return wm;
}

void free_watermark(Watermark *wm) {
    if (wm) {
        if (wm->data) free(wm->data);
        free(wm);
    }
}

double calculate_similarity(Watermark *wm1, Watermark *wm2) {
    if (wm1->width != wm2->width || wm1->height != wm2->height) return 0.0;
    int matches = 0;
    int total = wm1->width * wm1->height;
    for (int i = 0; i < total; i++) {
        if (wm1->data[i] == wm2->data[i]) matches++;
    }
    return (double)matches / total * 100.0;
}

double calculate_image_similarity(Image *img1, Image *img2) {
    if (img1->width != img2->width || img1->height != img2->height) return 0.0;
    int size = img1->width * img1->height * img1->channels;
    long long sum_sq_diff = 0;

    for (int i = 0; i < size; i++) {
        int diff = (int)img1->data[i] - (int)img2->data[i];
        sum_sq_diff += diff * diff;
    }

    double mse = (double)sum_sq_diff / size;
    double similarity = 100.0 / (1.0 + sqrt(mse));
    return similarity;
}

double calculate_psnr(Image *img1, Image *img2) {
    if (img1->width != img2->width || img1->height != img2->height) return 0.0;
    int size = img1->width * img1->height * img1->channels;
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = (double)img1->data[i] - (double)img2->data[i];
        mse += diff * diff;
    }
    mse /= size;
    if (mse == 0) return INFINITY;
    return 10.0 * log10((255.0 * 255.0) / mse);
}
