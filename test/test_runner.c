#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../include/watermark.h"

int test_watermark_embedding_and_extraction() {
    printf("=== Test 1: Watermark Embedding & Extraction Consistency ===\n");
    int width = 256;
    int height = 256;
    int channels = 3;
    int img_size = width * height * channels;

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (unsigned char*)malloc(img_size);

    Image *orig = (Image*)malloc(sizeof(Image));
    orig->width = width;
    orig->height = height;
    orig->channels = channels;
    orig->data = (unsigned char*)malloc(img_size);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            unsigned char r = (unsigned char)((x + y) % 256);
            unsigned char g = (unsigned char)((x * 2) % 256);
            unsigned char b = (unsigned char)((y * 3) % 256);
            img->data[idx] = r;
            img->data[idx + 1] = g;
            img->data[idx + 2] = b;
            orig->data[idx] = r;
            orig->data[idx + 1] = g;
            orig->data[idx + 2] = b;
        }
    }

    int wm_width = 64;
    int wm_height = 64;
    Watermark *wm = (Watermark*)malloc(sizeof(Watermark));
    wm->width = wm_width;
    wm->height = wm_height;
    wm->data = (unsigned char*)malloc(wm_width * wm_height);

    for (int y = 0; y < wm_height; y++) {
        for (int x = 0; x < wm_width; x++) {
            wm->data[y * wm_width + x] = ((x / 8 + y / 8) % 2 == 0) ? 255 : 0;
        }
    }

    Config cfg;
    cfg.alpha = 1;
    cfg.use_gpu = false;
    cfg.mpi_rank = 0;
    cfg.mpi_size = 1;

    embed_watermark_cpu(img, wm, &cfg, 0, height);
    Watermark *extracted = extract_watermark_cpu(img, &cfg);

    double similarity = calculate_similarity(wm, extracted);
    double psnr = calculate_psnr(orig, img);

    printf("  Extracted Watermark Similarity: %.2f%%\n", similarity);
    printf("  Embedded Image PSNR: %.2f dB\n", psnr);

    assert(similarity >= 99.0);
    assert(psnr >= 40.0);

    free_watermark(wm);
    free_watermark(extracted);
    free_image(img);
    free_image(orig);

    printf("✓ PASS: Embedding & Extraction test completed successfully.\n\n");
    return 0;
}

int test_psnr_and_image_similarity_metrics() {
    printf("=== Test 2: PSNR and Image Similarity Metric Validation ===\n");
    int width = 128;
    int height = 128;
    int channels = 3;
    int img_size = width * height * channels;

    Image *img1 = (Image*)malloc(sizeof(Image));
    img1->width = width;
    img1->height = height;
    img1->channels = channels;
    img1->data = (unsigned char*)malloc(img_size);
    memset(img1->data, 128, img_size);

    Image *img2 = (Image*)malloc(sizeof(Image));
    img2->width = width;
    img2->height = height;
    img2->channels = channels;
    img2->data = (unsigned char*)malloc(img_size);
    memset(img2->data, 128, img_size);

    double identical_psnr = calculate_psnr(img1, img2);
    double identical_sim = calculate_image_similarity(img1, img2);
    printf("  Identical Image PSNR: %.2f (inf expected), Similarity: %.2f%%\n", identical_psnr, identical_sim);
    assert(identical_sim == 100.0);

    for (int i = 0; i < img_size; i++) {
        img2->data[i] = (unsigned char)(img1->data[i] + 1);
    }
    double perturbed_psnr = calculate_psnr(img1, img2);
    printf("  1-LSB Perturbation PSNR: %.2f dB\n", perturbed_psnr);
    assert(perturbed_psnr > 45.0 && perturbed_psnr < 50.0);

    free_image(img1);
    free_image(img2);

    printf("✓ PASS: Metric validation completed successfully.\n\n");
    return 0;
}

int main() {
    printf("====================================================\n");
    printf("Image-Water-Mark-Processing Automated Test Suite\n");
    printf("====================================================\n\n");

    int res1 = test_watermark_embedding_and_extraction();
    int res2 = test_psnr_and_image_similarity_metrics();

    if (res1 == 0 && res2 == 0) {
        printf("ALL TESTS PASSED (100%% SUCCESS RATE)\n");
        return 0;
    }
    return 1;
}
