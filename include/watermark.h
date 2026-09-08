#ifndef WATERMARK_H
#define WATERMARK_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Image;

typedef struct {
    int width;
    int height;
    unsigned char *data;
} Watermark;

typedef struct {
    int alpha;
    bool use_gpu;
    int mpi_rank;
    int mpi_size;
} Config;

#ifdef __cplusplus
extern "C" {
#endif

Image* load_image(const char *filename);
void save_image(const char *filename, Image *img);
void free_image(Image *img);
Watermark* load_logo(const char *filename, int target_width, int target_height);
void free_watermark(Watermark *wm);
void embed_watermark_cuda(Image *img, Watermark *wm, Config *config, int start_row, int end_row);
Watermark* extract_watermark_cuda(Image *img, Config *config);
void embed_watermark_cpu(Image *img, Watermark *wm, Config *config, int start_row, int end_row);
Watermark* extract_watermark_cpu(Image *img, Config *config);
double calculate_similarity(Watermark *wm1, Watermark *wm2);
double calculate_image_similarity(Image *img1, Image *img2);
double calculate_psnr(Image *img1, Image *img2);

#ifdef __cplusplus
}
#endif

#endif
