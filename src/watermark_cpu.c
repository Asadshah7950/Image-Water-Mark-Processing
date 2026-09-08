#include <stdio. h>
#include <stdlib. h>
#include "../include/watermark.h"

void embed_watermark_cpu(Image *img, Watermark *wm, Config *cfg, int start_row, int end_row) {
    for (int y = start_row; y < end_row && y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int wm_x = (x * wm->width) / img->width;
            int wm_y = (y * wm->height) / img->height;

            unsigned char wm_bit = wm->data[wm_y * wm->width + wm_x];
            int img_idx = (y * img->width + x) * 3;

            if (wm_bit > 128) {
                img->data[img_idx] |= 1;
                img->data[img_idx + 1] |= 1;
                img->data[img_idx + 2] |= 1;
            } else {
                img->data[img_idx] &= 0xFE;
                img->data[img_idx + 1] &= 0xFE;
                img->data[img_idx + 2] &= 0xFE;
            }
        }
    }
}

Watermark* extract_watermark_cpu(Image *img, Config *cfg) {
    int wm_width = img->width / 4;
    int wm_height = img->height / 4;

    Watermark *wm = (Watermark*)malloc(sizeof(Watermark));
    wm->width = wm_width;
    wm->height = wm_height;
    wm->data = (unsigned char*)malloc(wm_width * wm_height);

    for (int wm_y = 0; wm_y < wm_height; wm_y++) {
        for (int wm_x = 0; wm_x < wm_width; wm_x++) {
            int img_x = (wm_x * img->width) / wm_width;
            int img_y = (wm_y * img->height) / wm_height;

            int img_idx = (img_y * img->width + img_x) * 3;

            int bit_count = 0;
            bit_count += (img->data[img_idx] & 1);
            bit_count += (img->data[img_idx + 1] & 1);
            bit_count += (img->data[img_idx + 2] & 1);

            wm->data[wm_y * wm_width + wm_x] = (bit_count >= 2) ? 255 : 0;
        }
    }

    return wm;
}
