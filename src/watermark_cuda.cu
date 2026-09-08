#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/watermark.h"

#define BLOCK_SIZE 16
#define CUDA_CHECK(call) { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
        exit(EXIT_FAILURE); \\
    } \\
}

__global__ void embed_lsb_kernel(unsigned char *img, unsigned char *wm,
                                  int img_width, int img_height,
                                  int wm_width, int wm_height,
                                  int start_row, int end_row) {
    int x = blockIdx.x * blockDim.x + threadIdx. x;
    int y = blockIdx.y * blockDim. y + threadIdx.y + start_row;

    if (x >= img_width || y >= end_row || y >= img_height) return;

    int wm_x = (x * wm_width) / img_width;
    int wm_y = (y * wm_height) / img_height;

    if (wm_x >= wm_width || wm_y >= wm_height) return;

    unsigned char wm_bit = wm[wm_y * wm_width + wm_x];
    int img_idx = (y * img_width + x) * 3;

    if (wm_bit > 128) {
        img[img_idx] |= 1;
        img[img_idx + 1] |= 1;
        img[img_idx + 2] |= 1;
    } else {
        img[img_idx] &= 0xFE;
        img[img_idx + 1] &= 0xFE;
        img[img_idx + 2] &= 0xFE;
    }
}

__global__ void extract_lsb_kernel(unsigned char *img, unsigned char *wm,
                                     int img_width, int img_height,
                                     int wm_width, int wm_height) {
    int wm_x = blockIdx.x * blockDim. x + threadIdx.x;
    int wm_y = blockIdx.y * blockDim. y + threadIdx.y;

    if (wm_x >= wm_width || wm_y >= wm_height) return;

    int img_x = (wm_x * img_width) / wm_width;
    int img_y = (wm_y * img_height) / wm_height;

    int img_idx = (img_y * img_width + img_x) * 3;

    int bit_count = 0;
    bit_count += (img[img_idx] & 1);
    bit_count += (img[img_idx + 1] & 1);
    bit_count += (img[img_idx + 2] & 1);

    wm[wm_y * wm_width + wm_x] = (bit_count >= 2) ? 255 : 0;
}

void embed_watermark_cuda(Image *img, Watermark *wm, Config *cfg, int start_row, int end_row) {
    int rows_to_process = end_row - start_row;
    int img_size = img->width * img->height * 3;
    int wm_size = wm->width * wm->height;

    unsigned char *d_img, *d_wm;

    CUDA_CHECK(cudaMalloc(&d_img, img_size));
    CUDA_CHECK(cudaMalloc(&d_wm, wm_size));

    CUDA_CHECK(cudaMemcpy(d_img, img->data, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wm, wm->data, wm_size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((img->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (rows_to_process + BLOCK_SIZE - 1) / BLOCK_SIZE);

    embed_lsb_kernel<<<grid, block>>>(d_img, d_wm, img->width, img->height,
                                       wm->width, wm->height, start_row, end_row);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(img->data, d_img, img_size, cudaMemcpyDeviceToHost));

    cudaFree(d_img);
    cudaFree(d_wm);
}

Watermark* extract_watermark_cuda(Image *img, Config *cfg) {
    int wm_width = img->width / 4;
    int wm_height = img->height / 4;

    Watermark *wm = (Watermark*)malloc(sizeof(Watermark));
    wm->width = wm_width;
    wm->height = wm_height;
    wm->data = (unsigned char*)malloc(wm_width * wm_height);

    int img_size = img->width * img->height * 3;
    int wm_size = wm_width * wm_height;

    unsigned char *d_img, *d_wm;

    CUDA_CHECK(cudaMalloc(&d_img, img_size));
    CUDA_CHECK(cudaMalloc(&d_wm, wm_size));

    CUDA_CHECK(cudaMemcpy(d_img, img->data, img_size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((wm_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (wm_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    extract_lsb_kernel<<<grid, block>>>(d_img, d_wm, img->width, img->height,
                                         wm_width, wm_height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(wm->data, d_wm, wm_size, cudaMemcpyDeviceToHost));

    cudaFree(d_img);
    cudaFree(d_wm);

    return wm;
}
