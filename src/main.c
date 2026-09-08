#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/watermark.h"

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <mode> [args]\\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *mode = argv[1];
    double start_time = MPI_Wtime();

    if (strcmp(mode, "embed") == 0 && argc >= 5) {
        if (rank == 0) {
            printf("\\n╔════════════════════════════════════╗\\n");
            printf("║   EMBED WATERMARK (MPI+CUDA)      ║\\n");
            printf("╚════════════════════════════════════╝\\n");
            printf("   MPI Processes: %d\\n\\n", size);
        }

        Image *img = NULL;
        Watermark *logo = NULL;
        int img_width, img_height, wm_size;

        if (rank == 0) {
            img = load_image(argv[2]);
            if (!img) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            wm_size = img->width / 4;
            logo = load_logo(argv[3], wm_size, wm_size);
            if (!logo) {
                free_image(img);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            img_width = img->width;
            img_height = img->height;
            printf("   Watermark size: %dx%d\\n", wm_size, wm_size);
        }

        MPI_Bcast(&img_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&img_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&wm_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            img = (Image*)malloc(sizeof(Image));
            img->width = img_width;
            img->height = img_height;
            img->channels = 3;
            img->data = (unsigned char*)malloc(img_width * img_height * 3);

            logo = (Watermark*)malloc(sizeof(Watermark));
            logo->width = wm_size;
            logo->height = wm_size;
            logo->data = (unsigned char*)malloc(wm_size * wm_size);
        }

        MPI_Bcast(img->data, img_width * img_height * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(logo->data, wm_size * wm_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        int rows_per_proc = img_height / size;
        int start_row = rank * rows_per_proc;
        int end_row = (rank == size - 1) ? img_height : (rank + 1) * rows_per_proc;

        if (rank == 0) printf("\\n🔧 Processing with MPI+GPU (Rank %d:  rows %d-%d)...\\n", rank, start_row, end_row);

        bool use_gpu = (strcmp(argv[4], "gpu") == 0);
        Config cfg = {1, use_gpu, rank, size};

        if (use_gpu) {
            embed_watermark_cuda(img, logo, &cfg, start_row, end_row);
        } else {
            embed_watermark_cpu(img, logo, &cfg, start_row, end_row);
        }

        if (rank == 0) {
            for (int r = 1; r < size; r++) {
                int r_start = r * rows_per_proc;
                int r_end = (r == size - 1) ? img_height : (r + 1) * rows_per_proc;
                int r_rows = r_end - r_start;
                MPI_Recv(img->data + r_start * img_width * 3, r_rows * img_width * 3,
                         MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            save_image("results/watermarked_image.png", img);
            printf("\\n✅ Success! \\n");
        } else {
            int my_rows = end_row - start_row;
            MPI_Send(img->data + start_row * img_width * 3, my_rows * img_width * 3,
                     MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }

        free_image(img);
        free_watermark(logo);

    } else if (strcmp(mode, "extract") == 0 && argc >= 4) {
        if (rank == 0) {
            printf("\\n╔════════════════════════════════════╗\\n");
            printf("║   EXTRACT WATERMARK               ║\\n");
            printf("╚════════════════════════════════════╝\\n\\n");

            Image *img = load_image(argv[2]);
            if (!img) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            bool use_gpu = (strcmp(argv[3], "gpu") == 0);
            Config cfg = {1, use_gpu, 0, 1};

            printf("\\n🔧 Extracting with %s... \\n", use_gpu ? "GPU" : "CPU");

            Watermark *extracted;
            if (use_gpu) {
                extracted = extract_watermark_cuda(img, &cfg);
            } else {
                extracted = extract_watermark_cpu(img, &cfg);
            }

            printf("   Extracted:  %dx%d\\n", extracted->width, extracted->height);

            Image *wm_img = (Image*)malloc(sizeof(Image));
            wm_img->width = extracted->width;
            wm_img->height = extracted->height;
            wm_img->channels = 3;
            wm_img->data = (unsigned char*)malloc(extracted->width * extracted->height * 3);
            for (int i = 0; i < extracted->width * extracted->height; i++) {
                wm_img->data[i*3] = wm_img->data[i*3+1] = wm_img->data[i*3+2] = extracted->data[i];
            }
            save_image("results/extracted_watermark.png", wm_img);

            printf("\\n✅ Success!\\n");

            free_image(wm_img);
            free_watermark(extracted);
            free_image(img);
        }

    } else if (strcmp(mode, "check") == 0 && argc >= 5) {
        if (rank == 0) {
            printf("\\n╔════════════════════════════════════╗\\n");
            printf("║   CHECK WATERMARK                 ║\\n");
            printf("╚════════════════════════════════════╝\\n\\n");

            Image *img = load_image(argv[2]);
            if (!img) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            bool use_gpu = (strcmp(argv[4], "gpu") == 0);
            Config cfg = {1, use_gpu, 0, 1};

            printf("\\n🔧 Extracting watermark with %s...\\n", use_gpu ? "GPU" : "CPU");

            Watermark *extracted;
            if (use_gpu) {
                extracted = extract_watermark_cuda(img, &cfg);
            } else {
                extracted = extract_watermark_cpu(img, &cfg);
            }

            printf("   Extracted size: %dx%d\\n", extracted->width, extracted->height);

            printf("\\n🔧 Loading original logo...\\n");
            Watermark *original_logo = load_logo(argv[3], extracted->width, extracted->height);

            if (! original_logo) {
                fprintf(stderr, "   Error loading logo\\n");
                free_image(img);
                free_watermark(extracted);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            printf("   Original size: %dx%d\\n", original_logo->width, original_logo->height);

            printf("\\n🔧 Comparing...\\n");
            double similarity = calculate_similarity(original_logo, extracted);

            printf("\\n📊 RESULTS:\\n");
            printf("   Similarity: %.2f%%%%\\n", similarity);

            if (similarity > 90.0) {
                printf("   ✅ VERIFIED - Watermark Present! \\n");
            } else if (similarity > 70.0) {
                printf("   ⚠️  PARTIAL - Possible Match\\n");
            } else {
                printf("   ❌ NOT FOUND - No Watermark\\n");
            }

            free_image(img);
            free_watermark(extracted);
            free_watermark(original_logo);
        }

    } else if (strcmp(mode, "similarity") == 0 && argc >= 4) {
        if (rank == 0) {
            printf("\\n╔════════════════════════════════════╗\\n");
            printf("║   IMAGE SIMILARITY CHECK          ║\\n");
            printf("╚════════════════════════════════════╝\\n\\n");

            Image *img1 = load_image(argv[2]);
            Image *img2 = load_image(argv[3]);

            if (!img1 || !img2) {
                fprintf(stderr, "Error loading images\\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            double similarity = calculate_image_similarity(img1, img2);
            double psnr = calculate_psnr(img1, img2);

            printf("\\n📊 RESULTS:\\n");
            printf("   Similarity: %.2f%%%%\\n", similarity);
            printf("   PSNR: %.2f dB\\n", psnr);

            if (similarity > 95.0) {
                printf("   ✅ VERY SIMILAR\\n");
            } else if (similarity > 80.0) {
                printf("   ✅ SIMILAR\\n");
            } else if (similarity > 60.0) {
                printf("   ⚠️  SOMEWHAT SIMILAR\\n");
            } else {
                printf("   ❌ DIFFERENT\\n");
            }

            free_image(img1);
            free_image(img2);
        }
    }

    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("\\n⏱️  Time:  %.3f sec (MPI:  %d processes)\\n\\n", end_time - start_time, size);
    }

    MPI_Finalize();
    return 0;
}
