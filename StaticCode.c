#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITERATION 255

struct complex {
    double real;
    double imag;
};

int calculate_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double lengthsq = 0;
    int iteration = 0;

    do {
        double temp = z_real * z_real - z_imag * z_imag + c.real;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = temp;
        lengthsq = z_imag * z_imag + z_real * z_real;
        iteration++;
    } while ((lengthsq < 4.0) && (iteration < MAX_ITERATION));

    return iteration;
}

void save_pgm(const char *filename, int *image) {
    FILE *pgmimg = fopen(filename, "wb");
    if (!pgmimg) {
        fprintf(stderr, "Failed to open file for writing\n");
        return;
    }
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", *(image + i * WIDTH + j));
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int rows_per_proc = HEIGHT / num_procs;
    int *local_image = (int *)malloc(sizeof(int) * rows_per_proc * WIDTH);
    int *entire_image = NULL;

    double start_time, end_time, total_time, avg_time = 0.0;

    start_time = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < WIDTH; j++) {
            struct complex c;
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i + rank * rows_per_proc - HEIGHT / 2.0) * 4.0 / HEIGHT;
            *(local_image + i * WIDTH + j) = calculate_pixel(c);
        }
    }
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
    printf("Processor %d: Execution time = %f seconds\n", rank, total_time);

    double *time_array = NULL;
    if (rank == 0) {
        time_array = (double *)malloc(sizeof(double) * num_procs);
    }
    MPI_Gather(&total_time, 1, MPI_DOUBLE, time_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sum = 0;
        for (int i = 0; i < num_procs; i++) {
            sum += time_array[i];
        }
        avg_time = sum / num_procs;
        printf("The average execution time is %f ms\n", avg_time * 1000);
        entire_image = (int *)malloc(sizeof(int) * HEIGHT * WIDTH);
    }
    MPI_Gather(local_image, rows_per_proc * WIDTH, MPI_INT, entire_image, rows_per_proc * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        save_pgm("mandelbrot_static.pgm", entire_image);
        free(entire_image);
    }

    free(local_image);
    if (rank == 0) {
        free(time_array);
    }
    MPI_Finalize();

    return 0;
}

