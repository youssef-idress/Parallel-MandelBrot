
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

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
    int iteration = 0;

    while (z_real * z_real + z_imag * z_imag < 4.0 && iteration < MAX_ITERATION) {
        double temp = z_real * z_real - z_imag * z_imag + c.real;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = temp;
        iteration++;
    }

    return iteration;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;

    pgmimg = fopen(filename, "wb");

    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char** argv) {
    int image[HEIGHT][WIDTH];
    struct complex c;
    int ranks, number_processors;
    double starting, ending, total, AVG = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ranks);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processors);

    int num_rows = HEIGHT / number_processors;
    int start_row = ranks * num_rows;
    int end_row = (ranks == number_processors - 1) ? HEIGHT : start_row + num_rows;

    starting = MPI_Wtime();

    if (ranks == 0) {
        for (int i = 1; i < number_processors; i++) {
            MPI_Send(&image[i * num_rows][0], num_rows * WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = calculate_pixel(c);
            }
        }

        for (int i = 1; i < number_processors; i++) {
            MPI_Status status;
            MPI_Recv(&image[i * num_rows][0], num_rows * WIDTH, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }

        save_pgm("mandelbrot_dynamic.pgm", image);
    } else {
        int localbuffer[num_rows][WIDTH];
        MPI_Recv(&localbuffer[0][0], num_rows * WIDTH, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (start_row + i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                localbuffer[i][j] = calculate_pixel(c);
            }
        }
        MPI_Send(&localbuffer[0][0], num_rows * WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    ending = MPI_Wtime();

    total = ending- starting;
    AVG = total/number_processors * 1000;
    if (ranks == 0) {
        printf("Average execution time: %f ms\n", AVG);
    } else {
        printf("Processor %d execution time: %f seconds\n", ranks, total);
    }

    MPI_Finalize();
    return 0;
}

