#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITERATION 255

struct complex{
    double real;
    double imag;
};

int calculate_pixel(struct complex c){
    double z_real = 0;
    double z_imag = 0;
    int lengthsq;
    int iteration = 0;

    do{
        double temp = z_real * z_real - z_imag * z_imag + c.real;
        z_imag = 2*z_real*z_imag + c.imag;
        z_real = temp;
        lengthsq = z_imag*z_imag + z_real*z_real;
        iteration++;
    }while((lengthsq<4.0)&&(iteration<MAX_ITERATION));

    return iteration;
}

void save_pgm(const char *filename, int *image){
    FILE* pgmimg;
    int temp;

    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    int count = 0;

    for (int i = 0;i<HEIGHT;i++){
        for(int j = 0;j<WIDTH;j++){
            temp = *(image + i * WIDTH + j);
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg,"\n");
    }
    fclose(pgmimg);
}

int main(int argc, char** argv){
    int ranks,number_processors;
    struct complex c;
    int *image = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ranks);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processors);

    int number_of_rows = HEIGHT/ number_processors;
    int *local_image = (int *)malloc(sizeof(int) * number_of_rows * WIDTH);
    int *entire_image = NULL;

    double starting_Time, end, total, AVG = 0.0;


    starting_Time = MPI_Wtime();
    for(int i = 0;i<number_of_rows;i++){
      for(int j = 0; j<WIDTH;j++){
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i + ranks * number_of_rows - HEIGHT / 2.0) * 4.0 / HEIGHT;
            *(local_image + i * WIDTH + j) = calculate_pixel(c);
        }
    }

    end = MPI_Wtime();
    total = end-starting_Time;
    printf("Processor %d: Execution time = %f seconds\n", ranks, total);

    double *time_array = (double *)malloc(sizeof(double) * number_processors);
    MPI_Gather(&total, 1, MPI_DOUBLE, time_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (ranks == 0){
        double sum = 0;
        for(int i = 0; i<number_processors;i++){
          sum+=time_array[i];
        }
        AVG = sum/number_processors;
        printf("The average execution time is %f ms", AVG/number_processors * 1000);
        entire_image = (int *)malloc(sizeof(int) * HEIGHT * WIDTH);
    }
    MPI_Gather(local_image, number_of_rows * WIDTH, MPI_INT, entire_image, number_of_rows * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    if(ranks == 0){
        save_pgm("mandlebrot_static.pgm", entire_image);
        free(entire_image);
    }


    free(local_image);
    free(time_array);
    MPI_Finalize();


    return 0;
}

