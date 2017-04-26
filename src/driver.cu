
#include <stdio.h>

#include "camera.h"
#include "world.h"
#include "line.h"
#include "raytracer.h"

#define BLOCKS 32
#define THREADS 256
#define MAX_REFLECTIONS 10

int main(int argc, char **argv)
{

	if (argc != 2) {
		printf("Please provide the scene file descriptor as an argument.\n");
		return EXIT_FAILURE;
	}

	FILE *fp = fopen(argv[1], "r");

	if (!fp) {
		printf("Unable to open file.\n");
		return EXIT_FAILURE;
	}

	camera_t *c = Camera_read(fp);
	printf("Read and Created camera on host\n");

	world_t *w = World_read(fp);
	printf("Read and created world on host\n");
	printf("Read world background=(%hu, %hu, %hu)\tgloabl_ambient=%f\n",
		w->bg.r, w->bg.g, w->bg.b, w->global_ambient);
	
	fclose(fp);

	camera_t *d_c = Camera_toDevice(c);
	printf("Copied camera to device\n");

	int size = c->height * c->width;

	line_t *d_r;
	cudaMalloc(&d_r, sizeof(line_t) * size);
	Camera_createRays(d_c, d_r, BLOCKS, THREADS);
	printf("Created rays from camera on device\n");
	
	// // Create frame
	// color_t *f, *d_f;

	// f = (color_t*)malloc(sizeof(color_t) * size);
	// cudaMalloc(&d_f, sizeof(color_t) * size);

	// // while (true) {

		// Raytracer(f, d_f, d_r, w, size, BLOCKS, THREADS, MAX_REFLECTIONS);
		// paint(f);
		// animate(w);

	// // }

	// cudaFree(d_r);
	// printf("Freed frame on host\n");

	// free(f);
	// printf("Freed frame on device\n");

	cudaFree(d_r);
	printf("Freed rays on device\n");

	Camera_freeDevice(d_c);
	printf("Freed camera on host\n");

	Camera_freeHost(c);
	printf("Freed camera on host\n");

	World_freeHost(w);
	printf("Freed world on host\n");

	return EXIT_SUCCESS;
}
