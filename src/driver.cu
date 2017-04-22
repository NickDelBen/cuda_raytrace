
#include <stdio.h>

#include "camera.h"
#include "world.h"
#include "line.h"
#include "raytracer.h"

#define BLOCKS 32
#define THREADS 32

int main(int argc, char ** argv)
{
	FILE* fp = fopen(argv[1], "r");

	camera_t* c = Camera_read(fp);
	printf("Read and Created camera on host\n");

	world_t* w = World_read(fp);
	printf("Read and created world on host\n");
	
	fclose(fp);

	printf("Read world background=(%hu, %hu, %hu)   gloabl_ambient=%f\n", w->bg[0], w->bg[1], w->bg[2], w->global_ambient);

	Camera_calculateVectors(c);
	printf("Calculated camera components and normal");

	camera_t* d_c = Camera_toDevice(c);
	printf("Copied camera to device\n");

	world_t* d_w = World_toDevice(w);
	printf("Copied world to device\n");







	line_t* d_r;
	cudaMalloc(&d_r, 256);
	// Camera_createRays(d_c, d_r, BLOCKS, THREADS);
	// printf("Created rays from camera on device\n");

	cudaFree(d_r);
	printf("Freed rays from device\n");

	Camera_freeHost(c);
	printf("Freed camera on host\n");

	World_freeHost(w);
	printf("Freed world on host\n");

	Camera_freeDevice(d_c);
	printf("Freed camera on device\n");

	World_freeDevice(d_w);
	printf("Freed world on device\n");

	return 0;
}

