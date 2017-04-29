
#include <stdio.h>
#include <time.h>

#include "camera.h"
#include "world.h"
#include "line.h"
#include "raytracer.h"
#include "canvas.h"

#define WINDOW_TITLE "CUDA Raytracer by Nick & Zaid\0"
#define BLOCKS 256
#define THREADS 64
#define MAX_REFLECTIONS 10

camera_t *h_camera;
color_t  *d_frame;
canvas_t *canvas;
line_t   *d_rays;
camera_t *d_camera;
world_t  *h_world;

void do_work()
{
	clock_t tick = clock();
	// Trace the next frame
	Raytracer(d_frame, d_rays, h_world, h_camera->width * h_camera->height, BLOCKS, THREADS, MAX_REFLECTIONS);
	// Copy the raytraced frame back to the host
	cudaMemcpy(canvas->pixels, d_frame, sizeof(color_t) * h_camera->width * h_camera->height, cudaMemcpyDeviceToHost);
	// animate(w);
	clock_t tock = clock();
	sprintf(canvas->message, "FPS: %.2lf\n", 1.0 / ((double)(tock - tick) / CLOCKS_PER_SEC));
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Please provide the scene file path as an argument.\n");
		return EXIT_FAILURE;
	}

	FILE *fp = fopen(argv[1], "r");

	if (!fp) {
		printf("Unable to open file.\n");
		return EXIT_FAILURE;
	}

	h_camera = Camera_read(fp);
	printf("Read and Created camera on host\n");

	h_world = World_read(fp);
	printf("Read and created world on host\n");
	
	fclose(fp);

	d_camera = Camera_toDevice(h_camera);
	printf("Copied camera to device\n");

	int h = h_camera->height,
		w = h_camera->width,
		size = h * w;

	cudaMalloc(&d_rays, sizeof(line_t) * size);
	Camera_createRays(h_camera, d_camera, d_rays, BLOCKS, THREADS);
	printf("Created rays from camera on device\n");

	cudaMalloc(&d_frame, sizeof(color_t) * size);
	printf("Created space for frame result on device\n");

	char* title = (char*)malloc(sizeof(char) * strlen(WINDOW_TITLE));
	memcpy(title, WINDOW_TITLE, strlen(WINDOW_TITLE));
	canvas = Canvas_create(h, w, title);
	free(title);
	printf("Created canvas\n");

	Canvas_setRenderFunction(canvas, do_work, 1);
	printf("Set canvas render function\n");

	// Begin the main render loop
	printf("Beginning raytracer loop\n");
	Canvas_startLoop(canvas, argc, argv);
	//Raytracer(d_frame, d_rays, h_world, h_camera->width * h_camera->height, BLOCKS, THREADS, MAX_REFLECTIONS);

	Canvas_free(canvas);
	printf("Freed canvas\n");

	cudaFree(d_frame);
	printf("Freed frame on device\n");

	cudaFree(d_rays);
	printf("Freed rays on device\n");

	Camera_freeDevice(d_camera);
	printf("Freed camera on device\n");

	Camera_freeHost(h_camera);
	printf("Freed camera on host\n");

	World_freeHost(h_world);
	printf("Freed world on host\n");

	return EXIT_SUCCESS;
}
