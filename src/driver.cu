
#include <stdio.h>
#include <time.h>

#include "camera.h"
#include "world.h"
#include "line.h"
#include "raytracer.h"
#include "canvas.h"
#include "sphere.h"

#define WINDOW_TITLE "CUDA Raytracer by Nick & Zaid\0"
#define BLOCKS 512 
#define THREADS 32
#define MAX_REFLECTIONS 4
#define SPEED_FACTOR 0.1

camera_t * h_camera;
COLOR    * d_frame;
canvas_t * canvas;
line_t   * d_rays;
camera_t * d_camera;
world_t  * h_world;
float    * sphere_speeds;


void do_work();
void animate_spheres();

void animate_spheres()
{
	int sphere_itr;
	sphere_t * s;
	float future;

	for (sphere_itr = 0; sphere_itr < h_world->n_objects; sphere_itr++) {
		if (h_world->objects[sphere_itr].type == TRIANGLE) {
			continue;
		}
		s = &(h_world->objects[sphere_itr].sphere);
		future = s->center[1] + sphere_speeds[sphere_itr] + (sphere_speeds[sphere_itr] < 0 ? -(s->radius) : s->radius);
		if (future < 10 || future > (h_camera->height - 10)) {
			sphere_speeds[sphere_itr] *= -1;
		}
		s->center[1] += sphere_speeds[sphere_itr];
	}
}

void do_work()
{
	clock_t tick = clock();
	// Trace the next frame
	Raytracer(d_frame, d_rays, h_world, h_camera->width * h_camera->height, BLOCKS, THREADS, MAX_REFLECTIONS);
	cudaDeviceSynchronize();
	// Copy the raytraced frame back to the host
	cudaMemcpy(canvas->pixels, d_frame, sizeof(COLOR) * CHANNELS * h_camera->width * h_camera->height, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	clock_t tock = clock();
	sprintf(canvas->message, "FPS: %.2lf\n", 1.0 / ((double)(tock - tick) / CLOCKS_PER_SEC));
	animate_spheres();
}

int main(int argc, char ** argv)
{
	if (argc != 2) {
		printf("Please provide the scene file path as an argument.\n");
		return EXIT_FAILURE;
	}

	FILE * fp = fopen(argv[1], "r");

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
	Camera_createRays<<<BLOCKS, THREADS>>>(d_camera, d_rays);
	printf("Created rays from camera on device\n");

	cudaMalloc(&d_frame, sizeof(COLOR) * CHANNELS * size);
	printf("Created space for frame result on device\n");

	char * title = (char *)malloc(sizeof(char) * strlen(WINDOW_TITLE));
	memcpy(title, WINDOW_TITLE, strlen(WINDOW_TITLE));
	canvas = Canvas_create(h, w, title);
	free(title);
	printf("Created canvas\n");

	Canvas_setRenderFunction(canvas, do_work, 1);
	printf("Set canvas render function\n");

	// Allocate space for he sphere animations
	sphere_speeds = (float *) malloc(sizeof(float) * h_world->n_objects);
	for (int sphere_itr = 0; sphere_itr < h_world->n_objects; sphere_itr++) {
		sphere_speeds[sphere_itr] = SPEED_FACTOR * ((float) (h_world->objects[sphere_itr].sphere.radius));
	}

	// Begin the main render loop
	printf("Beginning raytracer loop\n");
	Canvas_startLoop(canvas, argc, argv);
	// Raytracer(d_frame, d_rays, h_world, h_camera->width * h_camera->height, BLOCKS, THREADS, MAX_REFLECTIONS);

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

	free(sphere_speeds);

	return EXIT_SUCCESS;
}
