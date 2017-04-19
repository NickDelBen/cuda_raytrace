
#include "raytracer.h"

// Creates the rays from a camera on the device
void Camera_createRays (camera_t* camera, line_t* rays)
{
	unsigned int per_warp = BLOCKS * THREADS;
	unsigned int num_rays = camera->width * camera->height;
	Camera_createRays_k(camera, rays, num_rays / per_warp, num_rays % per_warp);
}

// Kernel for creating the rays from a camera on the device
__global__ void Camera_createRays_k (camera_t* camera, line_t* rays, unsigned int to_do, unsigned int extra)
{
	// Find real thread index
	unsigned int thread_real = gridDim.x * BlockIdx.x + threadIdx.x;
	// Find amount of work this thread has to do
	unsigned int per_thread = thread_real < extra ? to_do + 1 : to_do;
	// Find out the starting ray index for this thread
	unsigned int current_ray = to_do * thread_real + (thread_real < extra ? thread_real : extra);
	unsigned int current_row = current_ray / camera->width;
	unsigned int current_col = current_ray % camera->width;
	// Calculate the rays this thread is responsible for
	while (current_ray < current_ray + per_thread) {
		// Move to next ray
		current_ray++;
		current_col++;
		if (current_col == camera->width) {
			current_col = 0;
			current_row++;
		}
	}
	printf("This is dog\n");
}
