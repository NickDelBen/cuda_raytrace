
#include "raytracer.h"

// Creates the rays from a camera on the device
void Camera_createRays (camera_t* camera, line_t* rays, unsigned int blocks, unsigned int threads)
{
	unsigned int per_warp = blocks * threads;
	unsigned int num_rays = camera->width * camera->height;
	Camera_createRays_k<<<blocks, threads>>>(camera, rays, num_rays / per_warp, num_rays % per_warp);
}


// Kernel for creating the rays from a camera on the device
__global__ void Camera_createRays_k (camera_t* camera, line_t* rays, unsigned int to_do, unsigned int extra)
{
	float start[3], curr[3];
	unsigned int thread_real, per_thread, current_ray, current_pos;

	// Find real thread index
	thread_real = gridDim.x * blockIdx.x + threadIdx.x;
	// Find amount of work this thread has to do
	per_thread = thread_real < extra ? to_do + 1 : to_do;
	// Find out the starting ray index for this thread
	current_ray = to_do * thread_real + (thread_real < extra ? thread_real : extra);
	// Find the start position of the row
	current_pos = current_ray / camera->width;
	VECTOR_COPY(start, camera->comp_vert);
	VECTOR_SCALE(camera->comp_vert, current_pos);
	// Find location into current row we are
	current_pos = current_ray % ((int) camera->width);
	VECTOR_COPY(curr, camera->comp_horiz);
	VECTOR_SCALE(curr, current_pos);
	VECTOR_ADD(curr, curr, start);
	// Calculate the rays this thread is responsible for
	while (current_ray < current_ray + per_thread) {
		// Store this ray in the result ray location
		VECTOR_COPY(rays[current_ray].position, curr);
		VECTOR_COPY(rays[current_ray].direction, camera->normal);
		// Move to next ray
		current_ray++;
		current_pos++;
		if (current_pos >= camera->width) {
			// We have reached end of row
			current_pos = 0;
			// Increase start by a single row
			VECTOR_ADD(start, start, camera->comp_vert);
			// Current position is start of next row
			VECTOR_COPY(curr, start);
		} else {
			// We move to next column in row
			VECTOR_ADD(curr, curr, camera->comp_horiz);
		}
	}
	printf("This is dog\n");
}
