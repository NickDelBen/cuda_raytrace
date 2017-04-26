
#include "raytracer.h"

void Raytracer(color_t *f, color_t *d_f, line_t *d_r, world_t *w, int size,
	int blocks, int threads, int max_reflections)
{
	int b_work = size / blocks,
		r_size = sizeof(line_t) * b_work,
		f_size = sizeof(color_t) * b_work;

	// Copy animated world to device.
	world_t *d_w = World_toDevice(w);

	// Initialize frame
	Frame_init<<<blocks, threads, f_size>>>(d_f, size);


	line_t *rays;
	cudaMalloc(&rays, sizeof(line_t) * size);
	cudaMemcpy(rays, d_r, sizeof(line_t) * size, cudaMemcpyDeviceToDevice);

	// Traces rays bounces.
	for (int i = 0; i < max_reflections; ++i) {
		Raytracer_trace<<<blocks, threads, r_size + f_size>>>(rays, d_f, d_w,
			size, b_work);
	}

	// Frees world from device memory.
	World_freeDevice(d_w);

	// Copy frame to host
	cudaMemcpy(f, d_f, sizeof(color_t) * size, cudaMemcpyDeviceToHost);
}

__global__ void Raytracer_trace (line_t *d_r, color_t *d_f, world_t *d_w,
	int size, int b_work)
{
	int b_offset 	= blockIdx.x * b_work,
		t_work 		= b_work / blockDim.x,
		t_offset 	= threadIdx.x * t_work,
		offset 		= b_offset + t_offset;

	extern __shared__ float smem[];

	// Assign shared memory locations to the rays and frame arrays.
	line_t 	*rays 	= (line_t*)smem;
	color_t *frame 	= (color_t*)&rays[b_work],
			result;

	// Copy from global memory to shared memory.
	memcpy(&rays[t_offset], &d_r[offset], sizeof(line_t) * t_work);
	memcpy(&frame[t_offset], &d_f[offset], sizeof(color_t) * t_work);

	// Process all the pixels assigned to this thread
	for (int i = t_offset; i < t_offset + t_work; ++i) {
		Raytracer_calculatePixelColor(&result, d_w, &rays[i]);
		COLOR_ADD(&frame[i], &frame[i], &result);
	}

	// Copy the results of the trace on the frame tile to the global memory.
	memcpy(&d_r[offset], &rays[t_offset], sizeof(line_t) * t_work);
	memcpy(&d_f[offset], &frame[t_offset], sizeof(color_t) * t_work);
}

__device__ void Raytracer_calculatePixelColor(color_t *color, world_t *d_w,
	line_t *ray)
{
    color_t bg = d_w->bg;

    //sets the pixel color to the default background color
    color->r = bg.r;
    color->b = bg.b;
    color->g = bg.g;

    //iterates all world objects and finds if the rays intersect any of these
    //objects
    // for(node_t *node = world->objects->head; node != NULL;
    //     node = node->next) {
    //     object = node->element;
    //     distance = intersect(origin, ray, object);

    //     if (!isnan(distance) && (closest_distance == -1 
    //         || distance < closest_distance)) {
    //         closest_distance = distance;
    //         closest_object = object;
    //     }
    // }

    //calculates the color of the pixel if the ray at that point intersects an
    //object
    // if (closest_object != NULL) {
    //     evaluate_shading_model(&shading_model, raytracer, origin, ray,
    //         closest_object, closest_distance, depth);

    //     color->r = fmin(shading_model.r, 255);
    //     color->b = fmin(shading_model.b, 255);
    //     color->g = fmin(shading_model.g, 255);
    // }
}	
