
// Cuda raytracer library functions

#ifndef _h_raytracer
#define _h_raytracer

#include "world.h"
#include "frame.h"
#include "line.h"

/******************************
* Lunches the tracer kernel.
* @param f Frame memory location for storing the frame on the host.
* @param d_f Frame memory location for storing the frame on the device.
* @param d_r A pointer to the camera rays on the device.
* @param w A pointer to the world object.
* @param size The size of the portview panel.
* @param blocks The number of kernel blocks.
* @param threads The number of kernel threads.
* @param max_reflections The number of possible bounces for a light ray.
******************************/
void Raytracer(color_t *f, color_t *d_f, line_t *d_r, world_t *w, int size,
	int blocks, int threads, int max_reflections);

/******************************
* Traces the rays from the eye position into the scene.
* @param d_r A pointer to the camera rays on the device.
* @param d_f Frame memory location for storing the frame on the device.
* @param w A pointer to the world object.
* @param size The size of the portview panel.
* @param b_work The amount of work each block will do.
******************************/
__global__ void Raytracer_trace (line_t *d_r, color_t *d_f, world_t *d_w,
	int size, int b_work);

/******************************
* Calculates the color of each pixel.
* @param color A pointer that will container the resultant color.
* @param w A pointer to the world object.
* @param ray A pointer to a line_t object that has the ray equation.
******************************/
__device__ void Raytracer_calculatePixelColor(color_t *color, world_t *d_w,
	line_t *ray);

#endif
