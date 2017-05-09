
// Cuda raytracer library functions

#ifndef _h_raytracer
#define _h_raytracer

#include "world.h"
#include "line.h"
#include "vector3.h"
#include "helpers.h"

#define CUDA_DEVICE 0

/******************************
* Lunches the tracer kernel.
* @param d_f             Frame memory location for storing the frame on the
*						 device.
* @param d_r             A pointer to the camera rays on the device.
* @param w               A pointer to the world object.
* @param size            The size of the portview panel.
* @param blocks          The number of kernel blocks.
* @param threads         The number of kernel threads.
* @param max_reflections The number of possible bounces for a light ray.
******************************/
void Raytracer(COLOR * d_f, line_t * d_r, world_t * w, int size,
	int blocks, int threads, int max_reflections);

/******************************
* Traces the rays from the eye position into the scene.
* @param d_r    A pointer to the camera rays on the device.
* @param d_f    Frame memory location for storing the frame on the device.
* @param w      A pointer to the world object.
* @param w_size The size of the world.
* @param b_work The amount of work each block will do.
* @param t_work The amount of work each thread will do.
******************************/
__global__ void Raytracer_trace (line_t * d_r, COLOR * d_f, world_t * w,
	int w_size, int b_work, int t_work);

/******************************
* Calculates the color of each pixel.
* @param color A pointer that will containe the resultant color.
* @param w     A pointer to the world object.
* @param ray   A pointer to a line_t object that has the ray equation.
******************************/
__device__ void Raytracer_calculatePixelColor (COLOR * color, world_t * d_w,
	line_t * ray);

/******************************
* Calculates the shading model of each pixel.
* @param shading_model A pointer that will containe the resultant color.
* @param d_w           A pointer to the world object.
* @param i_object      A pointer to the object that is intersected.
* @param ray           A pointer to a line_t object that has the ray equation.
* @param distance      The distance between the camera and object.
******************************/
__device__ void Raytracer_evaluateShadingModel (COLOR * shading_model,
	world_t  * d_w, object_t * i_object, line_t * ray, float distance);

/******************************
* Calculates the diffuse factor.
* @param n A pointer to the normal.
* @param l A pointer to the light direction.
******************************/
__device__ float Raytracer_diffuse(float * n, float * l);

/******************************
* Calculates the specular factor.
* @param ray     A pointer to the initial ray direction.
* @param n       A pointer to the normal.
* @param l       A pointer to the light direction.
* @param fallout The fallout value.
******************************/
__device__ float Raytracer_specular(float * ray, float * n,
    float * l, float fallout);

#endif
