
// Cuda raytracer library functions

#ifndef _h_raytracer
#define _h_raytracer

#include "world.h"
#include "frame.h"
#include "line.h"

#define EPSILON 0.00001

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
void Raytracer(color_t *d_f, line_t *d_r, world_t *w, int size,
	int blocks, int threads, int max_reflections);

/******************************
* Traces the rays from the eye position into the scene.
* @param d_r    A pointer to the camera rays on the device.
* @param d_f    Frame memory location for storing the frame on the device.
* @param w      A pointer to the world object.
* @param size   The size of the portview panel.
* @param b_work The amount of work each block will do.
* @param t_work The amount of work each thread will do.
******************************/
__global__ void Raytracer_trace (line_t * d_r, color_t * d_f, world_t * d_w,
	int size, int b_work, int t_work);

/******************************
* Calculates the color of each pixel.
* @param color A pointer that will containe the resultant color.
* @param w     A pointer to the world object.
* @param ray   A pointer to a line_t object that has the ray equation.
******************************/
__device__ void Raytracer_calculatePixelColor (color_t *color, world_t *d_w,
	line_t *ray);

/******************************
* Calculates the shading model of each pixel.
* @param shading_model A pointer that will containe the resultant color.
* @param d_w           A pointer to the world object.
* @param i_object      A pointer to the object that is intersected.
* @param ray           A pointer to a line_t object that has the ray equation.
* @param distance      The distance between the camera and object.
******************************/
__device__ void Raytracer_evaluateShadingModel (color_t * shading_model,
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

/******************************
* Finds the intersection point between ray and object.
* @param intersection A pointer to the intersection coordinate that will be
					  populated.
* @param ray          A pointer to a line_t object that has the ray equation.
* @param distance     The distance between the ray and the object.
******************************/
__device__ void Raytracer_findIntersectionPoint(float * intersection,
	line_t * ray, float distance);

/******************************
* Finds the direction for the reflected ray.
* @param reflected A pointer to the reflected ray that will be populated.
* @param ray       A pointer to the direction of the initial ray.
* @param normal    A pointer to the normal of the intersection.
******************************/
__device__ void Raytracer_findReflectedRay(float * reflected, float * ray,
	float * normal);

/******************************
* Finds the intersection between an object and a ray.
* @param ray    A pointer to a line_t object that has the ray equation.
* @param object A pointer to the object that will be tested for intersection.
******************************/
__device__ float Object_intersect (line_t * ray, object_t * object);

/******************************
* Finds the intersection between an sphere and a ray.
* @param ray    A pointer to a line_t object that has the ray equation.
* @param sphere A pointer to the sphere that will be tested for intersection.
******************************/
__device__ float Sphere_intersect (line_t * ray, sphere_t * sphere);


/******************************
* Finds the intersection between an triangle and a ray.
* @param ray      A pointer to a line_t object that has the ray equation.
* @param triangle A pointer to the triangle that will be tested for intersection.
******************************/
__device__ float Triangle_intersect (line_t * ray, triangle_t * triangle);

/******************************
* Finds the normal on the point of intersection.
* @param normal       A pointer to a normal vector that will be populated.
* @param object       A pointer to the object that is intersected.
* @param intersection A pointer to the intersection point.
******************************/
__device__ void Object_normal (float * normal, object_t * object,
	float * intersection);

/******************************
* Finds the normal on the point of intersection.
* @param normal       A pointer to a normal vector that will be populated.
* @param sphere       A pointer to the sphere that is intersected.
* @param intersection A pointer to the intersection point.
******************************/
__device__ void Sphere_normal (float * normal, sphere_t * sphere,
	float * intersection);

#endif
