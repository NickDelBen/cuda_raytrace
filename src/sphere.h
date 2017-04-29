
// Describes a sphere in the raytracer system

#ifndef _h_sphere
#define _h_sphere

#include <stdio.h>

#include "line.h"
#include "vector3.h"

// Defines a sphere
typedef struct sphere_t {
	float center[3]; 	// Position of the object
	float radius;   	// Radius of the sphere
} sphere_t;

/******************************
* Reads sphere data from the specified file and sets specified sphere
* @param file   File to read sphere from
* @param sphere Sphere to store read data in
******************************/
void Sphere_readTo (FILE* file, sphere_t* sphere);

/******************************
* Finds the intersection between an sphere and a ray.
* @param ray    A pointer to a line_t object that has the ray equation.
* @param sphere A pointer to the sphere that will be tested for intersection.
******************************/
__device__ float Sphere_intersect (line_t * ray, sphere_t * sphere);

/******************************
* Finds the normal on the point of intersection.
* @param normal       A pointer to a normal vector that will be populated.
* @param sphere       A pointer to the sphere that is intersected.
* @param intersection A pointer to the intersection point.
******************************/
__device__ void Sphere_normal (float * normal, sphere_t * sphere,
	float * intersection);

#endif
