
// Describes a triangle in the raytracer system

#ifndef _h_helpers
#define _h_helpers

#include "line.h"
#include "vector3.h"

/******************************
* Finds the intersection point between ray and object.
* @param intersection A pointer to the intersection coordinate that will be
					  populated.
* @param ray          A pointer to a line_t object that has the ray equation.
* @param distance     The distance between the ray and the object.
******************************/
__device__ void findIntersectionPoint(float * intersection,
	line_t * ray, float distance);

/******************************
* Finds the direction for the reflected ray.
* @param reflected A pointer to the reflected ray that will be populated.
* @param ray       A pointer to the direction of the initial ray.
* @param normal    A pointer to the normal of the intersection.
******************************/
__device__ void findReflectedRay(float * reflected, float * ray,
	float * normal);

#endif
