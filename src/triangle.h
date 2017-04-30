
// Describes a triangle in the raytracer system

#ifndef _h_triangle
#define _h_triangle

#include <stdio.h>

#include "line.h"
#include "vector3.h"
#include "helpers.h"

#define EPSILON 0.00001
#define U 0
#define V 1
#define W 2

// Defines a triangle
typedef struct triangle_t {
	float points[3][DSPACE];
	float normal[DSPACE];
} triangle_t;

/******************************
* Reads triangle data from the specified file and sets specified triangle
* @param file   File to read triangle from
* @param triangle triangle to store read data in
******************************/
void Triangle_readTo (FILE * file, triangle_t * triangle);

/******************************
* Finds the intersection between an triangle and a ray.
* @param ray      A pointer to a line_t object that has the ray equation.
* @param triangle A pointer to the triangle that will be tested for intersection.
******************************/
__device__ float Triangle_intersect (line_t * ray, triangle_t * triangle);

#endif
