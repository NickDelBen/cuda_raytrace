
// Describes a triangle in the raytracer system

#ifndef _h_triangle
#define _h_triangle

#include <stdio.h>

#include "vector3.h"

// Defines a triangle
typedef struct triangle_t {
	float points[3][3];
	float normal[3];
} triangle_t;

/******************************
* Reads triangle data from the specified file and sets specified triangle
* @param file   File to read triangle from
* @param triangle triangle to store read data in
******************************/
void Triangle_readTo (FILE* file, triangle_t* triangle);

#endif
