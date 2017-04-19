
// Describes a plane in the raytracer system

#ifndef _h_plane
#define _h_plane

#include <stdio.h>

#include "object.h"

// Defines a plane
typedef struct plane_t {
	object_t props;  // Object properties of the plane
	float normal[3]; // Normal to the plane
} plane_t;

/******************************
* Reads plane data from the specified file and sets specified plane
* @param file  File to read plane from
* @param plane Plane to store read data in
******************************/
void Plane_readTo (FILE* file, plane_t* plane);

#endif
