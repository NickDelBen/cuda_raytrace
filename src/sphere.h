
// Describes a sphere in the raytracer system

#ifndef _h_sphere
#define _h_sphere

#include <stdio.h>

#include "object.h"

// Defines a sphere
typedef struct sphere_t {
	object_t props; // Object properties of the sphere
	float radius;   // Radius of the sphere
} sphere_t;

/******************************
* Reads sphere data from the specified file and sets specified sphere
* @param file   File to read sphere from
* @param sphere Sphere to store read data in
******************************/
void Sphere_readTo (FILE* file, sphere_t* sphere);

#endif
