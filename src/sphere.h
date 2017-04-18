
// Describes a sphere in the raytracer system

#ifndef _h_sphere
#define _h_sphere

#include "object.h"

// Defines a sphere
typedef struct sphere_t {
	object_t props; // Object properties of the sphere
	float radius;   // Radius of the sphere
} sphere_t;

#endif
