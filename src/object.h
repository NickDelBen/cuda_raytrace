
// Describes an object in the raytracer system

#ifndef _h_object
#define _h_object

#include "material.h"

// Defines an object
typedef struct object_t {
	float pos[3];     // Position of the object
	unsigned int mat; // ID of material properties of object
} object_t;

#endif
