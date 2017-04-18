
// Describes a plane in the raytracer system

#ifndef _h_plane
#define _h_plane

#include "object.h"

// Defines a plane
typedef struct plane_t {
	object_t props;  // Object properties of the plane
	float normal[3]; // Normal to the plane
} plane_t;

#endif
