
// Describes a line in the raytracer system

#ifndef _h_line
#define _h_line

#include "point.h"
#include "vector3.h"

// Defines a line
typedef struct line_t {
	POINT position[DSPACE];  // Position of the line
	float direction[DSPACE]; // Direction of line
} line_t;

#endif
