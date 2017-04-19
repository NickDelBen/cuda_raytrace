
// Describes a line in the raytracer system

#ifndef _h_line
#define _h_line

#include <math.h>

#define LINE_LENGTH(X)  (sqrt(((x)[0]*(x)[0]) + ((x)[1]*(x)[1]) + ((x)[2]*(x)[2])))

// Defines a line
typedef struct line_t {
	float position[3]; 	// Position of the line
	float direction[3]; // Direction of line
} line_t;

#endif
