
// Describes a color in the raytracer system

#ifndef _h_color
#define _h_color

#include "vector3.h"

// Defines a color
#define COLOR unsigned char

#define CHANNELS 3
#define R 0
#define G 1
#define B 2

/******************************
* Copies components of one color to another color
* @param c1 Color to copy components to
* @param c2 Color to copy components from
******************************/
#define COLOR_COPY(c1, c2) \
	VECTOR_COPY(c1, c2)

#endif
