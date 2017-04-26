
// Describes a color in the raytracer system

#ifndef _h_color
#define _h_color

// Defines a color
typedef struct {
	unsigned short int r, g, b;
} color_t;

/******************************
* Adds the compononents of a color to the specified color
* @param C Location to store C1 + C2
* @param C1 First color
* @param C2 Second color
******************************/
#define COLOR_ADD(C, C1, C2)   \
	(C)->r = (C1)->r + (C2)->r; \
	(C)->g = (C1)->g + (C2)->g; \
	(C)->b = (C1)->b + (C2)->b;

#endif
