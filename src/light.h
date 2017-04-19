
// Describes a light in the raytracer system

#ifndef _h_light
#define _h_light

#include <stdio.h>

// Defines a light
typedef struct light_t {
	float pos[3];                // Position of the light
	unsigned short int color[3]; // Color of the light
	float i;                     // Intensity of the light
} light_t;

/******************************
* Reads light data from the specified file and sets specified light
* @param file  File to read light from
* @param light Light to store read data in
******************************/
void Light_readTo (FILE* file, light_t* light);

#endif
