
// Describes a light in the raytracer system

#ifndef _h_light
#define _h_light

#include <stdio.h>

#include "vector3.h"
#include "color.h"

// Defines a light
typedef struct light_t {
	float pos[DSPACE];     // Position of the light
	COLOR color[CHANNELS]; // Color of the light
	float i;               // Intensity of the light
} light_t;

/******************************
* Reads light data from the specified file and sets specified light
* @param file  File to read light from
* @param light Light to store read data in
******************************/
void Light_readTo (FILE * file, light_t * light);

#endif
