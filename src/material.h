
// Describes a material in the raytracer system

#ifndef _h_material
#define _h_material

#include <stdio.h>

#include "color.h"

// Defines a material
typedef struct material_t {
	COLOR color[CHANNELS]; // Color of the material
	float reflectivity;    // Reflectivity of material
	float specular_power;  // Power of specular reaction
	float i_specular;      // Intensity of specular reaction
	float i_diffuse;       // Intensity of diffuse reaction
	float i_ambient;       // Intensity of ambient reaction
} material_t;

/******************************
* Reads material data from the specified file and sets specified material
* @param file     File to read material from
* @param material Material to store read data in
******************************/
void Material_readTo (FILE * file, material_t * material);

#endif
