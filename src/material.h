
// Describes a material in the raytracer system

#ifndef _h_material
#define _h_material

// Defines a material
typedef struct material_t {
	unsigned short int color[3]; // Color of the material
	float reflectivity;          // Reflectivity of material
	float specular_power;        // Power of specular reaction
	float intensity_specular;    // Intensity of specular reaction
	float intensity_diffuse;     // Intensity of diffuse reaction
	float intensity_ambient;     // Intensity of ambient reaction
} material_t;

#endif
