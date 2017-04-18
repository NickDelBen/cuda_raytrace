
// Describes a light in the raytracer system

#ifndef _h_light
#define _h_light

// Defines a light
typedef struct light_t {
	float pos[3];                // Position of the light
	unsigned short int color[3]; // Color of the light
	float i;                     // Intensity of the light
} light_t;

#endif
