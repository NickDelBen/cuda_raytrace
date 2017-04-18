
// Describes the world in the raytracer system

#ifndef _h_world
#define _h_world

#include <stdio.h>
#include <stdlib.h>

#include "material.h"
#include "light.h"
#include "plane.h"
#include "sphere.h"

// Defines a world
typedef struct world_t {
	unsigned short int bg[3]; // Color of world background
	float global_ambient;     // Global ambient light
	unsigned int n_lights;    // Number of lights in the world
	unsigned int n_materials; // Number of materials in the world
	unsigned int n_spheres;   // Number of spheres in the world
	unsigned int n_planes;    // Number of planes in the world
	light_t* lights;          // Lights in the world
	material_t* materials;    // Material details in the world
	sphere_t* spheres;        // Spheres in the world
	plane_t* planes;          // Planes in the world
} world_t;

/******************************
* Reads the details of a world from the specified file
* NOTE: Must free result with World_free()
* @param file File to read world from
******************************/
world_t* World_read(FILE* file);

#endif
