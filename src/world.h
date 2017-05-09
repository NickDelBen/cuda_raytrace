
// Describes the world in the raytracer system

#ifndef _h_world
#define _h_world

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "object.h"
#include "material.h"
#include "light.h"
#include "color.h"

// Defines a world
typedef struct world_t {
	COLOR bg[CHANNELS];       // Color of world background
	float global_ambient;     // Global ambient light
	unsigned int n_lights;    // Number of lights in the world
	unsigned int n_materials; // Number of materials in the world
	unsigned int n_objects;   // Number of spheres in the world
	light_t * lights;         // Lights in the world
	material_t * materials;   // Material details in the world
	object_t * objects;       // Objects in the world
} world_t;

/******************************
* Reads the details of a world from the specified file
* NOTE: Must free result with World_freeHost()
* @param file File to read world from
* @return pointer to location of new world on host
******************************/
world_t * World_read (FILE * file);

/******************************
* Copies the specified world to the device
* NOTE: Must free result with World_freeDevice()
* @param source World to copy to device
* @param size   A point to the integer that will be populated by the world size
* @return pointer to location of new world on device
******************************/
world_t * World_toDevice (world_t * source, int * size);

/******************************
* Copies the specified world to the device's shared memory
* @param smem   The shared memory location
* @param source World to copy to device's shared memory
* @return pointer to location of next available location in the shared memory
******************************/
__device__ world_t * World_toShared (void * smem, world_t * source);

/******************************
* Frees resources allocated for a world on the host
* @param world Pointer to location on host of world to free
******************************/
void World_freeHost (world_t * world);

/******************************
* Frees resources allocated for a world on the device
* @param world Pointer to location on device of World to free
******************************/
void World_freeDevice (world_t * world);

#endif
