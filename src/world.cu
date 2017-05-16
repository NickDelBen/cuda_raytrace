
#include "world.h"

// Reads the details of a world from the specified file
world_t * World_read (FILE * file)
{
	int item_i;
	// Allocate space for the result
	world_t * result = (world_t *) malloc(sizeof(world_t));

	// Read background color of the world
	fscanf(file, "BACKGROUND (%hhu, %hhu, %hhu)\n",
		&(result->bg[R]), &(result->bg[G]), &(result->bg[B]));
	// Read global ambient brightness
	fscanf(file, "AMBIENT %f\n", &(result->global_ambient));

	// Read the lights
	fscanf(file, "LIGHTS %u\n", &(result->n_lights));
	result->lights = (light_t *) malloc(sizeof(light_t) * result->n_lights);
	for (item_i = 0; item_i < result->n_lights; item_i++) {
		Light_readTo(file, &(result->lights[item_i]));
	}

	// Read the materials
	fscanf(file, "MATERIALS %u\n", &(result->n_materials));
	result->materials = (material_t *)malloc(sizeof(material_t) * result->n_materials);
	for (item_i = 0; item_i < result->n_materials; item_i++) {
		Material_readTo(file, &(result->materials[item_i]));
	}

	// Read the objects
	fscanf(file, "OBJECTS %u\n", &(result->n_objects));
	result->objects = (object_t *) malloc(sizeof(object_t) * result->n_objects);
	for (item_i = 0; item_i < result->n_objects; item_i++) {
		Object_readTo(file, &(result->objects[item_i]));
	}

	return result;
}

// Copies the specified world to the device
world_t * World_toDevice (world_t * source, int * size)
{
	world_t * final, 
			* result;

	int w_size = sizeof(world_t),
		l_size = sizeof(light_t) * source->n_lights,
		m_size = sizeof(material_t) * source->n_materials,
		o_size = sizeof(object_t) * source->n_objects;

	// Create temporary data to correct pointers on device
	result = (world_t *)malloc(w_size);
	memcpy(result, source, w_size);

	// Allocare space for the world objects
	cudaMalloc(&(result->lights), l_size);
	cudaMalloc(&(result->materials), m_size);
	cudaMalloc(&(result->objects), o_size);
	// Copy the world object data to the device
	cudaMemcpy(result->lights, source->lights, l_size, cudaMemcpyHostToDevice);
	cudaMemcpy(result->materials, source->materials, m_size, cudaMemcpyHostToDevice);
	cudaMemcpy(result->objects, source->objects, o_size, cudaMemcpyHostToDevice);

	// Allocate space for the world on the device
	cudaMalloc(&final, sizeof(world_t));
	// Copy the world data to the device
	cudaMemcpy(final, result, sizeof(world_t), cudaMemcpyHostToDevice);
	// Free the resources allocated for the temporary result
	free(result);

	*size = w_size + l_size + m_size + o_size;
	return final;
}

// Copies the specified world to the device's shared memory
__device__ world_t * World_toShared (void * smem, world_t * source)
{
	uint8_t * world_bytes = (uint8_t *) smem;
	int w_size = sizeof(world_t),
		l_size = sizeof(light_t) * source->n_lights,
		m_size = sizeof(material_t) * source->n_materials,
		o_size = sizeof(object_t) * source->n_objects;

	world_t * result = (world_t *) world_bytes;

	memcpy(world_bytes, source, w_size);

	// Set memory addresses
	result->lights = (light_t *)(world_bytes + w_size);
	result->materials = (material_t *)(world_bytes + w_size + l_size);
	result->objects = (object_t *)(world_bytes + w_size + l_size + m_size);

	// Copy the world object data to the device
	memcpy(result->lights, source->lights, l_size);
	memcpy(result->materials, source->materials, m_size);
	memcpy(result->objects, source->objects, o_size);

	return result;
}

// Frees resources allocated for a world on the host
void World_freeHost (world_t * world)
{
	// Free memory allocated for lights
	free(world->lights);
	// Free memory allocated for materials
	free(world->materials);
	// Free memory allocated for objects
	free(world->objects);
	// Free memory allocated for world object
	free(world);
}

// Frees resources allocated for a world on the device
void World_freeDevice (world_t * world)
{
	// Copy the world object back to host so we can read array locations
	world_t * temp = (world_t *) malloc(sizeof(world_t));
	cudaMemcpy(temp, world, sizeof(world_t), cudaMemcpyDeviceToHost);
	// Free memory allocated for lights
	cudaFree(temp->lights);
	// Free memory allocated for materials
	cudaFree(temp->materials);
	// Free memory allocated for objects
	cudaFree(temp->objects);
	// Free memory allocated for world object
	cudaFree(world);
	// Free temporary memory to get device address locations
	free(temp);
}
