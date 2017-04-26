
#include "world.h"

// Reads the details of a world from the specified file
world_t* World_read (FILE* file)
{
	int item_i;
	// Allocate space for the result
	world_t* result = (world_t*) malloc(sizeof(world_t));

	// Read background color of the world
	fscanf(file, "BG %hu %hu %hu\n", &(result->bg.r), &(result->bg.g), &(result->bg.b));
	// Read global ambient brightness
	fscanf(file, "AMB %f\n", &(result->global_ambient));

	// Read the lights
	fscanf(file, "LIGHTS %u\n", &(result->n_lights));
	result->lights = (light_t*) malloc(sizeof(light_t) * result->n_lights);
	for (item_i = 0; item_i < result->n_lights; item_i++) {
		Light_readTo(file, &(result->lights[item_i]));
	}

	// Read the materials
	fscanf(file, "MATERIALS %u\n", &(result->n_materials));
	result->materials = (material_t*) malloc(sizeof(material_t) * result->n_materials);
	for (item_i = 0; item_i < result->n_materials; item_i++) {
		Material_readTo(file, &(result->materials[item_i]));
	}

	// Read the spheres
	fscanf(file, "SPHERES %u\n", &(result->n_spheres));
	result->spheres = (sphere_t*) malloc(sizeof(sphere_t) * result->n_spheres);
	for (item_i = 0; item_i < result->n_spheres; item_i++) {
		Sphere_readTo(file, &(result->spheres[item_i]));
	}

	// Read the planes
	fscanf(file, "PLANES %u\n", &(result->n_planes));
	result->planes = (plane_t*) malloc(sizeof(plane_t) * result->n_planes);
	for (item_i = 0; item_i < result->n_planes; item_i++) {
		Plane_readTo(file, &(result->planes[item_i]));
	}

	return result;
}

// Copies the specified world to the device
world_t* World_toDevice (world_t* source)
{
	world_t* final;
	world_t* result;

	// Create temporary data to correct pointers on device
	result = (world_t*) malloc(sizeof(world_t));
	memcpy(result, source, sizeof(world_t));

	// Allocare space for the world objects
	cudaMalloc(&(result->lights), sizeof(light_t) * source->n_lights);
	cudaMalloc(&(result->materials), sizeof(material_t) * source->n_materials);
	cudaMalloc(&(result->spheres), sizeof(sphere_t) * source->n_spheres);
	cudaMalloc(&(result->planes), sizeof(plane_t) * source->n_planes);
	// Copy the world object data to the device
	cudaMemcpy(result->lights, source->lights, sizeof(light_t) * source->n_lights, cudaMemcpyHostToDevice);
	cudaMemcpy(result->materials, source->materials, sizeof(material_t) * source->n_materials, cudaMemcpyHostToDevice);
	cudaMemcpy(result->spheres, source->spheres, sizeof(sphere_t) * source->n_spheres, cudaMemcpyHostToDevice);
	cudaMemcpy(result->planes, source->planes, sizeof(plane_t) * source->n_planes, cudaMemcpyHostToDevice);

	// Allocate space for the world on the device
	cudaMalloc(&final, sizeof(world_t));
	// Copy the world data to the device
	cudaMemcpy(final, result, sizeof(world_t), cudaMemcpyHostToDevice);
	// Free the resources allocated for the temporary result
	free(result);

	return final;
}

// Frees resources allocated for a world on the host
void World_freeHost (world_t* world)
{
	// Free memory allocated for lights
	free(world->lights);
	// Free memory allocated for materials
	free(world->materials);
	// Free memory allocated for spheres
	free(world->spheres);
	// Free memory allocated for planes
	free(world->planes);
	// Free memory allocated for world object
	free(world);
}

// Frees resources allocated for a world on the device
void World_freeDevice (world_t* world)
{
	// Copy the world object back to host so we can read array locations
	world_t* temp = (world_t*) malloc(sizeof(world_t));
	cudaMemcpy(&temp, &world, sizeof(world_t), cudaMemcpyDeviceToHost);
	// Free memory allocated for lights
	cudaFree(temp->lights);
	// Free memory allocated for materials
	cudaFree(temp->materials);
	// Free memory allocated for spheres
	cudaFree(temp->spheres);
	// Free memory allocated for planes
	cudaFree(temp->planes);
	// Free memory allocated for world object
	cudaFree(world);
	// Free temporary memory to get device address locations
	free(temp);
}
