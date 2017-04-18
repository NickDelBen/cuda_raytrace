
#include "world.h"

// Reads the details of a world from the specified file
world_t* World_read(FILE* file)
{
	// Allocate space for the result
	world_t* result = (world_t*) malloc(sizeof(world_t));
	// Read background color of the world
	fscanf(file, "BG %hu %hu %hu\n", &(result->bg[0]), &(result->bg[1]), &(result->bg[2]));
	// Read global ambient brightness
	fscanf(file, "AMB %f\n", &(result->global_ambient));

	return result;
}
