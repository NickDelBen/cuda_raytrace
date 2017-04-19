
#include "light.h"

// Reads light data from the specified file and sets specified light
void Light_readTo (FILE* file, light_t* light)
{
	fscanf(file, "L %f %f %f %hu %hu %hu %f\n", &(light->pos[0]), &(light->pos[1]), &(light->pos[2]), 
		&(light->color[0]), &(light->color[1]), &(light->color[2]), &(light->i));
}
