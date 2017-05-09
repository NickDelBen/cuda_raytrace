
#include "light.h"

// Reads light data from the specified file and sets specified light
void Light_readTo (FILE * file, light_t * light)
{
	fscanf(file,
		"POSITION (%f, %f, %f), "
		"COLOR (%hhu, %hhu, %hhu), "
		"INTENSITY %f\n",
		&(light->pos[X]), &(light->pos[Y]), &(light->pos[Z]), 
		&(light->color[R]), &(light->color[G]), &(light->color[B]),
		&(light->i));
}