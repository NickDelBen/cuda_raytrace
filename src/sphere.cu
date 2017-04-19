
#include "sphere.h"

// Reads sphere data from the specified file and sets specified sphere
void Sphere_readTo (FILE* file, sphere_t* sphere)
{
	fscanf(file, "S %u %f %f %f %f\n", &(sphere->props.mat), 
		&(sphere->props.pos[0]), &(sphere->props.pos[1]), &(sphere->props.pos[2]), &(sphere->radius));
}
