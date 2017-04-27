
#include "sphere.h"

// Reads sphere data from the specified file and sets specified sphere
void Sphere_readTo (FILE* file, sphere_t* sphere)
{
	fscanf(file, "%f %f %f %f\n",
		&(sphere->center[0]), &(sphere->center[1]), &(sphere->center[2]),
		&(sphere->radius));
}
