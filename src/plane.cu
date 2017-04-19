
#include "plane.h"

// Reads plane data from the specified file and sets specified plane
void Plane_readTo (FILE* file, plane_t* plane)
{
	fscanf(file, "P %u %f %f %f %f %f %f\n", &(plane->props.mat), &(plane->props.pos[0]), &(plane->props.pos[1]), 
		&(plane->props.pos[2]), &(plane->normal[0]), &(plane->normal[1]), &(plane->normal[2]));
}
