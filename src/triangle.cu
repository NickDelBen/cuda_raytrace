
#include "triangle.h"

// Reads triangle data from the specified file and sets specified triangle
void Triangle_readTo (FILE* file, triangle_t* triangle)
{
	fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
		&(triangle->points[0][0]), &(triangle->points[0][1]), &(triangle->points[0][2]),
		&(triangle->points[1][0]), &(triangle->points[1][1]), &(triangle->points[1][2]),
		&(triangle->points[2][0]), &(triangle->points[2][1]), &(triangle->points[2][2]));

	float u[3], v[3];
	VECTOR_SUB(u, triangle->points[1], triangle->points[0]);
    VECTOR_SUB(v, triangle->points[2], triangle->points[0]);
    VECTOR_CROSSPRODUCT(triangle->normal, u, v);
    Vector_normalize(triangle->normal);
}
