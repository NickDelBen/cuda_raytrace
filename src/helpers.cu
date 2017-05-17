
#include "helpers.h"

// Find point of intersection
__device__ void findIntersectionPoint(float * intersection,
    line_t * ray, float distance)
{
    VECTOR_SCALE(intersection, ray->direction, distance);
    VECTOR_ADD(intersection, intersection, ray->position);
}

// Find the reflected ray.
__device__ void findReflectedRay(float * reflected, float * ray,
    float * normal)
{
    float c = fabs(2 * VECTOR_DOT(ray, normal));

    VECTOR_SCALE(reflected, normal, c);
    VECTOR_ADD(reflected, reflected, ray);
    Vector_normalize(reflected);
}