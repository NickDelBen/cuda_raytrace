
#include "helpers.h"

// Find point of intersection
__device__ void findIntersectionPoint(float * intersection,
    line_t * ray, float distance)
{
    VECTOR_COPY(intersection, ray->direction);
    VECTOR_SCALE(intersection, distance);
    VECTOR_ADD(intersection, intersection, ray->position);
}

// Find the reflected ray.
__device__ void findReflectedRay(float * reflected, float * ray,
    float * normal)
{
    float c = 2 * VECTOR_DOT(ray, normal);

    VECTOR_COPY(reflected, normal);
    VECTOR_SCALE(reflected, c);
    VECTOR_SUB(reflected, reflected, ray);
    VECTOR_SCALE(reflected, 1.0 / VECTOR_LENGTH(reflected));
}