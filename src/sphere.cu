
#include "sphere.h"

// Reads sphere data from the specified file and sets specified sphere
void Sphere_readTo (FILE * file, sphere_t * s)
{
	fscanf(file, 
        "CENTER (%f, %f, %f), "
        "RADIUS %f\n",
		&(s->center[X]), &(s->center[Y]), &(s->center[Z]),
		&(s->radius));
}

// Find intersection points between ray and sphere.
__device__ float Sphere_intersect (line_t * ray, sphere_t * s)
{
    float temp[DSPACE], b, c, d, sqrtd;

    //temp = ray origin - center
    VECTOR_SUB(temp, ray->position, s->center);

    //b = 2 * ray direction . (ray origin - sphere center)
    b = 2 * VECTOR_DOT(ray->direction, temp);

    //c = (ray origin - sphere center) . (ray origin - sphere center)
    // - radius^2
    c = VECTOR_DOT(temp, temp) - pow(s->radius, 2);

    //d = b^2 - 4 * a * c, a = 1
    d = pow(b, 2) - 4 * c;

    if (d < 0) {
        return NAN;
    } else if (d == 0) {
        return -b / 2;
    } else {
        sqrtd = sqrtf(d);
        c = (-b + sqrtd) / 2;
        d = (-b - sqrtd) / 2;
        return fabs(c) > fabs(d) ? d : c;
    }
}

// Find the normal at the intersection point.
__device__ void Sphere_normal (float * normal, sphere_t * s,
    float * intersection)
{
    VECTOR_SUB(normal, intersection, s->center);
    Vector_normalize(normal);
}