
#include "sphere.h"

// Reads sphere data from the specified file and sets specified sphere
void Sphere_readTo (FILE * file, sphere_t * sphere)
{
	fscanf(file, "CENTER (%f, %f, %f), "
        "RADIUS %f\n",
		&(sphere->center[X]), &(sphere->center[Y]), &(sphere->center[Z]),
		&(sphere->radius));
}

// Find intersection points between ray and sphere.
__device__ float Sphere_intersect (line_t * ray, sphere_t * sphere)
{
    float * center = sphere->center,
    	  radius = sphere->radius,
    	  temp[DSPACE], b, c, d, sqrtd;

    //temp = ray origin - center
    VECTOR_SUB(temp, ray->position, center);

    //b = 2 * ray direction . (ray origin - sphere center)
    b = 2 * VECTOR_DOT(ray->direction, temp);

    //c = (ray origin - sphere center) . (ray origin - sphere center)
    // - radius^2
    c = VECTOR_DOT(temp, temp) - pow(radius, 2);

    //d = b^2 - 4 * a * c, a = 1
    d = pow(b, 2) - 4 * c;

    if (d < 0) {
        return NAN;
    } else if (d == 0) {
        return -b / 2;
    } else {
        sqrtd = sqrt(d);
        return fmin(
            (-b + sqrtd) / 2,
            (-b - sqrtd) / 2
        );
    }
}

// Find the normal at the intersection point.
__device__ void Sphere_normal (float * normal, sphere_t * sphere,
    float * intersection)
{
    VECTOR_SUB(normal, intersection, sphere->center);
    Vector_normalize(normal);
}