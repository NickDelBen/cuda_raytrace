
#include "object.h"

// Reads object data from the specified file and sets specified object
void Object_readTo (FILE* file, object_t* object)
{
	fscanf(file, "%c %u ", &(object->type), &(object->mat));

	switch(object->type) {

		case SPHERE: {
			Sphere_readTo (file, &(object->sphere));
		} break;

		case TRIANGLE: {
			Triangle_readTo (file, &(object->triangle));
		} break;

	}
}

// Find intersection points between ray and object.
__device__ float Object_intersect (line_t * ray, object_t * object)
{
    switch(object->type) {

        case SPHERE: {
            return Sphere_intersect(ray, &(object->sphere));
        }

        case TRIANGLE: {
            return Triangle_intersect(ray, &(object->triangle));
        }

    }
    return NAN;
}

// Find the normal at the intersection point.
__device__ void Object_normal (float * normal, object_t * object,
	float * intersection)
{
    switch(object->type) {

        case SPHERE: {
            Sphere_normal(normal, &(object->sphere), intersection);
        } break;

        case TRIANGLE: {
            VECTOR_COPY(normal, object->triangle.normal);
        } break;

    }
}
