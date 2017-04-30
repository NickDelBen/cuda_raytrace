
// Describes an object in the raytracer system

#ifndef _h_object
#define _h_object

#include "sphere.h"
#include "triangle.h"

#define SPHERE 'S'
#define TRIANGLE 'T'

// Defines an object
typedef struct {
	char type; 					// Holds the type of the object
    union { 					// Object
        sphere_t sphere;
        triangle_t triangle;
    };
	unsigned int mat; 			// ID of material properties of object
} object_t;

/******************************
* Reads object data from the specified file and sets specified object
* @param file   File to read object from
* @param object Object to store read data in
******************************/
void Object_readTo (FILE * file, object_t * object);

/******************************
* Finds the intersection between an object and a ray.
* @param ray    A pointer to a line_t object that has the ray equation.
* @param object A pointer to the object that will be tested for intersection.
******************************/
__device__ float Object_intersect (line_t * ray, object_t * object);

/******************************
* Finds the normal on the point of intersection.
* @param normal       A pointer to a normal vector that will be populated.
* @param object       A pointer to the object that is intersected.
* @param intersection A pointer to the intersection point.
******************************/
__device__ void Object_normal (float * normal, object_t * object,
	float * intersection);

#endif
