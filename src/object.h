
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
        sphere_t *sphere;
        triangle_t *triangle;
    };
	unsigned int mat; 			// ID of material properties of object
} object_t;

/******************************
* Reads object data from the specified file and sets specified object
* @param file   File to read object from
* @param object Object to store read data in
******************************/
void Object_readTo (FILE* file, object_t* object);

#endif
