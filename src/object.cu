
#include "object.h"

// Reads object data from the specified file and sets specified object
void Object_readTo (FILE* file, object_t* object)
{
	fscanf(file, "%c %u ", &(object->type), &(object->mat));

	switch(object->type) {

		case SPHERE: {
			Sphere_readTo(file, object->sphere);
		} break;

		case TRIANGLE: {
			Triangle_readTo(file, object->triangle);
		} break;

	}
}