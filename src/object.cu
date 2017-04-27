
#include "object.h"

// Reads object data from the specified file and sets specified object
void Object_readTo (FILE* file, object_t* object)
{
	fscanf(file, "%c %u ", &(object->type), &(object->mat));

	switch(object->type) {

		case SPHERE: {
			printf("Reading sphere\n");
			Sphere_readTo (file, object->sphere);
			printf("Read sphere\n");
		} break;

		case TRIANGLE: {
			printf("Read triangle\n");
			Triangle_readTo (file, object->triangle);
		} break;

	}
}