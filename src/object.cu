
#include "object.h"

// Sets the properties of an object
void Object_setProps (object_t* obj, float x, float y, float z, unsigned short int r, unsigned short int g, unsigned short int b)
{
	// Set the position
	obj->pos[0] = x;
	obj->pos[1] = y;
	obj->pos[2] = z;
	// Set the color
	obj->color[0] = r;
	obj->color[1] = g;
	obj->color[2] = b;
}
