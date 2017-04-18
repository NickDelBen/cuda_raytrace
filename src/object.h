
// Describes an object in the raytracer system

#ifndef _h_object
#define _h_object

// Defines an object
typedef struct object_t {
	float pos[3];                // Position of the object
	unsigned short int color[3]; // Color of the object
} object_t;

/**************************
* Sets the properties of an object
* @param obj Object to set parameters of
* @param x   X componant of object position
* @param y   Y componant of object position
* @param z   Z componant of object position
* @param r   Red channel value of object color
* @param g   Green channel value of object color
* @param b   Blue channel value of object color
**************************/
void Object_setProps (object_t* obj, float x, float y, float z, unsigned short int r, unsigned short int g, unsigned short int b);

#endif
