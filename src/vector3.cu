
#include "vector3.h"

// Normalizes a vector
void Vector_normalize (float* vec)
{
	// Get length of vector
	float length = VECTOR_LENGTH(vec);
	// Scale the vector by length to normalize
	VECTOR_SCALE(vec, length);
}
