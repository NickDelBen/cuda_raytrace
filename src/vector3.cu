
#include "vector3.h"

// Normalizes a vector
__host__ __device__ void Vector_normalize (float* vec)
{
	// Get length of vector
	float length = VECTOR_LENGTH(vec);
	// Scale the vector by length to normalize
	VECTOR_SCALE(vec, 1.0 / length);
}
