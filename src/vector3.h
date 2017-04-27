
// Describes a line in the raytracer system

#ifndef _h_vector3
#define _h_vector3

#include <math.h>

// Macro to calculate length og a line
#define VECTOR_LENGTH(X)  \
	sqrt(                 \
		((X)[0]*(X)[0]) + \
		((X)[1]*(X)[1]) + \
		((X)[2]*(X)[2])   \
	)

/******************************
* Subtracts two vectors and stores result in a third vector
* @param R Location to store result of A - B
* @param A First vector
* @param B Second vector
******************************/
#define VECTOR_SUB(R, A, B)   \
	(R)[0] = (A)[0] - (B)[0]; \
	(R)[1] = (A)[1] - (B)[1]; \
	(R)[2] = (A)[2] - (B)[2];

/******************************
* Scales a vector by the specified scaler
* @param V Vector to scale
* @param N Amount so scale vector by
******************************/
#define VECTOR_SCALE(V, N)  \
	(V)[0] *= (N);          \
	(V)[1] *= (N);          \
	(V)[2] *= (N);

/******************************
* Returns the dot product of two the vectors
* @param A First vector
* @param B Seconf vector
******************************/
#define VECTOR_DOT(A, B)            \
	((A)[0]*(B)[0]) + ((A)[1]*(B)[1]) + ((A)[2]*(B)[2]);

/******************************
* Stores the cross product of two vectors in a third vector
* @param R Location to store A x B
* @param A First vector
* @param B Seconf vector
******************************/
#define VECTOR_CROSSPRODUCT(R, A, B)            \
	(R)[0] = ((A)[1]*(B)[2]) - ((A)[2]*(B)[1]); \
	(R)[1] = ((A)[2]*(B)[0]) - ((A)[0]*(B)[2]); \
	(R)[2] = ((A)[0]*(B)[1]) - ((A)[1]*(B)[0]);

/******************************
* Adds the compononents of a vector to the specified vector
* @param R Location to store A + B
* @param A First Vector
* @param B Second Vector
******************************/
#define VECTOR_ADD(R, A, B)   \
	(R)[0] = (A)[0] + (B)[0]; \
	(R)[1] = (A)[1] + (B)[1]; \
	(R)[2] = (A)[2] + (B)[2];

/******************************
* Copies components of one vector to another vector
* @param R Vector to copy components to
* @param T Vector to copy components from
******************************/
#define VECTOR_COPY(R, T) \
	(R)[0] = (T)[0];      \
	(R)[1] = (T)[1];      \
	(R)[2] = (T)[2];

/******************************
* Normalizes a vector
* @param vec Vector to normalize
******************************/
void Vector_normalize (float* vec);

#endif
