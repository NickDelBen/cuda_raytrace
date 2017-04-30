
// Describes a line in the raytracer system

#ifndef _h_vector3
#define _h_vector3

#include <math.h>

#define DSPACE 3
#define X 0
#define Y 1
#define Z 2

/******************************
* Adds the compononents of a vector to the specified vector
* @param w Location to store u + v
* @param u First Vector
* @param v Second Vector
******************************/
#define VECTOR_ADD(w, u, v)   \
	(w)[X] = (u)[X] + (v)[X]; \
	(w)[Y] = (u)[Y] + (v)[Y]; \
	(w)[Z] = (u)[Z] + (v)[Z];

/******************************
* Subtracts two vectors and stores result in a third vector
* @param w Location to store result of u - v
* @param u First vector
* @param v Second vector
******************************/
#define VECTOR_SUB(w, u, v)   \
	(w)[X] = (u)[X] - (v)[X]; \
	(w)[Y] = (u)[Y] - (v)[Y]; \
	(w)[Z] = (u)[Z] - (v)[Z];

/******************************
* Returns the dot product of two the vectors
* @param u First vector
* @param v Seconf vector
******************************/
#define VECTOR_DOT(u, v) \
	((u)[X]*(v)[X]) +    \
	((u)[Y]*(v)[Y]) +    \
	((u)[Z]*(v)[Z]);

/******************************
* Stores the cross product of two vectors in a third vector
* @param w Location to store u x v
* @param u First vector
* @param v Seconf vector
******************************/
#define VECTOR_CROSSPRODUCT(w, u, v)            \
	(w)[X] = ((u)[Y]*(v)[Z]) - ((u)[Z]*(v)[Y]); \
	(w)[Y] = ((u)[Z]*(v)[X]) - ((u)[X]*(v)[Z]); \
	(w)[Z] = ((u)[X]*(v)[Y]) - ((u)[Y]*(v)[X]);

/******************************
* Calculates the length of the vector
* @param u Vector whos length is to be calculated.
******************************/
#define VECTOR_LENGTH(u)  \
	sqrt(                 \
		((u)[X]*(u)[X]) + \
		((u)[Y]*(u)[Y]) + \
		((u)[Z]*(u)[Z])   \
	)

/******************************
* Scales a vector by the specified scaler
* @param u Vector to scale
* @param n Amount so scale vector by
******************************/
#define VECTOR_SCALE_2(u, n)  \
	(u)[X] *= (n);            \
	(u)[Y] *= (n);            \
	(u)[Z] *= (n);

/******************************
* Scales a vector by the specified scaler
* @param u Vector to save result in
* @param v Vector to scale
* @param n Amount so scale vector by
******************************/
#define VECTOR_SCALE_3(u, v, n)  \
	(u)[X] = (v)[X] * (n);       \
	(u)[Y] = (v)[Y] * (n);       \
	(u)[Z] = (v)[Z] * (n);

/******************************
* VECTOR_SCALE helper.
******************************/
#define VECTOR_SCALE_SELECTOR(_1, _2, _3, NAME, ...) NAME

/******************************
* Scales a vector by the specified scaler
******************************/
#define VECTOR_SCALE(...)					\
	VECTOR_SCALE_SELECTOR 					\
	(__VA_ARGS__, VECTOR_SCALE_3, VECTOR_SCALE_2)(__VA_ARGS__)

/******************************
* Copies components of one vector to another vector
* @param u Vector to copy components to
* @param v Vector to copy components from
******************************/
#define VECTOR_COPY(u, v) \
	(u)[X] = (v)[X];      \
	(u)[Y] = (v)[Y];      \
	(u)[Z] = (v)[Z];

/******************************
* Normalizes a vector
* @param vec Vector to normalize
******************************/
__host__ __device__ void Vector_normalize (float * vec);

#endif
