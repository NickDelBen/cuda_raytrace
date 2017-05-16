
// Describes a color in the raytracer system

#ifndef _h_color
#define _h_color

#include "vector3.h"

// Defines a color
#define COLOR unsigned char

#define CHANNELS 3
#define R 0
#define G 1
#define B 2
#define COLOR_MAX 255

/******************************
* Copies components of one color to another color
* @param c1 Color to copy components to
* @param c2 Color to copy components from
******************************/
#define COLOR_COPY(c1, c2) \
	VECTOR_COPY(c1, c2)

/******************************
* Adds two colors
* @param c  Color to save components in
* @param c1 First color
* @param c2 Second color
******************************/
#define COLOR_ADD(c, c1, c2) \
	(c)[R] = (COLOR)min(((int)(c1)[R] + (int)(c2)[R]), COLOR_MAX); \
	(c)[G] = (COLOR)min(((int)(c1)[G] + (int)(c2)[G]), COLOR_MAX); \
	(c)[B] = (COLOR)min(((int)(c1)[B] + (int)(c2)[B]), COLOR_MAX);

/******************************
* Scales a color by the specified scaler
* @param u color to scale
* @param n Amount so scale color by
******************************/
#define COLOR_SCALE_2(c, n)  								\
	(c)[R] = (COLOR)min((float)((c)[R]) * (n), COLOR_MAX);  \
	(c)[G] = (COLOR)min((float)((c)[G]) * (n), COLOR_MAX);  \
	(c)[B] = (COLOR)min((float)((c)[B]) * (n), COLOR_MAX);

/******************************
* Scales a color by the specified scaler
* @param c color to save result in
* @param c1 color to scale
* @param n Amount so scale color by
******************************/
#define COLOR_SCALE_3(c, c1, n)  							\
	(c)[R] = (COLOR)min((float)((c1)[R]) * (n), COLOR_MAX); \
	(c)[G] = (COLOR)min((float)((c1)[G]) * (n), COLOR_MAX); \
	(c)[B] = (COLOR)min((float)((c1)[B]) * (n), COLOR_MAX);

/******************************
* VECTOR_SCALE helper.
******************************/
#define COLOR_SCALE_SELECTOR(_1, _2, _3, NAME, ...) NAME

/******************************
* Scales a vector by the specified scaler
******************************/
#define COLOR_SCALE(...)					\
	COLOR_SCALE_SELECTOR 					\
	(__VA_ARGS__, COLOR_SCALE_3, COLOR_SCALE_2)(__VA_ARGS__)

#endif
