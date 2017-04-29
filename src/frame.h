
// Describes the frame in the raytracer system

#ifndef _h_frame
#define _h_frame

#include <stdio.h>

#include "color.h"

/******************************
* Initializes a frame of color_t objects on the device
* @param frame A pointer to the color_t array on the device
* @param size  Number of pixels in a frame
******************************/
__global__ void Frame_init (color_t* frame, int size);

#endif
