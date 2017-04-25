
// Ray tracer driver

#ifndef _h_driver
#define _h_driver

#include <stdio.h>

#include "camera.h"
#include "world.h"
#include "line.h"
#include "frame.h"

#define BLOCKS 32
#define THREADS 256

/******************************
* Traces the rays from the eye position into the scene
* @param f Frame memory location for storing the frame on the host
* @param d_f Frame memory location for storing the frame on the device
* @param d_r A pointer to the camera rays on the device
* @param w A pointer to the world object
* @param size The size of the portview panel.
******************************/
void trace(color_t *f, color_t * d_f, line_t *d_r, world_t *w, int size);

#endif
