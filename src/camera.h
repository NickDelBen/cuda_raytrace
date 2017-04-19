
// Describes a camera in the raytracer system

#ifndef _h_camera
#define _h_camera

#include <stdio.h>
#include <stdlib.h>

#include "line.h"

// Defines a camera
typedef struct camera_t {
	float position[3];    // Position of camera eye
	float top_right[3];   // First corner of imaging plane
	float bottom_left[3]; // Second corner of imaging plane
} camera_t;

/******************************
* Reads the details of a camera from the specified file
* NOTE: Must free result with Camera_freeHost()
* @param file File to read camera from
* @return pointer to location of new camera on host
******************************/
camera_t* Camera_read (FILE* file);

/******************************
* Copies the specified camera to the device
* NOTE: Must free result with Camera_freeDevice()
* @param source Camera on host to copy to device
* @return pointer to location of new camera on device
******************************/
camera_t* Camera_toDevice (camera_t* source);

/******************************
* Creates the rays from a camera on the device
* @param camera Camera on device to create rays for
* @param rays   Pointer to memory allocated on device for rays
******************************/
__global__ void Camera_createRays (camera_t* camera, line_t* rays);

/******************************
* Frees resources allocated for a camera on the host
* @param camera Camera to free resources for
******************************/
void Camera_freeHost (camera_t* camera);

/******************************
* Frees resources allocated for a camera on the device
* @param camera Camera to free resources for
******************************/
void Camera_freeDevice (camera_t* camera);

#endif
