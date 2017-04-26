
// Describes a camera in the raytracer system

#ifndef _h_camera
#define _h_camera

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "line.h"
#include "vector3.h"

// Defines a camera
typedef struct camera_t {
	float bottom_left[3];  // Second corner of imaging plane
	float top_left[3];     // Top left of imaging plane
	float top_right[3];    // First corner of imaging plane
	float bottom_right[3]; // Bottom right of imaging plane
	float height;          // Height of imaging plane view box
	float width;           // Width of imaging plane view box
	float comp_vert[3];    // Vertical unit component of imaging plane
	float comp_horiz[3];   // Horizontal component of imaging plane
	float normal[3];       // Normal to imaging plane
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
* Calculate the camera direction vectors and normal
* @param cam Camera to calculate direction vectors for
******************************/
void Camera_calculateVectors (camera_t* cam);

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
