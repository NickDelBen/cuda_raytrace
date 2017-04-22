
#include "camera.h"

// Reads the details of a camera from the specified file
camera_t* Camera_read (FILE* file)
{
	camera_t* result = (camera_t*) malloc(sizeof(camera_t));

	fscanf(file, "CAM %f %f %f %f %f %f %f %f %f %f %f %f\n", &(result->bottom_left[0]), &(result->bottom_left[1]), &(result->bottom_left[2]),
		&(result->top_left[0]), &(result->top_left[1]), &(result->top_left[2]),
		&(result->top_right[0]), &(result->top_right[1]), &(result->top_right[2]), 
		&(result->bottom_right[0]), &(result->bottom_right[1]), &(result->bottom_right[2]));

	// Get the direction vectors between the corners
	VECTOR_SUB(result->comp_vert, result->top_left, result->bottom_left);
	VECTOR_SUB(result->comp_horiz, result->bottom_right, result->bottom_left);
	// Store the height and width of imaging plane
	result->width = floor(VECTOR_LENGTH(result->comp_horiz));
	result->height = floor(VECTOR_LENGTH(result->comp_vert));

	return result;
}

// Copies the specified camera to the device
camera_t* Camera_toDevice (camera_t* source)
{
	camera_t* result;

	// Allocate space for camera on device
	cudaMalloc(&result, sizeof(camera_t));
	// Copy camera to device
	cudaMemcpy(result, source, sizeof(camera_t), cudaMemcpyHostToDevice);

	return result;
}

// Calculate the camera direction vectors and normal
void Camera_calculateVectors (camera_t* cam)
{
	// Get the direction vectors between the corners
	VECTOR_SUB(cam->comp_vert, cam->top_left, cam->bottom_left);
	VECTOR_SUB(cam->comp_horiz, cam->bottom_right, cam->bottom_left);
	// Turn the direction vectors into unit vectors
	Vector_normalize(cam->comp_vert);
	Vector_normalize(cam->comp_horiz);
	// Find the normal
	VECTOR_CROSSPRODUCT(cam->normal, cam->comp_vert, cam->comp_horiz);
}

// Frees resources allocated for a camera on the host
void Camera_freeHost (camera_t* camera)
{
	free(camera);
}

// Frees resources allocated for a camera on the device
void Camera_freeDevice (camera_t* camera)
{
	cudaFree(camera);
}
