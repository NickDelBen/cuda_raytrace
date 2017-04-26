
#include "frame.h"

// Initialize the frame
__global__ void Frame_init (color_t *frame, int size)
{
	int b_work 		= size / gridDim.x,
		t_work 		= b_work / blockDim.x,
		t_offset 	= threadIdx.x * t_work,
		offset 		= blockIdx.x * b_work + t_offset;

	extern __shared__ color_t thread_pixles[];

	for (int i = t_offset; i < t_offset + t_work; ++i) {
		thread_pixles[i].r = 0;
		thread_pixles[i].g = 0;
		thread_pixles[i].b = 0;
	}

	memcpy(&frame[offset], &thread_pixles[t_offset], sizeof(color_t) * t_work);
}
