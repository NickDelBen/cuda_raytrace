
#include <stdio.h>

#include "world.h"

int main(int argc, char ** argv)
{
	FILE* fp = fopen("world.txt", "r");
	world_t* w = World_read(fp);
	fclose(fp);

	printf("Read world background=(%hu, %hu, %hu)   gloabl_ambient=%f\n", w->bg[0], w->bg[1], w->bg[2], w->global_ambient);

	return 0;
}

