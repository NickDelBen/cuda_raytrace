
#include "material.h"

// Reads material data from the specified file and sets specified material
void Material_readTo (FILE* file, material_t* material)
{
	fscanf(file, "M %hu %hu %hu %f %f %f %f %f\n", &(material->color[0]), &(material->color[1]), &(material->color[2]), &(material->reflectivity), 
		&(material->specular_power), &(material->i_specular), &(material->i_diffuse), &(material->i_ambient));
}
