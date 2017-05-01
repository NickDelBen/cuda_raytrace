
#include "material.h"

// Reads material data from the specified file and sets specified material
void Material_readTo (FILE * file, material_t * material)
{
	fscanf(file, 
		"COLOR (%hhu, %hhu, %hhu), "
		"REFLECTIVITY %f, "
		"SPECULAR POWER %f, SPECULAR %f, "
		"DIFFUSE %f, "
		"AMBIENT %f\n",
		&(material->color[R]), &(material->color[G]), &(material->color[B]),
		&(material->reflectivity), 
		&(material->specular_power), &(material->i_specular),
		&(material->i_diffuse),
		&(material->i_ambient));
}
