
#include "triangle.h"

// Reads triangle data from the specified file and sets specified triangle
void Triangle_readTo (FILE * file, triangle_t * triangle)
{
	fscanf(file, "POINTS (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n",
		&(triangle->points[U][X]), &(triangle->points[U][Y]), &(triangle->points[U][Z]),
		&(triangle->points[V][X]), &(triangle->points[V][Y]), &(triangle->points[V][Z]),
		&(triangle->points[W][X]), &(triangle->points[W][Y]), &(triangle->points[W][Z]));

	float temp1[DSPACE], temp2[DSPACE];
	VECTOR_SUB(temp1, triangle->points[V], triangle->points[U]);
    VECTOR_SUB(temp2, triangle->points[W], triangle->points[U]);
    VECTOR_CROSSPRODUCT(triangle->normal, temp1, temp2);
    Vector_normalize(triangle->normal);
}

// Find intersection points between ray and triangle.
__device__ float Triangle_intersect (line_t * ray, triangle_t * triangle)
{
    float * direction = ray->direction,
    	  * normal = triangle->normal,
    	  * p0 = triangle->points[U],
    	  intersection[DSPACE], u[DSPACE], v[DSPACE], w[DSPACE], temp[DSPACE],
          d, numerator, denominator, uu, uv, uw, vv, vw, s, t;

    //denominator = n . direction
    denominator = VECTOR_DOT(normal, direction);

    if (fabs(denominator) < EPSILON) {
        return NAN;
    }

    //numerator = n . (p0 - origin)
    VECTOR_SUB(temp, p0, ray->position);
    numerator = VECTOR_DOT(normal, temp);

    d = numerator / denominator;

    if (d < 0) {
        return NAN;
    }

    findIntersectionPoint(intersection, ray, d);

    VECTOR_SUB(u, triangle->points[V], p0);
    VECTOR_SUB(v, triangle->points[W], p0);
    VECTOR_SUB(w, intersection, p0);

    uu = VECTOR_DOT(u, u);
    uv = VECTOR_DOT(u, v);
    uw = VECTOR_DOT(u, w);
    vv = VECTOR_DOT(v, v);
    vw = VECTOR_DOT(v, w);

    denominator = uv * uv - uu * vv;

    s = (uv * vw - vv * uw) / denominator;
    if (s < 0.0 || s > 1.0) {
        return NAN;
    }

    t = (uv * uw - uu * vw) / denominator;
    if (t < 0.0 || s + t > 1.0) {
        return NAN;
    }

    return d;
}