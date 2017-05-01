
#include "triangle.h"

// Reads triangle data from the specified file and sets specified triangle
void Triangle_readTo (FILE * file, triangle_t * t)
{
	fscanf(file, "POINTS "
        "(%f, %f, %f), "
        "(%f, %f, %f), "
        "(%f, %f, %f)\n",
		&(t->points[U][X]), &(t->points[U][Y]), &(t->points[U][Z]),
		&(t->points[V][X]), &(t->points[V][Y]), &(t->points[V][Z]),
		&(t->points[W][X]), &(t->points[W][Y]), &(t->points[W][Z]));

	float temp1[DSPACE], temp2[DSPACE];
	VECTOR_SUB(temp1, t->points[V], t->points[U]);
    VECTOR_SUB(temp2, t->points[W], t->points[U]);
    VECTOR_CROSSPRODUCT(t->normal, temp1, temp2);
    Vector_normalize(t->normal);
}

// Find intersection points between ray and triangle.
__device__ float Triangle_intersect (line_t * ray, triangle_t * t)
{
    float * direction = ray->direction,
    	  * normal = t->normal,
    	  * p0 = t->points[U],
    	  intersection[DSPACE], u[DSPACE], v[DSPACE], w[DSPACE], temp[DSPACE],
          d, denominator, uu, uv, uw, vv, vw, r, s;

    //denominator = n . direction
    denominator = VECTOR_DOT(normal, direction);

    if (fabs(denominator) < EPSILON) {
        return NAN;
    }

    //numerator = n . (p0 - origin)
    VECTOR_SUB(temp, p0, ray->position);

    d = VECTOR_DOT(normal, temp) / denominator;

    if (d < 0) {
        return NAN;
    }

    findIntersectionPoint(intersection, ray, d);

    VECTOR_SUB(u, t->points[V], p0);
    VECTOR_SUB(v, t->points[W], p0);
    VECTOR_SUB(w, intersection, p0);

    uu = VECTOR_DOT(u, u);
    uv = VECTOR_DOT(u, v);
    uw = VECTOR_DOT(u, w);
    vv = VECTOR_DOT(v, v);
    vw = VECTOR_DOT(v, w);

    denominator = uv * uv - uu * vv;

    r = (uv * vw - vv * uw) / denominator;
    if (r < 0.0 || r > 1.0) {
        return NAN;
    }

    s = (uv * uw - uu * vw) / denominator;
    if (s < 0.0 || r + s > 1.0) {
        return NAN;
    }

    return d;
}