# Derivations

Mathematical derivations for the core intersection and reflection formulas used
in `intersect.cuh`.

## Sphere Intersection

A point `P` lies on a sphere's surface if its distance from the sphere's center
equals the sphere's radius:

`length(P - center) = radius`


Squaring both sides avoids the square root in `length()`:

`(P - center) · (P - center) = radius²`


A point along a ray at distance `t` is `P = origin + t * direction`. Let
`oc = origin - center`. Substituting:

`(oc + tdirection) · (oc + tdirection) = radius²`


Expanding the dot product (distributing exactly like multiplication):

`t²(direction·direction) + 2t(oc·direction) + (oc·oc - radius²) = 0`


This is a standard quadratic `at² + bt + c = 0`, with:

`a = direction · direction`
`b = 2 * (oc · direction)`
`c = oc · oc - radius²`


Solved with the quadratic formula:

`t = (-b ± sqrt(b² - 4ac)) / (2a)`


The discriminant `b² - 4ac` determines the number of real intersections

The two roots are:

- `t0 = (-b - sqrt(discriminant)) / (2a)` — the near/entry point
- `t1 = (-b + sqrt(discriminant)) / (2a)` — the far/exit point

Only positive roots (`t > 0.001`) are valid. This threshold serves two purposes:

- rejecting intersections behind the ray's origin
- rejecting near-zero self-intersections caused by floating-point imprecision,
  which occur when a ray starts exactly on a surface.

`t0` is checked first, since it represents the closer, visible surface. If
`t0` is invalid, `t1` is checked as a fallback, covering rays that originate
from inside or exactly on the sphere. 

## Plane Intersection

The ground plane is defined as every point where the z-coordinate equals a
fixed height `plane_z`, spanning infinitely across x and y.

A point along a ray at distance `t` has z-coordinate `origin.z + t * direction.z`.
Setting this equal to the plane's height and solving for `t`:

`origin.z + t * direction.z = plane_z
t = (plane_z - origin.z) / direction.z`


If `direction.z` is very close to zero, the ray is nearly parallel to the
plane and would need to travel an unrealistically large distance to reach it;
this is treated as a miss rather than risking a near-divide-by-zero result.

## Reflection

Given an incoming ray direction `d` and a surface normal `n`, the reflected
direction is found by decomposing `d` into two parts:

- a component parallel to `n`, pointing into the surface
- a component tangent to the surface

The perpendicular component is the projection of `d` onto `n`:

`n * (d · n)`


Reflection reverses only the perpendicular component and leaves the tangent
component unchanged:

- subtracting the perpendicular component once cancels it
- subtracting it a second time flips it to point outward instead

`reflected = d - 2 * (d · n) * n`