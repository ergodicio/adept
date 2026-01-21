diff: jj diff -f main -t@

- In general let's try to be very clear about the units of all quantities. We should either operate on floats which have the dimensions in the name, or on Pint/jpu `Quantities` that come bundled with a specific unit. For now limit changes to just the current diff.
