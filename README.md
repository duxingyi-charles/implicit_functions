## Implicit functions

This library provides implicit function and its gradient evaulation for the following:

* Common analytical primitives (spheres, cylinders, cones, etc.)
* Implicit function generated from point cloud using
[Variational Implicit Point Set Surface (VIPSS)](https://www.cse.wustl.edu/~taoju/research/vipss.pdf)
* Implicit shape defined via WGSL shaders (requires the `-DIMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT`
  flag)

## Build

```sh
mkdir build
cd build

cmake ..
# or
cmake .. -DIMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT=ON # to enable shader support

make -j8
```

## Implicit function specification

### Primitives

Primitives are defined by a set of shape-specific parameters stored in a JSON file. Multiple
functions can be stored in the same JSON file. For example, a sphere centered at the origin with
radius 1 is defined as:

```json
{
  "sphere": {
    "type": "sphere",
    "center": [0.0, 0.0, 0.0],
    "radius": 1.0
  }
}
```

See [primitives](examples/primitives.json) for a list of example specifications.

### VIPSS

VIPSS function are defined by a set of points and their radial basis function (RBF) coefficients.
Both can be generated using the [VIPSS codebase](https://github.com/adshhzy/VIPSS). Here is an
example specification:

```json
{
  "vipss": {
    "type": "vipss",
    "points": "points_file.xyz",
    "rbf_coeffs": "coeffs_file.txt"
  }
}
```

### Shaders

Shader-based implicit functions take in a shader file and a parameter, detla, for finite-difference
approximation of the gradients. Here is an example specification:

```json
{
  "shader": {
    "type": "shader",
    "shader": "key.wgsl",
    "delta": 1e-3
  }
}
```
