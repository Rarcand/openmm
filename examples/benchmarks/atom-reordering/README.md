# Atom-reordering prototype

This branch is a self-contained CUDA/OpenCL experiment for OpenMM issue #5406. It includes native source, CMake correctness tests and three Python tools. It is not a complete replacement for every GPU configuration.

- [Build and test instructions](PROTOTYPE.md)
- [Measurement status](MEASUREMENTS.md)

The fixed spatial path uses unrestricted Hilbert sorting every 250 steps, full exclusion tiles, original atom IDs and separate spatial positions/forces. Exclusion tiles, masks and block-row lookup are rebuilt on the GPU. Short exclusion rows use a fixed 32-entry cache capacity; longer rows use global memory.

Start review with `CudaSpatialNonbonded.cpp`, the shared and CUDA `spatialNonbonded` kernels, `CudaNonbondedUtilities.cpp` and `findInteractingBlocks.cu`. `OpenCLSpatialNonbonded.cpp` supplies the second backend; `SpatialNonbondedView.h` documents the shared data interface.

Historical reports, raw results and alternative implementations remain in the separate `issue-5406-atom-reordering` experimental checkout. They are not required to build or run this branch.
