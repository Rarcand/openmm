# Prototype changes

- Keep simulation data in original atom order, with separate spatial positions and forces for nonbonded calculations.
- Sort individual atoms on the GPU using Hilbert keys every 250 steps. CUDA and OpenCL share the same spatial key generation and Hilbert implementation. Maintain forward and inverse mappings between original IDs and spatial indices.
- Keep exclusion pairs in original IDs. After reordering, rebuild exclusion tiles, masks and the neighbor-list exclusion lookup on the GPU. Cache short exclusion rows using a capacity of up to 32 entries; read longer rows from global memory.
- Register per-atom arrays through a shared interface and refresh their sorted copies after parameter changes. Combine spatial forces with original-order forces after reciprocal-space calculations and before force consumers.
- Implement the shared interface on CUDA and OpenCL. Keep original execution as the default; explicit spatial mode rejects unsupported configurations, while auto mode selects the original implementation for them.
- Include CMake correctness tests and three tools for building, validation and benchmarking. Tests cover array updates, force accumulation, reordering, checkpoints, centroid bonds crossing periodic boundaries and exclusion rows exceeding the cache capacity.

See [PROTOTYPE.md](PROTOTYPE.md) for build commands and supported configurations.
