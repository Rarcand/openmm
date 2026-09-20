# Stable atom identity and spatial nonbonded work

This change implements GPU atom reordering for issue #5406. Context creation selects the execution path automatically; there are no new performance configuration properties.

The authoritative positions and forces retain original particle IDs. Supported CUDA and OpenCL nonbonded execution uses a Hilbert-sorted view with forward and inverse mappings. Periodic translations preserve connected groups without requiring identical-molecule matching. Per-particle producers invalidate or update their spatial copies before the nonbonded consumer, and forces are merged before integration and virtual-site force distribution.

For the packed path, the original-ID exclusion adjacency remains authoritative. Reordering constructs a sparse GPU table mapping block/partner pairs to exact 32-bit masks. Packed neighbor entries carry these masks, so an off-diagonal exclusion does not require a full dedicated exclusion tile. Only diagonal tiles use the dedicated path. Hash collisions use linear probing; membership filters only reject definite misses. The full-tile work-view API rejects packed representations that it cannot describe.

The implementation reuses valid neighbor lists and fuses position gathering with preparation work. CUDA overlaps atomic force merging with PME completion. OpenCL combines its existing force reduction with the merge and combines fixed-point writes for consecutive tiles sharing a block. Neither changes coordinate precision. Small exhaustive cases avoid spatial sorting when supported; this uses OpenMM's existing neighbor-list decision, not a new fitted threshold. Spatial sorting retains a fixed 250-step cadence.

Selection is conservative. Multiple devices, DPD integrators (including those inside CompoundIntegrator), unsupported Force implementations, nonperiodic/no-cutoff systems, and unsupported OpenCL layouts retain existing execution. The OpenCL spatial path currently requires a 32-lane GPU. HIP retains its existing algorithm; its declaration is updated for the shared nonbonded capability argument. The selection rules are in `SpatialNonbondedPolicy.h`.

## Validation

Configure and build OpenMM normally with the desired GPU backends, then run:

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On systems with multiple OpenCL implementations, use the existing `OPENCL_TEST_PLATFORM_INDEX` and `OPENCL_TEST_DEVICE_INDEX` CMake test settings to select the hardware. These do not tune the architecture.

The spatial-work tests exercise registered arrays, resizing, independent force contributions, parameter updates, checkpoints, reinitialization, centroid geometry across periodic recentering, nested and linked DPD integrator selection, and rejection of incompatible packed views. The OpenCL NonbondedForce suite includes a box-change regression where coordinates do not move but the neighbor list must be invalidated.

With this build installed into the active Python environment, the existing benchmark can exercise the default path without architecture settings:

```sh
python examples/benchmarks/benchmark.py --platform CUDA --test pme --precision mixed --seconds 30
python examples/benchmarks/benchmark.py --platform OpenCL --test pme --precision mixed --seconds 30
```

Performance must be compared with an unmodified build of the same upstream commit on the same OS/device/precision. The older experimental measurements at `0da03998` do not qualify this upstream-based revision. Passing force tests does not establish bitwise agreement between different spatial summation orders.

This implementation and its tests were developed with AI assistance. Submission remains subject to OpenMM's AI policy and contributor review.
