# Build and test

## Native correctness tests (no installed Python OpenMM package needed)

Requirements: a C++ toolchain, CMake, Ninja, Python for the build driver, CUDA toolkit and OpenCL headers/loader, plus an NVIDIA GPU driver. On Windows use a Visual Studio developer shell with `CUDA_PATH` set. Build and run on the same OS.

From the repository root:

```sh
python examples/benchmarks/atom-reordering/build.py --build-dir build/prototype --tests
ctest --test-dir build/prototype -R '^SpatialWorkView' --output-on-failure
```

For one backend, add `--backends CUDA` or `--backends OpenCL`. The tests construct their own systems; no downloaded fixtures, archived scripts or prebuilt DLLs are needed. They check independent array consumers, force accumulation, exclusions, updates, reordering, checkpoints and reinitialization in mixed/double precision. The native fixture currently selects device 0. It also constructs an exclusion row longer than 32 entries and checks the global-memory path against legacy forces/energy before and after reordering.

## Python comparison with pristine OpenMM

For the broader 12-case matrix, additionally use NumPy and Python OpenMM bindings matching this source revision. The native test build does not build/install Python bindings. Do not assume an arbitrary installed OpenMM wheel matches this development checkout. Pristine reference libraries must be built from base commit `0da03998df892bbb0a954ad3767c30a0cc53a11c`, with the same toolchain and precision. Each command must run in its own process.

Set `PRISTINE_BUILD` and `PROTOTYPE_BUILD` below to directories containing the corresponding core library and selected backend library (on Windows, DLLs; Linux, shared objects):

```sh
python examples/benchmarks/atom-reordering/validate.py --backend CUDA --mode pristine --library-dir PRISTINE_BUILD --output build/reference.json
python examples/benchmarks/atom-reordering/validate.py --backend CUDA --library-dir PROTOTYPE_BUILD --reference build/reference.json.npz --output build/check.json
```

Repeat for OpenCL and with `--precision double`. `--device-index` and `--opencl-platform-index` select devices for this Python matrix. Outputs refuse to overwrite an earlier run.

`benchmark.py` accepts `--system system.xml --state state.xml`, plus the same backend/build/precision selection. Use `--mode pristine` for the reference. Match all simulation settings and interleave independent processes. Its batches are not independent confidence samples.

## Scope

`AtomReordering=baseline` is the default. `inverse` strictly requests the spatial implementation; `auto` falls back to original execution when the shared support policy rejects a configuration. Query `AtomReorderingStatus` for the route/reason. The sorting method, interval and full-tile representation are fixed on this review branch.

Supported spatial execution includes periodic standard nonbonded methods (including PME, Ewald and LJPME), periodic custom nonbonded interactions without interaction groups, and the ordinary bonded/external forces listed in `SpatialNonbondedPolicy.h`. Parameter producers invalidate registered arrays; sorted charges/parameters refresh before consumption. Spatial forces merge after reciprocal work and before integrators/virtual sites consume forces.

Nonperiodic/NoCutoff systems, custom interaction groups, forces outside the allowlist, multiple devices and OpenCL devices without the supported 32-wide GPU layout use the original path in auto mode. These are explicit scope limits, not completed spatial migrations. HIP spatial support and third-party plugin migration are not included.
