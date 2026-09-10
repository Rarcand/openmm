"""Run OpenMM from an explicit build and record reproducible, uninstrumented timing.

Use separate processes/builds for baseline and prototype. Do not interpret batches
from one process as independent sessions. See PROTOTYPE.md for usage.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
_handles = []

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_openmm(backend, directory=None):
    """Load a selected build before the Python bindings; otherwise use installation."""
    libraries = {}
    if directory:
        if 'openmm' in sys.modules:
            raise RuntimeError('Select a native build before importing OpenMM')
        directory = Path(directory).resolve()
        empty = tempfile.TemporaryDirectory(prefix='openmm-empty-plugins-')
        _handles.append(empty)
        os.environ['OPENMM_PLUGIN_DIR'] = empty.name
        if os.name == 'nt':
            folders = [directory]
            if 'CUDA_PATH' in os.environ:
                folders += [Path(os.environ['CUDA_PATH']) / 'bin', Path(os.environ['CUDA_PATH']) / 'bin/x64']
            for folder in folders:
                if folder.exists():
                    _handles.append(os.add_dll_directory(str(folder)))
            core, plugin = (directory / 'OpenMM.dll', directory / ('OpenMM' + backend + '.dll'))
        else:
            suffix = '.dylib' if __import__('sys').platform == 'darwin' else '.so'
            core, plugin = (directory / ('libOpenMM' + suffix), directory / ('libOpenMM' + backend + suffix))
        _handles.append(ctypes.CDLL(str(core), mode=ctypes.RTLD_GLOBAL))
        import openmm as mm
        mm.Platform.loadPluginLibrary(str(plugin))
        libraries = {str(path): digest(path) for path in (core, plugin)}
    else:
        import openmm as mm
    return (mm, libraries)

def properties(backend, precision, mode):
    props = dict(Precision=precision, DeviceIndex='0')
    if backend == 'OpenCL':
        props['OpenCLPlatformIndex'] = '0'
    # Pristine OpenMM does not expose the experimental property.
    if mode != 'pristine':
        props['AtomReordering'] = mode
    return props

def snapshot(context):
    import numpy as np
    from openmm import unit
    state = context.getState(forces=True, energy=True)
    return dict(forces=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        energy=np.asarray(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))

def check_snapshot(actual, expected):
    import numpy as np
    delta = actual['forces'] - expected['forces']
    scale = np.sqrt(np.mean(expected['forces'] ** 2))
    assert np.isfinite(delta).all()
    assert np.sqrt(np.mean(delta ** 2)) <= 0.001 + 0.0001 * scale
    assert np.max(np.abs(delta)) <= 0.001 + 0.001 * scale
    np.testing.assert_allclose(actual['energy'], expected['energy'], rtol=0.0001, atol=0.001)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['CUDA', 'OpenCL'], required=True)
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--device-index', default='0')
    p.add_argument('--opencl-platform-index', default='0')
    p.add_argument('--mode', choices=['pristine', 'baseline', 'auto', 'inverse'], default='auto')
    p.add_argument('--precision', choices=['mixed', 'double'], default='mixed')
    p.add_argument('--system', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, help='Baseline .npz snapshot to compare')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--timestep', type=float, default=0.002, help='ps')
    a = p.parse_args()
    if a.steps < 1 or a.batches < 1 or a.warmup < 0:
        p.error('steps/batches must be positive; warmup must be nonnegative')
    if a.output.exists() or Path(str(a.output) + '.npz').exists():
        p.error('Output exists; preserve previous runs')
    mm, libraries = load_openmm(a.backend, a.library_dir)
    import numpy as np
    system = mm.XmlSerializer.deserialize(a.system.read_text())
    state = mm.XmlSerializer.deserialize(a.state.read_text())
    integrator = mm.VerletIntegrator(a.timestep)
    platform = mm.Platform.getPlatformByName(a.backend)
    props = properties(a.backend, a.precision, a.mode)
    props['DeviceIndex'] = a.device_index
    if a.backend == 'OpenCL':
        props['OpenCLPlatformIndex'] = a.opencl_platform_index
    context = mm.Context(system, integrator, platform, props)
    context.setState(state)
    initial = snapshot(context)
    if a.reference:
        with np.load(a.reference) as expected:
            check_snapshot(initial, expected)
    integrator.step(a.warmup)
    context.getState(energy=True)
    samples = []
    for _ in range(a.batches):
        start = time.perf_counter()
        integrator.step(a.steps)
        energy = context.getState(energy=True).getPotentialEnergy()._value
        elapsed = time.perf_counter() - start
        assert np.isfinite(energy)
        samples.append(elapsed)
    context.setState(state)
    check_snapshot(snapshot(context), initial)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(a.output) + '.npz', **initial)
    result = dict(backend=a.backend,
        precision=a.precision,
        requested_mode=a.mode,
        properties={k: platform.getPropertyValue(context, k) for k in platform.getPropertyNames()},
        libraries=libraries,
        system_sha256=digest(a.system),
        state_sha256=digest(a.state),
        steps_per_batch=a.steps,
        warmup=a.warmup,
        seconds=samples,
        median_ms_per_step=1000 * statistics.median(samples) / a.steps,
        numerical_checks_passed=True,
        interpretation='One process/session; batches are correlated, not confidence-interval samples.')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
