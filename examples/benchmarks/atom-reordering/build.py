"""Configure and build the CUDA/OpenCL prototype with ordinary CMake tools.

On Windows, run in a Visual Studio developer shell with CUDA_PATH set. The
Python bindings must match this OpenMM revision; this builds native libraries.
"""
import argparse
import os
from pathlib import Path
import subprocess

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--backends', nargs='+', choices=['CUDA', 'OpenCL'], default=['CUDA', 'OpenCL'])
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--tests', action='store_true', help='Build the independent C++ work-view test libraries')
    p.add_argument('--generator', default='Ninja')
    p.add_argument('--cmake-option', action='append', default=[])
    a = p.parse_args()
    command = ['cmake',
        '-S',
        str(a.source.resolve()),
        '-B',
        str(a.build_dir.resolve()),
        '-G',
        a.generator,
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_TESTING=' + ('ON' if a.tests else 'OFF')]
    for backend in ('CUDA', 'OpenCL'):
        command.append('-DOPENMM_BUILD_' + backend.upper() + '_LIB=' + ('ON' if backend in a.backends else 'OFF'))
    for feature in ('HIP_LIB',
        'CPU_LIB',
        'PYTHON_WRAPPERS',
        'C_AND_FORTRAN_WRAPPERS',
        'AMOEBA_PLUGIN',
        'DRUDE_PLUGIN',
        'RPMD_PLUGIN',
        'PME_PLUGIN',
        'EXAMPLES'):
        command.append('-DOPENMM_BUILD_' + feature + '=OFF')
    if os.name == 'nt' and 'CUDA_PATH' in os.environ:
        cuda = Path(os.environ['CUDA_PATH'])
        command += ['-DCUDAToolkit_ROOT=' + str(cuda),
            '-DNVRTC_LIB=' + str(cuda / 'lib/x64/nvrtc.lib'),
            '-DOPENCL_INCLUDE_DIR=' + str(cuda / 'include'),
            '-DOPENCL_LIBRARY=' + str(cuda / 'lib/x64/OpenCL.lib')]
    subprocess.run(command + a.cmake_option, check=True)
    targets = ['OpenMM' + backend for backend in a.backends]
    if a.tests:
        targets += ['SpatialWorkViewTest' + backend for backend in a.backends]
        targets += ['TestSpatialWorkView' + backend for backend in a.backends]
    subprocess.run(['cmake', '--build', str(a.build_dir.resolve()), '--config', 'Release', '--parallel', str(a.jobs), '--target'] + targets,
        check=True)
if __name__ == '__main__':
    main()
