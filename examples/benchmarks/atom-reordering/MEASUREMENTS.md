# Measurement status

The trimmed review branch was built and tested on native Windows with an RTX 3070:

| Check | Result |
|---|---|
| CUDA/OpenCL, mixed/double CTest | 4/4 pass; 901,128 assertions per case |
| Exclusion rows longer than 32 entries | Forces/energy match legacy ordering before and after reordering |
| 12-case pristine OpenMM matrix | 240/240 force/energy snapshots pass |
| Independent array consumers | Pass on both backends and precisions |

The tests use generated systems. Production source hashes were checked against the native build. Local build logs, validation outputs and library hashes are retained under `build/review-validation`; failed test-compilation attempts are retained too. No archived files are needed to reproduce the CTest checks.

Earlier performance measurements predate the shared interface and this trimming. CUDA protein measured +2.46% (mixed) and +1.73% (double) versus pristine; adjusted one-sided upper bounds were +2.62% and +1.77%. OpenCL water timing was inconclusive because pristine repeat controls failed. Those numbers do not qualify this branch, other workloads, other GPUs or WSL as a Windows proxy.

No new performance campaign is part of preparing this runnable review branch. Historical logs and detailed reports remain in the experimental checkout; they are not build dependencies.
