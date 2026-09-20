// Conservative block-membership filter for original-ID exclusion rows.
// SPDX-License-Identifier: LGPL-3.0-or-later
// Shared CUDA/HIP/OpenCL-compatible integer logic. Hash collisions only cause
// extra exact scans: a set bit never authorizes omission of an interaction.
DEVICE unsigned int spatialExclusionBlockBit(int block) {
    return 1u << ((unsigned int) block & 31u);
}

DEVICE bool spatialExclusionMayContain(unsigned int filter, int block) {
    return (filter & spatialExclusionBlockBit(block)) != 0;
}
