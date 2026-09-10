// Experimental spatial atom view. All authoritative arrays use original atom IDs.
// SPDX-License-Identifier: LGPL-3.0-or-later

__device__ unsigned int spreadSpatialBits(unsigned int x) {
    x &= 1023;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    return (x | (x << 2)) & 0x09249249;
}

/* Specialized from libraries/hilbert/src/hilbert.cpp, hilbert_c2i, by Doug Moore.
 * Hilbert Curve implementation copyright 1998, Rice University
 * Copyright (c) 1998-2000, Rice University
 * Modified by OpenAI Codex, 2026-09-08: specialize to 3 dimensions, 10 bits,
 * and accept already interleaved coordinates for GPU execution.
 *
 * This software is copyrighted by Rice University. It may be freely copied,
 * modified, and redistributed, provided that the copyright notice is
 * preserved on all copies.
 * There is no warranty or other guarantee of fitness for this software,
 * it is provided solely "as is". Bug reports or fixes may be sent
 * to the author, who may or may not act on them as he desires.
 * You may include this software in a program or other software product,
 * but must display the notice:
 * Hilbert Curve implementation copyright 1998, Rice University
 * in any place where the end-user would see your own copyright.
 * If you modify this software, you should include a notice giving the
 * name of the person performing the modification, the date of modification,
 * and the reason for such modification.
 */
__device__ unsigned int spatialHilbert(unsigned int coords) {
    coords ^= coords >> 3;
    unsigned int index = 0, rotation = 0, flipBit = 0;
    for (int b = 27; b >= 0; b -= 3) {
        unsigned int bits = flipBit ^ ((coords >> b) & 7u);
        bits = ((bits >> rotation) | (bits << (3-rotation))) & 7u;
        index = (index << 3) | bits;
        flipBit = 1u << rotation;
        bits &= -bits & 3u;
        while (bits) { bits >>= 1; ++rotation; }
        if (++rotation >= 3) rotation -= 3;
    }
    index ^= (0x3FFFFFFFu/7u) >> 1;
    for (int shift = 1; shift < 30; shift *= 2)
        index ^= index >> shift;
    return index;
}

extern "C" __global__ void spatialKeys(const real4* pos, int2* keys,
        real4 boxX, real4 boxY, real4 boxZ
        ) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        real4 p = pos[i];
        real z = p.z/boxZ.z;
        real y = (p.y-z*boxZ.y)/boxY.y;
        real x = (p.x-y*boxY.x-z*boxZ.x)/boxX.x;
        x -= floor(x); y -= floor(y); z -= floor(z);
        unsigned int key = spreadSpatialBits(min(1023, (int)(1024*x))) |
                (spreadSpatialBits(min(1023, (int)(1024*y))) << 1) |
                (spreadSpatialBits(min(1023, (int)(1024*z))) << 2);
        key = spatialHilbert(key);
        keys[i] = make_int2(key, i);
    }
}

extern "C" __global__ void spatialMaps(const int2* keys, int* order, int* inverse
        ) {
    for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < PADDED_NUM_ATOMS; s += blockDim.x*gridDim.x) {
        int i = s < NUM_ATOMS ? keys[s].y : s;
        order[s] = i;
        inverse[i] = s;
    }
}

extern "C" __global__ void spatialPositions(const real4* original, real4* sorted, const int* order,
        unsigned long long* forces) {
    for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < PADDED_NUM_ATOMS; s += blockDim.x*gridDim.x) {
        sorted[s] = original[order[s]];
        forces[s] = forces[s+PADDED_NUM_ATOMS] = forces[s+2*PADDED_NUM_ATOMS] = 0;
    }
}

extern "C" __global__ void spatialMerge(unsigned long long* original,
        const unsigned long long* sorted, const int* map) {
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < NUM_ATOMS; index += blockDim.x*gridDim.x) {
        int i = index, s = map[index];
        for (int axis = 0; axis < 3; axis++)
            original[i+axis*PADDED_NUM_ATOMS] += sorted[s+axis*PADDED_NUM_ATOMS];
    }
}

extern "C" __global__ void spatialExclusionKeys(const int2* pairs, const int* inverse, int4* records) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int a = inverse[pairs[i].x], b = inverse[pairs[i].y];
        if (a/32 < b/32) { int temp = a; a = b; b = temp; }
        // Same tile ordering as the legacy list: lower block first, then upper block.
        records[i] = make_int4((b/32)*NUM_BLOCKS+a/32, a, b, 0);
    }
}

extern "C" __global__ void spatialMarkTiles(const int4* records, int* flags) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x)
        flags[i] = (i == 0 || records[i].x != records[i-1].x) ? 1 : 0;
}

// Inclusive scan within each 256-element chunk. Grid-stride chunks account for
// CudaContext::executeKernel() limiting the number of thread blocks.
extern "C" __global__ void spatialScanBlocks(int* values, int* sums, int n) {
    __shared__ int work[256];
    int t = threadIdx.x;
    for (int chunk = blockIdx.x; chunk < (n+255)/256; chunk += gridDim.x) {
        int i = 256*chunk+t;
        work[t] = i < n ? values[i] : 0;
        __syncthreads();
        for (int stride = 1; stride < 256; stride *= 2) {
            int value = t >= stride ? work[t-stride] : 0;
            __syncthreads();
            work[t] += value;
            __syncthreads();
        }
        if (i < n) values[i] = work[t];
        if (t == 255) sums[chunk] = work[t];
        __syncthreads();
    }
}

// Only the much smaller chunk-sum array is scanned serially, on the GPU.
extern "C" __global__ void spatialScanOffsets(const int* sums, int* offsets, int* total, int chunks) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < chunks; i++) { offsets[i] = sum; sum += sums[i]; }
        total[0] = sum;
    }
}

extern "C" __global__ void spatialBuildTiles(const int4* records, int* ids,
        const int* offsets, int2* tiles, int* rowCounts) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int tile = ids[i]+offsets[i/256]-1;
        ids[i] = tile;
        if (i == 0 || records[i].x != records[i-1].x) {
            int x = records[i].y/32, y = records[i].z/32;
            tiles[tile] = make_int2(x, y);
            atomicAdd(&rowCounts[x], 1);
            if (x != y) atomicAdd(&rowCounts[y], 1);
        }
    }
}

extern "C" __global__ void spatialRows(const int* counts, const int* offsets, unsigned int* rows) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_BLOCKS; i += blockDim.x*gridDim.x)
        rows[i+1] = counts[i]+offsets[i/256];
    if (blockIdx.x == 0 && threadIdx.x == 0) rows[0] = 0;
}

extern "C" __global__ void spatialInitMasks(const int* count, unsigned int* masks) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < count[0]*32; i += blockDim.x*gridDim.x)
        masks[i] = 0xFFFFFFFFu;
}

extern "C" __global__ void spatialMasks(const int4* records, const int* ids, unsigned int* masks) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int a = records[i].y, b = records[i].z;
        atomicAnd(&masks[ids[i]*32+(a%32)], ~(1u << (b%32)));
        // Off-diagonal records are already oriented by block. Diagonal tiles
        // need both directions even though only one original pair is stored.
        if (a/32 == b/32)
            atomicAnd(&masks[ids[i]*32+(b%32)], ~(1u << (a%32)));
    }
}

extern "C" __global__ void spatialAdjacency(const int2* tiles, const int* count,
        const unsigned int* rows, int* cursors, unsigned int* indices) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < count[0]; i += blockDim.x*gridDim.x) {
        int2 tile = tiles[i];
        indices[rows[tile.x]+atomicAdd(&cursors[tile.x], 1)] = tile.y;
        if (tile.x != tile.y)
            indices[rows[tile.y]+atomicAdd(&cursors[tile.y], 1)] = tile.x;
    }
}
