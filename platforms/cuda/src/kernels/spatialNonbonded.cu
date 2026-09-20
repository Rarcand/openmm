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
#ifdef IDENTITY_SPATIAL
        keys[i] = make_int2(i, i);
#else
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
#endif
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
#ifdef FUSED_SPATIAL
        forces[s] = forces[s+PADDED_NUM_ATOMS] = forces[s+2*PADDED_NUM_ATOMS] = 0;
#endif
    }
}

extern "C" __global__ void spatialMerge(unsigned long long* original,
        const unsigned long long* sorted, const int* map) {
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < NUM_ATOMS; index += blockDim.x*gridDim.x) {
        int i = index, s = map[index];
        for (int axis = 0; axis < 3; axis++)
            atomicAdd(&original[i+axis*PADDED_NUM_ATOMS], sorted[s+axis*PADDED_NUM_ATOMS]);
    }
}

extern "C" __global__ void spatialExclusionKeys(const int2* pairs, const int* inverse, int2* records) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int a = inverse[pairs[i].x], b = inverse[pairs[i].y];
        if (a/32 < b/32) { int temp = a; a = b; b = temp; }
        // Same tile ordering as the legacy list: lower block first, then upper block.
        records[i] = make_int2((b/32)*NUM_BLOCKS+a/32, (a&31) | ((b&31)<<5) | ((a/32 == b/32)<<10));
    }
}

// Map the explicit exclusion set, without sorting pairs or constructing
// off-diagonal exclusion tiles. CSR rows retain original atom IDs.
extern "C" __global__ void spatialMapExclusions(const int* original, const int* inverse, int* mapped) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x)
        mapped[i] = inverse[original[i]];
}

extern "C" __global__ void spatialDiagonalMasks(const int* order, const int* rows,
        const int* columns, unsigned int* masks
#ifdef EXCLUSION_BLOCK_FILTER
        , unsigned int* blockFilter
#endif
        ) {
    for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < PADDED_NUM_ATOMS; s += blockDim.x*gridDim.x) {
        unsigned int allowed = 0;
#ifdef EXCLUSION_BLOCK_FILTER
        unsigned int filter = 0;
#endif
        if (s < NUM_ATOMS) {
            allowed = 0xFFFFFFFFu;
            int original = order[s];
            for (int j = rows[original]; j < rows[original+1]; j++) {
                int other = columns[j];
#ifdef EXCLUSION_BLOCK_FILTER
                filter |= spatialExclusionBlockBit(other/32);
#endif
                if (other/32 == s/32) allowed &= ~(1u << (other%32));
            }
        }
        masks[s] = allowed;
#ifdef EXCLUSION_BLOCK_FILTER
        // Rebuilt with every permutation, including zeroes for padded ranks.
        // This adds no extra launch or second traversal of the CSR rows.
        blockFilter[s] = filter;
#endif
    }
}

extern "C" __global__ void spatialMarkTiles(const int2* records, int* flags) {
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

extern "C" __global__ void spatialBuildTiles(const int2* records, int* ids,
        const int* offsets, int2* tiles, int* rowCounts) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int tile = ids[i]+offsets[i/256]-1;
        ids[i] = tile;
        if (i == 0 || records[i].x != records[i-1].x) {
            int y = records[i].x/NUM_BLOCKS, x = records[i].x-y*NUM_BLOCKS;
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

extern "C" __global__ void spatialMasks(const int2* records, const int* ids, unsigned int* masks) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_EXCLUSIONS; i += blockDim.x*gridDim.x) {
        int a = records[i].y&31, b = (records[i].y>>5)&31;
        atomicAnd(&masks[ids[i]*32+(a%32)], ~(1u << (b%32)));
#ifdef COMPACT_EXCLUSIONS
        // Off-diagonal records are already oriented by block. Diagonal tiles
        // need both directions even though only one original pair is stored.
        if (records[i].y & 1024)
            atomicAnd(&masks[ids[i]*32+(b%32)], ~(1u << (a%32)));
#endif
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

/** Partition a rebuilt packed list by physical exclusions, not geometric mask bits.
 * One warp inspects a whole tile. Clean and masked indices grow from opposite
 * ends of one capacity-sized array; their total is exactly interactionCount[0].
 * Cached lists keep their partition. Overflow is handled by the normal retry.
 */

__device__ __forceinline__ unsigned int blockExclusionHash(unsigned int atom) {
    unsigned int h = atom*0x9e3779b9u;
    return h^(h>>16);
}

extern "C" __global__ void blockCountExclusions(const int* order, const int* rows, const int* columns, unsigned int* counts) {
    for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < NUM_ATOMS; s += blockDim.x*gridDim.x) {
        int original = order[s];
        for (int j = rows[original]; j < rows[original+1]; j++) {
            int row = columns[j]/32;
            if (row != s/32) atomicAdd(counts+row, 1u);
        }
    }
}

extern "C" __global__ void blockLayout(const unsigned int* counts, int2* rows) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int offset = 0;
    for (int row = 0; row < NUM_BLOCKS; row++) {
        unsigned int capacity = 2;
        while (capacity < 2*counts[row]) capacity *= 2;
        rows[row] = make_int2(offset, capacity);
        offset += capacity;
    }
}

extern "C" __global__ void blockInsertExclusions(const int* order, const int* originalRows,
        const int* columns, const int2* rows, uint2* table, unsigned int* filters) {
    for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < NUM_ATOMS; s += blockDim.x*gridDim.x) {
        int original = order[s];
        unsigned int hash = blockExclusionHash(s);
        for (int j = originalRows[original]; j < originalRows[original+1]; j++) {
            int other = columns[j], row = other/32;
            if (row == s/32) continue;
            int2 range = rows[row];
            unsigned int slot = hash & (range.y-1);
            while (true) {
                unsigned int key = atomicCAS(&table[range.x+slot].x, 0u, (unsigned int) s+1);
                if (key == 0 || key == (unsigned int) s+1) {
                    atomicOr(&table[range.x+slot].y, 1u<<(other%32));
                    break;
                }
                slot = (slot+1)&(range.y-1);
            }
            atomicOr(filters+8*row+((hash&255)>>5), 1u<<(hash&31));
        }
    }
}

// Validate both missing entries and every stored mask against authoritative CSR.
extern "C" __global__ void blockVerifyExclusions(const int* order, const int* originalRows,
        const int* columns, const int2* rows, const uint2* table, const unsigned int* filters, int* errors) {
    int start = blockIdx.x*blockDim.x+threadIdx.x, stride = blockDim.x*gridDim.x;
    for (int s = start; s < NUM_ATOMS; s += stride) {
        int original = order[s];
        unsigned int hash = blockExclusionHash(s);
        for (int j = originalRows[original]; j < originalRows[original+1]; j++) {
            int other = columns[j], row = other/32;
            if (row == s/32) continue;
            int2 range = rows[row];
            unsigned int slot = hash&(range.y-1), mask = 0;
            for (int probe = 0; probe < range.y; probe++) {
                uint2 entry = table[range.x+slot];
                if (entry.x == (unsigned int) s+1) { mask = entry.y; break; }
                if (entry.x == 0) break;
                slot = (slot+1)&(range.y-1);
            }
            if (!(mask&(1u<<(other%32))) || !(filters[8*row+((hash&255)>>5)]&(1u<<(hash&31)))) atomicAdd(errors, 1);
        }
    }
    for (int row = start; row < NUM_BLOCKS; row += stride) {
        int2 range = rows[row];
        for (int slot = 0; slot < range.y; slot++) {
            uint2 entry = table[range.x+slot];
            if (entry.x == 0) continue;
            if (entry.x > NUM_ATOMS) { atomicAdd(errors, 1); continue; }
            int original = order[entry.x-1];
            unsigned int expected = 0;
            for (int j = originalRows[original]; j < originalRows[original+1]; j++)
                if (columns[j]/32 == row) expected |= 1u<<(columns[j]%32);
            if (entry.y != expected) atomicAdd(errors, 1);
        }
    }
}
