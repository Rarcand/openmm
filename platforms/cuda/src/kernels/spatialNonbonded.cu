// Spatial data gathering and exclusion rebuilding.
// SPDX-License-Identifier: LGPL-3.0-or-later

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
