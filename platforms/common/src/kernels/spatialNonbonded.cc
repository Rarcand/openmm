// Spatial data gathering and exclusion rebuilding.
// SPDX-License-Identifier: LGPL-3.0-or-later

KERNEL void spatialMaps(GLOBAL const int2* keys, GLOBAL int* order, GLOBAL int* inverse
        ) {
    for (int s = GLOBAL_ID; s < PADDED_NUM_ATOMS; s += GLOBAL_SIZE) {
        int i = s < NUM_ATOMS ? keys[s].y : s;
        order[s] = i;
        inverse[i] = s;
    }
}

KERNEL void spatialPositions(GLOBAL const real4* original, GLOBAL real4* sorted, GLOBAL const int* order,
        GLOBAL mm_ulong* forces) {
    for (int s = GLOBAL_ID; s < PADDED_NUM_ATOMS; s += GLOBAL_SIZE) {
        sorted[s] = original[order[s]];
        forces[s] = forces[s+PADDED_NUM_ATOMS] = forces[s+2*PADDED_NUM_ATOMS] = 0;
    }
}

KERNEL void spatialMerge(GLOBAL mm_ulong* original,
        GLOBAL const mm_ulong* sorted, GLOBAL const int* map) {
    for (int index = GLOBAL_ID; index < NUM_ATOMS; index += GLOBAL_SIZE) {
        int i = index, s = map[index];
        for (int axis = 0; axis < 3; axis++)
            original[i+axis*PADDED_NUM_ATOMS] += sorted[s+axis*PADDED_NUM_ATOMS];
    }
}

KERNEL void spatialExclusionKeys(GLOBAL const int2* pairs, GLOBAL const int* inverse, GLOBAL int4* records) {
    for (int i = GLOBAL_ID; i < NUM_EXCLUSIONS; i += GLOBAL_SIZE) {
        int a = inverse[pairs[i].x], b = inverse[pairs[i].y];
        if (a/32 < b/32) { int temp = a; a = b; b = temp; }
        // Same tile ordering as the legacy list: lower block first, then upper block.
        records[i] = make_int4((b/32)*NUM_BLOCKS+a/32, a, b, 0);
    }
}

// Map the explicit exclusion set, without sorting pairs or constructing
// off-diagonal exclusion tiles. CSR rows retain original atom IDs.
KERNEL void spatialMapExclusions(GLOBAL const int* original, GLOBAL const int* inverse, GLOBAL int* mapped) {
    for (int i = GLOBAL_ID; i < NUM_EXCLUSIONS; i += GLOBAL_SIZE)
        mapped[i] = inverse[original[i]];
}

KERNEL void spatialDiagonalMasks(GLOBAL const int* order, GLOBAL const int* rows,
        GLOBAL const int* columns, GLOBAL unsigned int* masks) {
    for (int s = GLOBAL_ID; s < PADDED_NUM_ATOMS; s += GLOBAL_SIZE) {
        unsigned int allowed = 0;
        if (s < NUM_ATOMS) {
            allowed = 0xFFFFFFFFu;
            int original = order[s];
            for (int j = rows[original]; j < rows[original+1]; j++) {
                int other = columns[j];
                if (other/32 == s/32) allowed &= ~(1u << (other%32));
            }
        }
        masks[s] = allowed;
    }
}

KERNEL void spatialMarkTiles(GLOBAL const int4* records, GLOBAL int* flags) {
    for (int i = GLOBAL_ID; i < NUM_EXCLUSIONS; i += GLOBAL_SIZE)
        flags[i] = (i == 0 || records[i].x != records[i-1].x) ? 1 : 0;
}

// Inclusive scan within each 256-element chunk. Grid-stride chunks account for
// ComputeContext launch limits restricting the number of thread blocks.
KERNEL void spatialScanBlocks(GLOBAL int* values, GLOBAL int* sums, int n) {
    LOCAL int work[256];
    int t = LOCAL_ID;
    for (int chunk = GROUP_ID; chunk < (n+255)/256; chunk += NUM_GROUPS) {
        int i = 256*chunk+t;
        work[t] = i < n ? values[i] : 0;
        SYNC_THREADS;
        for (int stride = 1; stride < 256; stride *= 2) {
            int value = t >= stride ? work[t-stride] : 0;
            SYNC_THREADS;
            work[t] += value;
            SYNC_THREADS;
        }
        if (i < n) values[i] = work[t];
        if (t == 255) sums[chunk] = work[t];
        SYNC_THREADS;
    }
}

// Only the much smaller chunk-sum array is scanned serially, on the GPU.
KERNEL void spatialScanOffsets(GLOBAL const int* sums, GLOBAL int* offsets, GLOBAL int* total, int chunks) {
    if (GROUP_ID == 0 && LOCAL_ID == 0) {
        int sum = 0;
        for (int i = 0; i < chunks; i++) { offsets[i] = sum; sum += sums[i]; }
        total[0] = sum;
    }
}

KERNEL void spatialBuildTiles(GLOBAL const int4* records, GLOBAL int* ids,
        GLOBAL const int* offsets, GLOBAL int2* tiles, GLOBAL int* rowCounts) {
    for (int i = GLOBAL_ID; i < NUM_EXCLUSIONS; i += GLOBAL_SIZE) {
        int tile = ids[i]+offsets[i/256]-1;
        ids[i] = tile;
        if (i == 0 || records[i].x != records[i-1].x) {
            int x = records[i].y/32, y = records[i].z/32;
            tiles[tile] = make_int2(x, y);
            ATOMIC_ADD(&rowCounts[x], 1);
            if (x != y) ATOMIC_ADD(&rowCounts[y], 1);
        }
    }
}

KERNEL void spatialRows(GLOBAL const int* counts, GLOBAL const int* offsets, GLOBAL unsigned int* rows) {
    for (int i = GLOBAL_ID; i < NUM_BLOCKS; i += GLOBAL_SIZE)
        rows[i+1] = counts[i]+offsets[i/256];
    if (GROUP_ID == 0 && LOCAL_ID == 0) rows[0] = 0;
}

KERNEL void spatialInitMasks(GLOBAL const int* count, GLOBAL unsigned int* masks) {
    for (int i = GLOBAL_ID; i < count[0]*32; i += GLOBAL_SIZE)
        masks[i] = 0xFFFFFFFFu;
}

KERNEL void spatialMasks(GLOBAL const int4* records, GLOBAL const int* ids, GLOBAL unsigned int* masks) {
    for (int i = GLOBAL_ID; i < NUM_EXCLUSIONS; i += GLOBAL_SIZE) {
        int a = records[i].y, b = records[i].z;
        ATOMIC_AND(&masks[ids[i]*32+(a%32)], ~(1u << (b%32)));
        // Off-diagonal records are already oriented by block. Diagonal tiles
        // need both directions even though only one original pair is stored.
        if (a/32 == b/32)
            ATOMIC_AND(&masks[ids[i]*32+(b%32)], ~(1u << (a%32)));
    }
}

KERNEL void spatialAdjacency(GLOBAL const int2* tiles, GLOBAL const int* count,
        GLOBAL const unsigned int* rows, GLOBAL int* cursors, GLOBAL unsigned int* indices) {
    for (int i = GLOBAL_ID; i < count[0]; i += GLOBAL_SIZE) {
        int2 tile = tiles[i];
        indices[rows[tile.x]+ATOMIC_ADD(&cursors[tile.x], 1)] = tile.y;
        if (tile.x != tile.y)
            indices[rows[tile.y]+ATOMIC_ADD(&cursors[tile.y], 1)] = tile.x;
    }
}
