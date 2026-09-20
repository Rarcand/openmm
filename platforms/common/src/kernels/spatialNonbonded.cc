// Experimental spatial atom view. All authoritative arrays use original atom IDs.
// SPDX-License-Identifier: LGPL-3.0-or-later

DEVICE unsigned int spreadSpatialBits(unsigned int x) {
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
DEVICE unsigned int spatialHilbert(unsigned int coords) {
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

KERNEL void spatialKeys(GLOBAL const real4* pos, GLOBAL int2* keys,
        real4 boxX, real4 boxY, real4 boxZ
#ifdef GROUPED_SPATIAL
        , GLOBAL const int* groupAtoms, GLOBAL const int2* groupBounds
#endif
        ) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
#ifdef GROUPED_SPATIAL
        // Every rank in a group computes the same centroid in the same order.
        // The rank tie-breaker keeps its atoms contiguous even on key collisions.
        int2 range = groupBounds[i];
        real4 anchor = pos[groupAtoms[range.x]];
        real az = anchor.z/boxZ.z;
        real ay = (anchor.y-az*boxZ.y)/boxY.y;
        real ax = (anchor.x-ay*boxY.x-az*boxZ.x)/boxX.x;
        real x = 0, y = 0, z = 0;
        for (int j = range.x; j < range.y; j++) {
            real4 p = pos[groupAtoms[j]];
            real fz = p.z/boxZ.z;
            real fy = (p.y-fz*boxZ.y)/boxY.y;
            real fx = (p.x-fy*boxY.x-fz*boxZ.x)/boxX.x;
            x += fx-floor(fx-ax+0.5f);
            y += fy-floor(fy-ay+0.5f);
            z += fz-floor(fz-az+0.5f);
        }
        real scale = RECIP((real) (range.y-range.x));
        x *= scale; y *= scale; z *= scale;
#else
        real4 p = pos[i];
        real z = p.z/boxZ.z;
        real y = (p.y-z*boxZ.y)/boxY.y;
        real x = (p.x-y*boxY.x-z*boxZ.x)/boxX.x;
#endif
        x -= floor(x); y -= floor(y); z -= floor(z);
        unsigned int key = spreadSpatialBits(min(1023, (int)(1024*x))) |
                (spreadSpatialBits(min(1023, (int)(1024*y))) << 1) |
                (spreadSpatialBits(min(1023, (int)(1024*z))) << 2);
#ifdef HILBERT_SPATIAL
        key = spatialHilbert(key);
#endif
        keys[i] = make_int2(key, i);
    }
}

KERNEL void spatialMaps(GLOBAL const int2* keys, GLOBAL int* order, GLOBAL int* inverse
#ifdef GROUPED_SPATIAL
        , GLOBAL const int* groupAtoms
#endif
        ) {
    for (int s = GLOBAL_ID; s < PADDED_NUM_ATOMS; s += GLOBAL_SIZE) {
#ifdef GROUPED_SPATIAL
        int i = s < NUM_ATOMS ? groupAtoms[keys[s].y] : s;
#else
        int i = s < NUM_ATOMS ? keys[s].y : s;
#endif
        order[s] = i;
        inverse[i] = s;
    }
}

KERNEL void spatialPositions(GLOBAL const real4* original, GLOBAL real4* sorted, GLOBAL const int* order,
        GLOBAL mm_ulong* forces) {
    for (int s = GLOBAL_ID; s < PADDED_NUM_ATOMS; s += GLOBAL_SIZE) {
        sorted[s] = original[order[s]];
#ifdef FUSED_SPATIAL
        forces[s] = forces[s+PADDED_NUM_ATOMS] = forces[s+2*PADDED_NUM_ATOMS] = 0;
#endif
    }
}

KERNEL void spatialMerge(GLOBAL mm_ulong* original,
        GLOBAL const mm_ulong* sorted, GLOBAL const int* map) {
    for (int index = GLOBAL_ID; index < NUM_ATOMS; index += GLOBAL_SIZE) {
#ifdef INVERSE_MERGE
        int i = index, s = map[index];
#else
        int s = index, i = map[index];
#endif
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
#ifdef COMPACT_EXCLUSIONS
        // Off-diagonal records are already oriented by block. Diagonal tiles
        // need both directions even though only one original pair is stored.
        if (a/32 == b/32)
            ATOMIC_AND(&masks[ids[i]*32+(b%32)], ~(1u << (a%32)));
#endif
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
