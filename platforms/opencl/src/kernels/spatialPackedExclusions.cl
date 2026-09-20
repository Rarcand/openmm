// SPDX-License-Identifier: LGPL-3.0-or-later
// Exact sparse block-to-partner masks, rebuilt when the atom order changes.
inline unsigned int blockExclusionHash(unsigned int atom) {
    unsigned int h = atom*0x9e3779b9u;
    return h^(h>>16);
}

__kernel void blockCountExclusions(__global const int* order, __global const int* rows, __global const int* columns, __global unsigned int* counts) {
    for (int s = get_global_id(0); s < NUM_ATOMS; s += get_global_size(0)) {
        int original = order[s];
        for (int j = rows[original]; j < rows[original+1]; j++) {
            int row = columns[j]/32;
            if (row != s/32) atomic_add(counts+row, 1u);
        }
    }
}

__kernel void blockLayout(__global const unsigned int* counts, __global int2* rows) {
    if (get_global_id(0) != 0) return;
    int offset = 0;
    for (int row = 0; row < NUM_BLOCKS; row++) {
        unsigned int capacity = 2;
        while (capacity < 2*counts[row]) capacity *= 2;
        rows[row] = (int2)(offset, capacity);
        offset += capacity;
    }
}

__kernel void blockInsertExclusions(__global const int* order, __global const int* originalRows,
        __global const int* columns, __global const int2* rows, __global uint2* table, __global unsigned int* filters) {
    for (int s = get_global_id(0); s < NUM_ATOMS; s += get_global_size(0)) {
        int original = order[s];
        unsigned int hash = blockExclusionHash(s);
        for (int j = originalRows[original]; j < originalRows[original+1]; j++) {
            int other = columns[j], row = other/32;
            if (row == s/32) continue;
            int2 range = rows[row];
            unsigned int slot = hash & (range.y-1);
            while (true) {
                unsigned int key = atomic_cmpxchg(((__global unsigned int*) table)+2*(range.x+slot), 0u, (unsigned int) s+1);
                if (key == 0 || key == (unsigned int) s+1) {
                    atomic_or(((__global unsigned int*) table)+2*(range.x+slot)+1, 1u<<(other%32));
                    break;
                }
                slot = (slot+1)&(range.y-1);
            }
            atomic_or(filters+8*row+((hash&255)>>5), 1u<<(hash&31));
        }
    }
}
