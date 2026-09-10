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
        ) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
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
