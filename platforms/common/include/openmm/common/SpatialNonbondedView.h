#ifndef OPENMM_SPATIALNONBONDEDVIEW_H_
#define OPENMM_SPATIALNONBONDEDVIEW_H_
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "openmm/common/ArrayInterface.h"
namespace OpenMM {
/** A borrowed, full-tile work view for one force evaluation.
 *
 * Acquire after parameter producers, on the Context's execution queue, through
 * NonbondedUtilities::getSpatialWorkView(). Do not retain it across evaluations,
 * resizes or Context reinitialization. Register the consumer's force group with
 * addInteraction() so prepareInteractions() runs when that group is requested.
 * If another producer updates original data after acquisition, invalidate the
 * registered arrays and acquire again before consuming them.
 *
 * Positions and all work indices are spatial. spatialToOriginal[s] is the stable
 * atom ID; originalToSpatial[id] is its inverse. Positions are real4 (w is charge).
 * Only forces is writable: accumulate signed 64-bit fixed-point values scaled
 * by 2^32, in three padded-atom planes. The backend completes force merging and PME before virtual sites and
 * integrators read original forces; CUDA may overlap the merge with PME.
 *
 * Exclusion tiles are int2 atom-block pairs (32 atoms/block). Masks are 32 uint
 * words per tile; a cleared bit denotes an excluded pair. Their allocated size
 * is capacity, not the active count: read exclusionCount[0] on the device.
 * Rows and columns describe symmetric block adjacency. Never omit alternate
 * calculations required by a generic interaction's excluded-pair semantics.
 *
 * With a neighbor list, ordinaryBlocks[i] pairs with ordinaryAtoms[32*i:32*i+32].
 * interactionCount[0] gives active ordinary tiles. CUDA optionally also stores
 * int2 sparse pairs, counted by interactionCount[1]; OpenCL returns null for
 * singlePairs. Without a neighbor list these pointers are null; enumerate block
 * pairs, skipping the exclusion tiles. Packed-mask experiments are not exposed
 * through this full-tile contract.
 */
struct SpatialNonbondedView {
    ArrayInterface *positions, *forces;
    ArrayInterface *spatialToOriginal, *originalToSpatial;
    ArrayInterface *exclusionTiles, *exclusionMasks, *exclusionCount;
    ArrayInterface *exclusionRows, *exclusionColumns;
    ArrayInterface *ordinaryBlocks, *ordinaryAtoms, *interactionCount, *singlePairs;
    int numAtoms, paddedAtoms, orderRevision;
    bool usesNeighborList;
};
}
#endif
