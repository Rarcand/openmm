#ifndef OPENMM_CUDASPATIALNONBONDED_H_
#define OPENMM_CUDASPATIALNONBONDED_H_

#include "CudaArray.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeSort.h"
#include "openmm/common/ReorderedArraySet.h"
#include <map>
#include <memory>

namespace OpenMM {
class CudaNonbondedUtilities;
class System;

/** Experimental spatial view for the CUDA default nonbonded kernel.
 * Original positions, velocities, PME, exceptions, and bonded arrays retain atom IDs.
 * Uses unrestricted GPU Hilbert sorting and inverse force merging.
 */
class CudaSpatialNonbonded {
public:
    CudaSpatialNonbonded(CudaContext& context, CudaNonbondedUtilities& nonbonded, bool inverse);
    void initialize(const System& system);
    void prepare();
    CudaArray& getAtomOrder() { return order; }
    CudaArray& getInverseAtomOrder() { return inverseOrder; }
    void gatherParameters(bool includeForces);
    void mergeForces();
    void invalidateParameters();
    void requestReorder() { reorderRequested = true; }
    int getReorderCount() const { return reorderCount; }
    bool gathersInBounds() const { return boundsGather; }
    int getBoundsTileLanes() const { return boundsTileLanes; }
    CudaArray positions, forces, exclusionCount;
    std::vector<std::unique_ptr<CudaArray> > parameters;
private:
    class AtomSortTrait;
    class ExclusionSortTrait;
    void reorder();
    void remapExclusions();
    void scan(CudaArray& values, int length, CudaArray& total);
    void execute(const std::string& name, std::vector<void*> args, int size);
    CudaContext& cc;
    CudaNonbondedUtilities& nb;
    bool inverseMerge, reorderRequested, forcesPending;
    int reorderInterval, reorderCount;
    CudaArray atomKeys, order, inverseOrder, originalExclusions, exclusionRecords;
    CudaArray tileIds, rowCounts, rowCursors, scanSums, scanOffsets, scanTotal;
    ComputeSort atomSorter, exclusionSorter;
    ReorderedArraySet* parameterArrays; // Borrowed from ComputeContext.
    std::map<std::string, CUfunction> kernels;
    std::vector<void*> parameterArgs;
    int numRecords;
    bool boundsGather;
    int boundsTileLanes;
};
}
#endif
