#ifndef OPENMM_CUDASPATIALNONBONDED_H_
#define OPENMM_CUDASPATIALNONBONDED_H_

#include "CudaArray.h"
#include "openmm/Vec3.h"
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
 * Hilbert ordering and measured execution selection are internal policies.
 */
class CudaSpatialNonbonded {
public:
    CudaSpatialNonbonded(CudaContext& context, CudaNonbondedUtilities& nonbonded);
    void initialize(const System& system);
    void prepare();
    void releaseUnusedExecutionBuffers();
    CudaArray& getAtomOrder() { return order; }
    CudaArray& getInverseAtomOrder() { return inverseOrder; }
    void gatherParameters(bool includeForces);
    void mergeForces();
    void invalidateParameters();
    int getReorderCount() const { return reorderCount; }
    bool usesPackedExclusions() const { return packedExclusions; }
    bool usesExclusionFilter() const { return exclusionFilter; }
    bool usesDirectForces() const { return directForces; }
    bool usesIndexedPositions() const { return indexedPositions; }
    bool usesIdentityOrder() const { return identityOrdering; }
    bool gathersInBounds() const { return boundsGather; }
    int getBoundsTileLanes() const { return boundsTileLanes; }
    void resizeNeighborMasks();
    int verifyBlockTable();
    CudaArray positions, forces, exclusionCount, neighborMasks;
    std::vector<std::unique_ptr<CudaArray> > parameters;
private:
    friend class CudaNonbondedUtilities;
    void initializePackedExclusions();
    void bindExecution();
    bool originalNeighborList;
    std::vector<void*> baseForceArgs, baseBoundsArgs, baseSortArgs, baseNeighborArgs;
    class AtomSortTrait;
    class ExclusionSortTrait;
    void reorder();
    void remapExclusions();
    void scan(CudaArray& values, int length, CudaArray& total);
    void execute(const std::string& name, std::vector<void*> args, int size);
    CudaContext& cc;
    CudaNonbondedUtilities& nb;
    bool reorderRequested, forcesPending;
    bool fused, packedExclusions, directForces;
    bool exclusionFilter;
    int reorderInterval, reorderCount;
    CudaArray atomKeys, order, inverseOrder, originalExclusions, exclusionRecords;
    CudaArray tileIds, rowCounts, rowCursors, scanSums, scanOffsets, scanTotal;
    CudaArray originalRows, originalColumns, mappedColumns;
    CudaArray exclusionBlockFilter;
    CudaArray blockRows, blockCounts, blockTable, blockFilters, blockErrors;
    int forceMaskArgIndex, neighborMaskArgIndex, neighborFilterArgIndex;
    ComputeSort atomSorter, exclusionSorter;
    ReorderedArraySet* parameterArrays; // Borrowed from ComputeContext.
    std::map<std::string, CUfunction> kernels;
    std::map<int, std::map<std::string, CUfunction> > executionKernels;
    std::vector<void*> parameterArgs;
    int numRecords;
    bool boundsGather, indexedPositions, identityOrdering;
    Vec3 activeBox[3];
    int boundsTileLanes;
};
}
#endif
