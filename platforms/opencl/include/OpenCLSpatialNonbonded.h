#ifndef OPENMM_OPENCLSPATIALNONBONDED_H_
#define OPENMM_OPENCLSPATIALNONBONDED_H_
#include "OpenCLArray.h"
#include "openmm/Vec3.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeSort.h"
#include "openmm/common/ReorderedArraySet.h"
#include <map>
#include <memory>
namespace OpenMM {
class OpenCLNonbondedUtilities;
class System;
class OpenCLSpatialNonbonded {
public:
    OpenCLSpatialNonbonded(OpenCLContext& context, OpenCLNonbondedUtilities& nonbonded);
    void initialize(const System& system);
    void prepare();
    OpenCLArray& getInverseAtomOrder() { return inverseOrder; }
    bool usesBoundsGather() const;
    bool usesIdentityOrdering() const { return identityOrdering; }
    OpenCLArray& getPositions();
    OpenCLArray& getForces();
    bool usesPackedExclusions() const { return packedExclusions; }
    void resizeNeighborMasks();
    OpenCLArray neighborMasks, blockRows, blockTable, blockFilters;
    OpenCLArray& getAtomOrder() { return order; }
    void gatherParameters(bool includeForces);
    // Returns true if the original-order force reduction was also performed.
    bool mergeForces();
    void invalidateParameters();
    int getReorderCount() const { return reorderCount; }
    OpenCLArray positions, forces, exclusionCount;
    std::vector<std::unique_ptr<OpenCLArray> > parameters;
private:
    class AtomSortTrait;
    class ExclusionSortTrait;
    void reorder();
    void remapExclusions();
    void scan(OpenCLArray& values, int length, OpenCLArray& total);
    ComputeKernel bind(const std::string& name, std::vector<ArrayInterface*> arrays);
    void execute(const std::string& name, std::vector<ArrayInterface*> arrays, int size);
    OpenCLContext& cc;
    OpenCLNonbondedUtilities& nb;
    bool inverseMerge, forcesPending, packedExclusions, identityOrdering;
    OpenCLArray originalRows, originalColumns, mappedColumns, blockCounts;
    int reorderInterval, reorderCount, numRecords;
    OpenCLArray atomKeys, order, inverseOrder, originalExclusions, exclusionRecords;
    OpenCLArray tileIds, rowCounts, rowCursors, scanSums, scanOffsets, scanTotal;
    Vec3 activeBox[3];
    ComputeSort atomSorter, exclusionSorter;
    ReorderedArraySet* parameterArrays; // Borrowed from ComputeContext.
    ComputeKernel mergeReductionKernel;
    std::map<std::string, ComputeKernel> kernels;
    std::map<std::string, int> boundArgs;
};
}
#endif
