// Experimental CUDA implementation of issue #5406.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaSpatialNonbonded.h"
#include "CudaContext.h"
#include "CudaNonbondedUtilities.h"
#include "CudaKernelSources.h"
#include "CommonKernelSources.h"
#include "OpenMM.h"
#include "openmm/common/SpatialNonbondedPolicy.h"
#include <algorithm>
#include <limits>
#include <sstream>

using namespace OpenMM;
using namespace std;

class CudaSpatialNonbonded::AtomSortTrait : public ComputeSortImpl::SortTrait {
public:
    int getDataSize() const { return sizeof(mm_int2); }
    int getKeySize() const { return 8; }
    const char* getDataType() const { return "int2"; }
    const char* getKeyType() const { return "unsigned long long"; }
    const char* getMinKey() const { return "0ull"; }
    const char* getMaxKey() const { return "0xFFFFFFFFFFFFFFFFull"; }
    const char* getMaxValue() const { return "make_int2(-1, -1)"; }
    const char* getSortKey() const { return "(((unsigned long long)(unsigned int)value.x)<<32) | (unsigned int)value.y"; }
};

class CudaSpatialNonbonded::ExclusionSortTrait : public ComputeSortImpl::SortTrait {
public:
    int getDataSize() const { return sizeof(mm_int4); }
    int getKeySize() const { return 4; }
    const char* getDataType() const { return "int4"; }
    const char* getKeyType() const { return "int"; }
    const char* getMinKey() const { return "0"; }
    const char* getMaxKey() const { return "2147483647"; }
    const char* getMaxValue() const { return "make_int4(2147483647, 0, 0, 0)"; }
    const char* getSortKey() const { return "value.x"; }
};

CudaSpatialNonbonded::CudaSpatialNonbonded(CudaContext& context, CudaNonbondedUtilities& nonbonded, bool inverse) :
        cc(context), nb(nonbonded), inverseMerge(true), reorderRequested(true), forcesPending(false), reorderCount(0), parameterArrays(nullptr) {
    boundsGather = true;
    boundsTileLanes = 8;
    reorderInterval = 250;
}

void CudaSpatialNonbonded::initialize(const System& system) {
    if (cc.getPlatformData().contexts.size() != 1)
        throw OpenMMException("Experimental AtomReordering supports one CUDA device only");
    if (cc.getNumAtoms() == 0 || !nb.useCutoff || !nb.usePeriodic)
        throw OpenMMException("Experimental AtomReordering requires a periodic nonbonded cutoff");
    std::string reason = SpatialNonbondedPolicy::unsupportedReason(system);
    if (!reason.empty()) throw OpenMMException(reason);
    int n = cc.getNumAtoms(), padded = cc.getPaddedNumAtoms(), blocks = cc.getNumAtomBlocks();
    if ((long long) blocks*blocks >= numeric_limits<int>::max())
        throw OpenMMException("Too many atoms for experimental spatial exclusion keys");
    vector<mm_int2> pairs;
    bool compact = true;
    for (int i = 0; i < n; i++) {
        for (int j : nb.atomExclusions[i])
            if (!compact || i <= j)
                pairs.push_back(mm_int2(i, j));
    }
    if (pairs.empty() || pairs.size() > numeric_limits<int>::max()/32)
        throw OpenMMException("Invalid spatial exclusion capacity");
    numRecords = pairs.size();
    // Cache short adjacency rows; the neighbor-list kernel reads longer rows
    // from global memory. No spatial permutation requires recompilation.
    nb.maxExclusions = min(blocks, 32);
    positions.initialize(cc, padded, cc.getPosq().getElementSize(), "spatialPositions");
    forces.initialize<long long>(cc, 3*padded, "spatialForces");
    atomKeys.initialize<mm_int2>(cc, n, "spatialKeys");
    order.initialize<int>(cc, padded, "spatialOrder");
    inverseOrder.initialize<int>(cc, padded, "spatialInverseOrder");
    exclusionCount.initialize<int>(cc, 1, "spatialExclusionCount");
    atomSorter = cc.createSort(new AtomSortTrait(), n, false);

    originalExclusions.initialize<mm_int2>(cc, numRecords, "originalExclusions");
    originalExclusions.upload(pairs);
    exclusionRecords.initialize<mm_int4>(cc, numRecords, "spatialExclusionRecords");
    tileIds.initialize<int>(cc, numRecords, "spatialTileIds");
    rowCounts.initialize<int>(cc, blocks, "spatialRowCounts");
    rowCursors.initialize<int>(cc, blocks, "spatialRowCursors");
    int chunks = (max(numRecords, blocks)+255)/256;
    scanSums.initialize<int>(cc, chunks, "spatialScanSums");
    scanOffsets.initialize<int>(cc, chunks, "spatialScanOffsets");
    scanTotal.initialize<int>(cc, 1, "spatialScanTotal");
    nb.exclusionTiles.resize(numRecords);
    nb.exclusions.resize(32*(size_t) numRecords);
    nb.exclusionIndices.resize(2*(size_t) numRecords);
    exclusionSorter = cc.createSort(new ExclusionSortTrait(), numRecords, false);

    // Small standard-force systems can omit the neighbor list altogether.
    // Their positions still need gathering, so retain the standalone pass.
    boundsGather = boundsGather && nb.useNeighborList && nb.numTiles > 0;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(n);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(padded);
    defines["NUM_BLOCKS"] = cc.intToString(blocks);
    defines["NUM_EXCLUSIONS"] = cc.intToString(numRecords);

    if (inverseMerge) defines["INVERSE_MERGE"] = "1";
    if (compact) defines["COMPACT_EXCLUSIONS"] = "1";
    defines["FUSED_SPATIAL"] = "1";
    defines["HILBERT_SPATIAL"] = "1";
    parameterArrays = &cc.getReorderedArraySet();
    for (int i = 0; i < nb.parameters.size(); i++) {
        ComputeParameterInfo& param = nb.parameters[i];
        if (!param.isConstant() || param.getArray().getSize() < n)
            throw OpenMMException("Unsupported spatial per-particle parameter: "+param.getName());
        parameters.push_back(unique_ptr<CudaArray>(new CudaArray(cc, padded, param.getSize(), "spatial_"+param.getName())));
        cc.clearBuffer(*parameters.back());
        cc.addReorderedArray(param.getArray(), *parameters.back());
    }
    parameterArrays->seal();
    // Parameters may be produced by a force kernel after prepareInteractions().
    // Refresh them (and posq.w charges) immediately before the default pair kernel.
    // Producers invalidate this cache for updateParametersInContext and offsets.
    string source = CudaKernelSources::vectorOps+CudaKernelSources::spatialNonbonded;
    source += "\nextern \"C\" __global__ void spatialParameters(const real4* original, real4* sorted, const int* order"+parameterArrays->getArguments("")+") {\n"
              "for (int s=blockIdx.x*blockDim.x+threadIdx.x; s<NUM_ATOMS; s+=blockDim.x*gridDim.x) {\n"
              "sorted[s].w=original[order[s]].w;\n"+parameterArrays->getGatherSource("s", "order[s]")+"}}\n";
    CUmodule module = cc.createModule(source, defines);
    for (const string& name : {"spatialKeys", "spatialMaps", "spatialPositions", "spatialMerge", "spatialExclusionKeys",
            "spatialMarkTiles", "spatialScanBlocks", "spatialScanOffsets", "spatialBuildTiles", "spatialRows",
            "spatialInitMasks", "spatialMasks", "spatialAdjacency", "spatialParameters"})
        kernels[name] = cc.getKernel(module, name);
    nb.forceArgs[0] = &forces.getDevicePointer();
    nb.forceArgs[2] = &positions.getDevicePointer();
    nb.forceArgs.push_back(&exclusionCount.getDevicePointer());

    nb.findBlockBoundsArgs[6] = &positions.getDevicePointer();
    if (boundsGather) {
        nb.findBlockBoundsArgs.push_back(&cc.getPosq().getDevicePointer());
        nb.findBlockBoundsArgs.push_back(&order.getDevicePointer());
        nb.findBlockBoundsArgs.push_back(&forces.getDevicePointer());
    }
    nb.sortBoxDataArgs[nb.useLargeBlocks ? 12 : 5] = &positions.getDevicePointer();
    nb.findInteractingBlocksArgs[9] = &positions.getDevicePointer();
    parameterArgs = {&cc.getPosq().getDevicePointer(), &positions.getDevicePointer(), &order.getDevicePointer()};
    for (const auto& entry : parameterArrays->getEntries()) {
        parameterArgs.push_back(&cc.unwrap(*entry.original).getDevicePointer());
        parameterArgs.push_back(&cc.unwrap(*entry.sorted).getDevicePointer());
    }
}

void CudaSpatialNonbonded::execute(const string& name, vector<void*> args, int size) {
    cc.executeKernel(kernels.at(name), args.data(), size, 256);
}

void CudaSpatialNonbonded::scan(CudaArray& values, int length, CudaArray& total) {
    execute("spatialScanBlocks", {&values.getDevicePointer(), &scanSums.getDevicePointer(), &length}, length);
    int chunks = (length+255)/256;
    execute("spatialScanOffsets", {&scanSums.getDevicePointer(), &scanOffsets.getDevicePointer(), &total.getDevicePointer(), &chunks}, 1);
}

void CudaSpatialNonbonded::reorder() {
    nb.beginPhase("spatial_sort_and_maps");
    vector<void*> keyArgs = {&cc.getPosq().getDevicePointer(), &atomKeys.getDevicePointer(),
            cc.getPeriodicBoxVecXPointer(), cc.getPeriodicBoxVecYPointer(), cc.getPeriodicBoxVecZPointer()};

    execute("spatialKeys", keyArgs, cc.getNumAtoms());
    atomSorter->sort(atomKeys);
    vector<void*> mapArgs = {&atomKeys.getDevicePointer(), &order.getDevicePointer(), &inverseOrder.getDevicePointer()};
    execute("spatialMaps", mapArgs, cc.getPaddedNumAtoms());

    nb.endPhase();
    remapExclusions();
}

void CudaSpatialNonbonded::remapExclusions() {
    nb.beginPhase("exclusion_remap");

    execute("spatialExclusionKeys", {&originalExclusions.getDevicePointer(), &inverseOrder.getDevicePointer(), &exclusionRecords.getDevicePointer()}, numRecords);
    exclusionSorter->sort(exclusionRecords);
    execute("spatialMarkTiles", {&exclusionRecords.getDevicePointer(), &tileIds.getDevicePointer()}, numRecords);
    scan(tileIds, numRecords, exclusionCount);
    cc.clearBuffer(rowCounts);
    execute("spatialBuildTiles", {&exclusionRecords.getDevicePointer(), &tileIds.getDevicePointer(), &scanOffsets.getDevicePointer(),
            &nb.exclusionTiles.getDevicePointer(), &rowCounts.getDevicePointer()}, numRecords);
    scan(rowCounts, cc.getNumAtomBlocks(), scanTotal);
    execute("spatialRows", {&rowCounts.getDevicePointer(), &scanOffsets.getDevicePointer(), &nb.exclusionRowIndices.getDevicePointer()}, cc.getNumAtomBlocks());
    execute("spatialInitMasks", {&exclusionCount.getDevicePointer(), &nb.exclusions.getDevicePointer()}, numRecords);
    execute("spatialMasks", {&exclusionRecords.getDevicePointer(), &tileIds.getDevicePointer(), &nb.exclusions.getDevicePointer()}, numRecords);
    cc.clearBuffer(rowCursors);
    execute("spatialAdjacency", {&nb.exclusionTiles.getDevicePointer(), &exclusionCount.getDevicePointer(),
            &nb.exclusionRowIndices.getDevicePointer(), &rowCursors.getDevicePointer(), &nb.exclusionIndices.getDevicePointer()}, numRecords);

    nb.endPhase();
    nb.forceRebuildNeighborList = true;
    cc.setStepsSinceReorder(0);
    reorderRequested = false;
    cc.invalidateReorderedArrays();
    reorderCount++;
    if (nb.diagnostics)
        exclusionCount.download(&nb.diagnosticExclusionTiles);
}

void CudaSpatialNonbonded::prepare() {
    forcesPending = false;
    if (reorderRequested || cc.getStepsSinceReorder() >= reorderInterval)
        reorder();
    if (boundsGather)
        return;
    nb.beginPhase("position_gather_and_clear");
    execute("spatialPositions", {&cc.getPosq().getDevicePointer(), &positions.getDevicePointer(), &order.getDevicePointer(), &forces.getDevicePointer()}, cc.getPaddedNumAtoms());
    nb.endPhase();
}

void CudaSpatialNonbonded::invalidateParameters() {
    cc.invalidateReorderedArrays();
}

void CudaSpatialNonbonded::gatherParameters(bool includeForces) {
    if (includeForces && !forcesPending) {

        forcesPending = true;
    }
    if (parameterArrays->isDirty()) {
        parameterArrays->validate();
        nb.beginPhase("parameter_gather");
        execute("spatialParameters", parameterArgs, cc.getNumAtoms());
        nb.endPhase();
        parameterArrays->markGathered();
    }
}

void CudaSpatialNonbonded::mergeForces() {
    if (!forcesPending)
        return;
    // Called after ForcePostComputations, including the PME queue wait. No original
    // force writer can overlap this non-atomic one-to-one merge.
    nb.beginPhase("force_merge");
    execute("spatialMerge", {&cc.getForce().getDevicePointer(), &forces.getDevicePointer(),
            &(inverseMerge ? inverseOrder.getDevicePointer() : order.getDevicePointer())}, cc.getNumAtoms());
    nb.endPhase();
    forcesPending = false;
}
