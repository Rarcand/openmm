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
    int getDataSize() const { return sizeof(mm_int2); }
    int getKeySize() const { return 4; }
    const char* getDataType() const { return "int2"; }
    const char* getKeyType() const { return "int"; }
    const char* getMinKey() const { return "0"; }
    const char* getMaxKey() const { return "2147483647"; }
    const char* getMaxValue() const { return "make_int2(2147483647, 0)"; }
    const char* getSortKey() const { return "value.x"; }
};

CudaSpatialNonbonded::CudaSpatialNonbonded(CudaContext& context, CudaNonbondedUtilities& nonbonded) :
        cc(context), nb(nonbonded), reorderRequested(true), forcesPending(false), reorderCount(0), parameterArrays(nullptr) {
    const auto& properties = cc.getPlatformData().propertyValues;
    indexedPositions = false;
    identityOrdering = false;
    directForces = properties.at("AtomReorderingForceAccumulation") == "direct";
    boundsGather = true;
    boundsTileLanes = 8;
    fused = !directForces;
    packedExclusions = properties.at("AtomReorderingPackedExclusions") == "true";
    exclusionFilter = true;
    reorderInterval = 250;
}

void CudaSpatialNonbonded::initialize(const System& system) {
    if (cc.getPlatformData().contexts.size() != 1)
        throw OpenMMException("Experimental AtomReordering supports one CUDA device only");
    if (cc.getNumAtoms() == 0 || !nb.useCutoff || !nb.usePeriodic)
        throw OpenMMException("Experimental AtomReordering requires a periodic nonbonded cutoff");
    std::string reason = SpatialNonbondedPolicy::unsupportedReason(system);
    if (!reason.empty()) throw OpenMMException(reason);
    // Exhaustive traversal gains no geometric pruning from a spatial permutation.
    identityOrdering = !nb.useNeighborList && nb.canUseIndexedPositions;
    if (identityOrdering) {
        directForces = true;
        fused = false;
        cc.getPlatformData().propertyValues["AtomReorderingForceAccumulation"] = "direct";
    }
    if (identityOrdering && (nb.useNeighborList || !nb.canUseIndexedPositions))
        throw OpenMMException("Identity ordering requires exhaustive standard nonbonded execution");
    int n = cc.getNumAtoms(), padded = cc.getPaddedNumAtoms(), blocks = cc.getNumAtomBlocks();

    vector<mm_int2> pairs;
    for (int i = 0; i < n; i++) {
        for (int j : nb.atomExclusions[i])
            if (i <= j)
                pairs.push_back(mm_int2(i, j));
    }
    if (pairs.empty() || pairs.size() > numeric_limits<int>::max()/32)
        throw OpenMMException("Invalid spatial exclusion capacity");
    numRecords = pairs.size();
    // Cache short adjacency rows; the neighbor-list kernel reads longer rows
    // from global memory. No spatial permutation requires recompilation.
    nb.maxExclusions = min(blocks, 32);
    positions.initialize(cc, padded, cc.getPosq().getElementSize(), "spatialPositions");
    // Direct accumulation only needs a placeholder for the unused gather-kernel
    // argument; all real force writes target the authoritative original buffer.
    forces.initialize<long long>(cc, directForces ? 1 : 3*padded, "spatialForces");
    atomKeys.initialize<mm_int2>(cc, n, "spatialKeys");
    order.initialize<int>(cc, padded, "spatialOrder");
    inverseOrder.initialize<int>(cc, padded, "spatialInverseOrder");
    exclusionCount.initialize<int>(cc, 1, "spatialExclusionCount");
    atomSorter = cc.createSort(new AtomSortTrait(), n, false);

    packedExclusions = nb.useNeighborList && nb.canUseIndexedPositions;
    // Packed masks do not encode a quadratic block-pair key.
    if (!packedExclusions && (long long) blocks*blocks >= numeric_limits<int>::max())
        throw OpenMMException("Too many atoms for experimental spatial exclusion keys");
    if (packedExclusions) {
        initializePackedExclusions();
    }
    else {
        originalExclusions.initialize<mm_int2>(cc, numRecords, "originalExclusions");
        originalExclusions.upload(pairs);
        exclusionRecords.initialize<mm_int2>(cc, numRecords, "spatialExclusionRecords");
        tileIds.initialize<int>(cc, numRecords, "spatialTileIds");
        rowCounts.initialize<int>(cc, blocks, "spatialRowCounts");
        rowCursors.initialize<int>(cc, blocks, "spatialRowCursors");
        int chunks = (max(numRecords, blocks)+255)/256;
        scanSums.initialize<int>(cc, chunks, "spatialScanSums");
        scanOffsets.initialize<int>(cc, chunks, "spatialScanOffsets");
        scanTotal.initialize<int>(cc, 1, "spatialScanTotal");
        // Identity ordering preserves the original exclusion-tile membership.
        if (!identityOrdering)
            nb.exclusionTiles.resize(numRecords);
        nb.exclusions.resize(32*(size_t) nb.exclusionTiles.getSize());
        nb.exclusionIndices.resize(2*(size_t) numRecords);
        exclusionSorter = cc.createSort(new ExclusionSortTrait(), numRecords, false);
    }

    originalNeighborList = nb.useNeighborList;
    baseForceArgs = nb.forceArgs;
    baseBoundsArgs = nb.findBlockBoundsArgs;
    baseSortArgs = nb.sortBoxDataArgs;
    baseNeighborArgs = nb.findInteractingBlocksArgs;
    bindExecution();
}

void CudaSpatialNonbonded::initializePackedExclusions() {
    int n = cc.getNumAtoms(), padded = cc.getPaddedNumAtoms(), blocks = cc.getNumAtomBlocks();
    if (!nb.canOmitExcludedPairs)
        throw OpenMMException("Packed exclusions require omission support from every nonbonded interaction");
    // Original-ID CSR is the authoritative exclusion set, including self.
    // Only the column values need mapping when the spatial order changes.
    if (!originalRows.isInitialized()) {
        vector<int> rows(n+1, 0), columns;
        for (int i = 0; i < n; i++) {
            columns.insert(columns.end(), nb.atomExclusions[i].begin(), nb.atomExclusions[i].end());
            rows[i+1] = columns.size();
        }
        numRecords = columns.size();
        originalRows.initialize<int>(cc, rows.size(), "spatialOriginalRows");
        originalColumns.initialize<int>(cc, numRecords, "spatialOriginalColumns");
        mappedColumns.initialize<int>(cc, numRecords, "spatialMappedColumns");
        if (exclusionFilter)
            exclusionBlockFilter.initialize<unsigned int>(cc, padded, "spatialExclusionBlockFilter");
        blockRows.initialize<mm_int2>(cc, blocks, "blockRows");
        blockCounts.initialize<unsigned int>(cc, blocks, "blockCounts");
        blockTable.initialize<mm_int2>(cc, 4*(size_t) numRecords+2*blocks, "blockTable");
        blockFilters.initialize<unsigned int>(cc, 8*blocks, "blockFilters");
        blockErrors.initialize<int>(cc, 1, "blockErrors");
        originalRows.upload(rows);
        originalColumns.upload(columns);
    }
    numRecords = originalColumns.getSize();
    nb.useNeighborList = true;
    nb.maxExclusions = 1;
    nb.exclusionTiles.resize(blocks);
    nb.exclusions.resize(padded);
    vector<mm_int2> diagonal(blocks);
    for (int i = 0; i < blocks; i++) diagonal[i] = mm_int2(i, i);
    nb.exclusionTiles.upload(diagonal);
    exclusionCount.upload(&blocks);
    if (!neighborMasks.isInitialized())
        neighborMasks.initialize<unsigned int>(cc, 32*(size_t) nb.maxTiles, "spatialNeighborMasks");
    else
        neighborMasks.resize(32*(size_t) nb.maxTiles);

}

void CudaSpatialNonbonded::bindExecution() {
    int n = cc.getNumAtoms(), padded = cc.getPaddedNumAtoms(), blocks = cc.getNumAtomBlocks();
    // Small standard-force systems can omit the neighbor list altogether.
    // Their positions still need gathering, so retain the standalone pass.
    boundsGather = boundsGather && nb.useNeighborList && nb.numTiles > 0;
    indexedPositions = directForces && !nb.useNeighborList && nb.canUseIndexedPositions;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(n);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(padded);
    defines["NUM_BLOCKS"] = cc.intToString(blocks);
    defines["NUM_EXCLUSIONS"] = cc.intToString(numRecords);

    if (identityOrdering) defines["IDENTITY_SPATIAL"] = "1";
    defines["INVERSE_MERGE"] = "1";
    defines["COMPACT_EXCLUSIONS"] = "1";
    if (fused) defines["FUSED_SPATIAL"] = "1";
    if (exclusionFilter) defines["EXCLUSION_BLOCK_FILTER"] = "1";
    defines["HILBERT_SPATIAL"] = "1";
    if (parameterArrays == nullptr) {
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
    }
    // Parameters may be produced by a force kernel after prepareInteractions().
    // Refresh them (and posq.w charges) immediately before the default pair kernel.
    // Producers invalidate this cache for updateParametersInContext and offsets.
    string source = CudaKernelSources::vectorOps+CudaKernelSources::spatialNonbonded;
    if (exclusionFilter)
        source = CommonKernelSources::spatialExclusionFilter+source;
    source += "\nextern \"C\" __global__ void spatialParameters(const real4* original, real4* sorted, const int* order"+parameterArrays->getArguments("")+") {\n"
              "for (int s=blockIdx.x*blockDim.x+threadIdx.x; s<NUM_ATOMS; s+=blockDim.x*gridDim.x) {\n"
              "sorted[s].w=original[order[s]].w;\n"+parameterArrays->getGatherSource("s", "order[s]")+"}}\n";
    int variant = (fused ? 1 : 0)+(exclusionFilter ? 2 : 0)+(identityOrdering ? 4 : 0);
    if (executionKernels.find(variant) == executionKernels.end()) {
        CUmodule module = cc.createModule(source, defines);
        for (const string& name : {"spatialKeys", "spatialMaps", "spatialPositions", "spatialMerge", "spatialExclusionKeys",
                "spatialMarkTiles", "spatialScanBlocks", "spatialScanOffsets", "spatialBuildTiles", "spatialRows",
                "spatialInitMasks", "spatialMasks", "spatialAdjacency", "spatialParameters",
                "spatialMapExclusions", "spatialDiagonalMasks", "blockCountExclusions", "blockLayout", "blockInsertExclusions", "blockVerifyExclusions"})
            kernels[name] = cc.getKernel(module, name);
        executionKernels[variant] = kernels;
    }
    else
        kernels = executionKernels.at(variant);
    nb.forceArgs[0] = &(directForces ? cc.unwrap(cc.getForce()).getDevicePointer() : forces.getDevicePointer());
    nb.forceArgs[2] = &(indexedPositions ? cc.unwrap(cc.getPosq()).getDevicePointer() : positions.getDevicePointer());
    nb.forceArgs.push_back(&exclusionCount.getDevicePointer());
    if (packedExclusions) {
        forceMaskArgIndex = nb.forceArgs.size();
        nb.forceArgs.push_back(&neighborMasks.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&order.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&originalRows.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&mappedColumns.getDevicePointer());
        neighborMaskArgIndex = nb.findInteractingBlocksArgs.size();
        nb.findInteractingBlocksArgs.push_back(&neighborMasks.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&blockRows.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&blockTable.getDevicePointer());
        nb.findInteractingBlocksArgs.push_back(&blockFilters.getDevicePointer());
        if (exclusionFilter) {
            neighborFilterArgIndex = nb.findInteractingBlocksArgs.size();
            nb.findInteractingBlocksArgs.push_back(&exclusionBlockFilter.getDevicePointer());
        }
    }

    if (directForces)
        nb.forceArgs.push_back(&order.getDevicePointer());
    nb.findBlockBoundsArgs[6] = &positions.getDevicePointer();
    if (boundsGather) {
        nb.findBlockBoundsArgs.push_back(&cc.unwrap(cc.getPosq()).getDevicePointer());
        nb.findBlockBoundsArgs.push_back(&order.getDevicePointer());
        if (!directForces)
            nb.findBlockBoundsArgs.push_back(&forces.getDevicePointer());
        if (packedExclusions) {
            cc.addAutoclearBuffer(nb.rebuildNeighborList);
            nb.findBlockBoundsArgs.push_back(&nb.sortedBlockCenter.getDevicePointer());
            nb.findBlockBoundsArgs.push_back(&nb.sortedBlockBoundingBox.getDevicePointer());
            nb.findBlockBoundsArgs.push_back(&nb.oldPositions.getDevicePointer());
            nb.findBlockBoundsArgs.push_back(&nb.interactionCount.getDevicePointer());
            nb.findBlockBoundsArgs.push_back(&nb.forceRebuildNeighborList);
        }
    }
    nb.sortBoxDataArgs[nb.useLargeBlocks ? 12 : 5] = &positions.getDevicePointer();
    nb.findInteractingBlocksArgs[9] = &positions.getDevicePointer();
    parameterArgs = {&cc.unwrap(cc.getPosq()).getDevicePointer(), &positions.getDevicePointer(), &order.getDevicePointer()};
    for (const auto& entry : parameterArrays->getEntries()) {
        parameterArgs.push_back(&cc.unwrap(*entry.original).getDevicePointer());
        parameterArgs.push_back(&cc.unwrap(*entry.sorted).getDevicePointer());
    }
}

void CudaSpatialNonbonded::releaseUnusedExecutionBuffers() {
    // Profiling is finished. Keep only the selected representation's storage.
    vector<CudaArray*> unused;
    if (packedExclusions) {
        unused = {&originalExclusions, &exclusionRecords, &tileIds, &rowCounts, &rowCursors,
                  &scanSums, &scanOffsets, &scanTotal, &nb.exclusionIndices};
        exclusionSorter = ComputeSort();
    }
    else
        unused = {&originalRows, &originalColumns, &mappedColumns, &exclusionBlockFilter, &neighborMasks};
    if (directForces) unused.push_back(&forces);
    for (CudaArray* array : unused)
        if (array->isInitialized()) array->resize(1);
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
    vector<void*> keyArgs = {&cc.unwrap(cc.getPosq()).getDevicePointer(), &atomKeys.getDevicePointer(),
            cc.getPeriodicBoxVecXPointer(), cc.getPeriodicBoxVecYPointer(), cc.getPeriodicBoxVecZPointer()};

    execute("spatialKeys", keyArgs, cc.getNumAtoms());
    if (!identityOrdering) atomSorter->sort(atomKeys);
    vector<void*> mapArgs = {&atomKeys.getDevicePointer(), &order.getDevicePointer(), &inverseOrder.getDevicePointer()};
    execute("spatialMaps", mapArgs, cc.getPaddedNumAtoms());

    remapExclusions();
}

void CudaSpatialNonbonded::remapExclusions() {
    if (packedExclusions) {
        execute("spatialMapExclusions", {&originalColumns.getDevicePointer(), &inverseOrder.getDevicePointer(),
                &mappedColumns.getDevicePointer()}, numRecords);
        vector<void*> diagonalArgs = {&order.getDevicePointer(), &originalRows.getDevicePointer(),
                &mappedColumns.getDevicePointer(), &nb.exclusions.getDevicePointer()};
        if (exclusionFilter)
            diagonalArgs.push_back(&exclusionBlockFilter.getDevicePointer());
        execute("spatialDiagonalMasks", diagonalArgs, cc.getPaddedNumAtoms());
        cc.clearBuffer(blockCounts);
        cc.clearBuffer(blockFilters);
        cc.clearBuffer(blockTable);
        execute("blockCountExclusions", {&order.getDevicePointer(), &originalRows.getDevicePointer(),
                &mappedColumns.getDevicePointer(), &blockCounts.getDevicePointer()}, cc.getNumAtoms());
        execute("blockLayout", {&blockCounts.getDevicePointer(), &blockRows.getDevicePointer()}, 1);
        execute("blockInsertExclusions", {&order.getDevicePointer(), &originalRows.getDevicePointer(),
                &mappedColumns.getDevicePointer(), &blockRows.getDevicePointer(),
                &blockTable.getDevicePointer(), &blockFilters.getDevicePointer()}, cc.getNumAtoms());
    }
    else {
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
    }
    nb.forceRebuildNeighborList = true;
    cc.setStepsSinceReorder(0);
    reorderRequested = false;
    cc.invalidateReorderedArrays();
    reorderCount++;
}

void CudaSpatialNonbonded::prepare() {
    forcesPending = false;
    if (packedExclusions) {
        Vec3 box[3];
        cc.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                if (box[i][j] != activeBox[i][j]) nb.forceRebuildNeighborList = true;
            activeBox[i] = box[i];
        }
    }
    if (reorderRequested || (!identityOrdering && cc.getStepsSinceReorder() >= reorderInterval))
        reorder();
    else if (identityOrdering && cc.getStepsSinceReorder() >= reorderInterval)
        cc.setStepsSinceReorder(0);
    if (boundsGather || indexedPositions)
        return;
    execute("spatialPositions", {&cc.unwrap(cc.getPosq()).getDevicePointer(), &positions.getDevicePointer(), &order.getDevicePointer(), &forces.getDevicePointer()}, cc.getPaddedNumAtoms());
}

void CudaSpatialNonbonded::invalidateParameters() {
    cc.invalidateReorderedArrays();
}

void CudaSpatialNonbonded::gatherParameters(bool includeForces) {
    if (includeForces && !directForces && !forcesPending) {
        if (!fused) {
            cc.clearBuffer(forces);
        }
        forcesPending = true;
    }
    if (parameterArrays->isDirty()) {
        parameterArrays->validate();
        execute("spatialParameters", parameterArgs, cc.getNumAtoms());
        parameterArrays->markGathered();
    }
}

void CudaSpatialNonbonded::mergeForces() {
    if (!forcesPending)
        return;
    // Merge on the main stream before the PME wait. Atomic addition permits
    // the PME stream to finish writing original-order forces concurrently.
    execute("spatialMerge", {&cc.unwrap(cc.getForce()).getDevicePointer(), &forces.getDevicePointer(),
            &inverseOrder.getDevicePointer()}, cc.getNumAtoms());
    forcesPending = false;
}

void CudaSpatialNonbonded::resizeNeighborMasks() {
    if (!packedExclusions) return;
    neighborMasks.resize(32*(size_t) nb.maxTiles);
    // Bind by the registered slot, not by a trailing offset: optional arrays
    // can follow the masks and must not be replaced when the masks grow.
    nb.forceArgs[forceMaskArgIndex] = &neighborMasks.getDevicePointer();
    nb.findInteractingBlocksArgs[neighborMaskArgIndex] = &neighborMasks.getDevicePointer();
}



int CudaSpatialNonbonded::verifyBlockTable() {
    if (!packedExclusions) return 0;
    cc.clearBuffer(blockErrors);
    execute("blockVerifyExclusions", {&order.getDevicePointer(), &originalRows.getDevicePointer(),
            &mappedColumns.getDevicePointer(), &blockRows.getDevicePointer(),
            &blockTable.getDevicePointer(), &blockFilters.getDevicePointer(), &blockErrors.getDevicePointer()}, cc.getNumAtoms());
    int errors = 0;
    blockErrors.download(&errors);
    return errors;
}
