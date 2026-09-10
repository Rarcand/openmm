// Experimental OpenCL spatial view for OpenMM #5406.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "OpenCLSpatialNonbonded.h"
#include "OpenCLContext.h"
#include "OpenCLNonbondedUtilities.h"
#include "CommonKernelSources.h"
#include "OpenCLKernelSources.h"
#include "OpenMM.h"
#include "openmm/common/SpatialNonbondedPolicy.h"
#include <algorithm>
#include <limits>
#include <sstream>
using namespace OpenMM;
using namespace std;
class OpenCLSpatialNonbonded::AtomSortTrait : public ComputeSortImpl::SortTrait {
public:
    int getDataSize() const { return sizeof(mm_int2); }
    int getKeySize() const { return 8; }
    const char* getDataType() const { return "int2"; }
    const char* getKeyType() const { return "mm_ulong"; }
    const char* getMinKey() const { return "0ull"; }
    const char* getMaxKey() const { return "0xFFFFFFFFFFFFFFFFull"; }
    const char* getMaxValue() const { return "make_int2(-1, -1)"; }
    const char* getSortKey() const { return "(((mm_ulong)(unsigned int)value.x)<<32) | (unsigned int)value.y"; }
};

class OpenCLSpatialNonbonded::ExclusionSortTrait : public ComputeSortImpl::SortTrait {
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

OpenCLSpatialNonbonded::OpenCLSpatialNonbonded(OpenCLContext& context, OpenCLNonbondedUtilities& nonbonded) :
        cc(context), nb(nonbonded), forcesPending(false), reorderCount(0), parameterArrays(nullptr) {
    inverseMerge = true;
    reorderInterval = 250;
}
void OpenCLSpatialNonbonded::initialize(const System& system) {
    if (cc.getPlatformData().contexts.size() != 1)
        throw OpenMMException("Experimental AtomReordering supports one OpenCL device only");
    if (nb.deviceIsCpu || cc.getSIMDWidth() != 32)
        throw OpenMMException("Experimental OpenCL atom ordering currently requires a 32-lane GPU");
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

    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(n);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(padded);
    defines["NUM_BLOCKS"] = cc.intToString(blocks);
    defines["NUM_EXCLUSIONS"] = cc.intToString(numRecords);
    defines["COMPACT_EXCLUSIONS"] = "1";
    defines["FUSED_SPATIAL"] = "1";
    defines["ATOMIC_AND(dest, value)"] = "atom_and(dest, value)";

    if (inverseMerge) defines["INVERSE_MERGE"] = "1";
    defines["HILBERT_SPATIAL"] = "1";
    parameterArrays = &cc.getReorderedArraySet();
    for (int i = 0; i < nb.parameters.size(); i++) {
        ComputeParameterInfo& param = nb.parameters[i];
        if (!param.isConstant() || param.getArray().getSize() < n || param.getNumComponents() == 3)
            throw OpenMMException("Unsupported spatial per-particle parameter: "+param.getName());
        parameters.push_back(unique_ptr<OpenCLArray>(new OpenCLArray(cc, padded, param.getSize(), "spatial_"+param.getName())));
        cc.clearBuffer(*parameters.back());
        cc.addReorderedArray(param.getArray(), *parameters.back());
    }
    parameterArrays->seal();
    string source = CommonKernelSources::spatialNonbonded;
    source += "\nKERNEL void spatialParameters(GLOBAL const real4* original, GLOBAL real4* sorted, GLOBAL const int* order"+parameterArrays->getArguments("GLOBAL")+") {\n"
              "for (int s=GLOBAL_ID; s<NUM_ATOMS; s+=GLOBAL_SIZE) {\n"
              "sorted[s].w=original[order[s]].w;\n"+parameterArrays->getGatherSource("s", "order[s]")+"}}\n";
    ComputeProgram program = cc.compileProgram(source, defines);
    for (const string& name : {"spatialKeys", "spatialMaps", "spatialPositions", "spatialMerge", "spatialExclusionKeys",
            "spatialMarkTiles", "spatialScanBlocks", "spatialScanOffsets", "spatialBuildTiles", "spatialRows",
            "spatialInitMasks", "spatialMasks", "spatialAdjacency", "spatialParameters"})
        kernels[name] = program->createKernel(name);
    vector<ArrayInterface*> parametersArgs = {&cc.getPosq(), &positions, &order};
    for (const auto& entry : parameterArrays->getEntries()) {
        parametersArgs.push_back(entry.original);
        parametersArgs.push_back(entry.sorted);
    }
    bind("spatialParameters", parametersArgs);
}
ComputeKernel OpenCLSpatialNonbonded::bind(const string& name, vector<ArrayInterface*> arrays) {
    ComputeKernel kernel = kernels.at(name);
    if (boundArgs.count(name)) {
        for (int i = 0; i < arrays.size(); i++) kernel->setArg(i, *arrays[i]);
    }
    else {
        for (ArrayInterface* array : arrays) kernel->addArg(*array);
        boundArgs[name] = arrays.size();
    }
    return kernel;
}
void OpenCLSpatialNonbonded::execute(const string& name, vector<ArrayInterface*> arrays, int size) {
    bind(name, arrays)->execute(size, 256);
}
void OpenCLSpatialNonbonded::scan(OpenCLArray& values, int length, OpenCLArray& total) {
    bool first = !boundArgs.count("spatialScanBlocks");
    ComputeKernel kernel = bind("spatialScanBlocks", {&values, &scanSums});
    if (first) kernel->addArg(length); else kernel->setArg(2, length);
    kernel->execute(length, 256);
    int chunks = (length+255)/256;
    first = !boundArgs.count("spatialScanOffsets");
    kernel = bind("spatialScanOffsets", {&scanSums, &scanOffsets, &total});
    if (first) kernel->addArg(chunks); else kernel->setArg(3, chunks);
    kernel->execute(1, 256);
}
void OpenCLSpatialNonbonded::reorder() {
    bool first = !boundArgs.count("spatialKeys");
    ComputeKernel kernel = bind("spatialKeys", {&cc.getPosq(), &atomKeys});
    if (first) {
        kernel->addArg(); kernel->addArg(); kernel->addArg();

    }
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(2, cc.getPeriodicBoxVecXDouble()); kernel->setArg(3, cc.getPeriodicBoxVecYDouble()); kernel->setArg(4, cc.getPeriodicBoxVecZDouble());
    }
    else {
        kernel->setArg(2, cc.getPeriodicBoxVecX()); kernel->setArg(3, cc.getPeriodicBoxVecY()); kernel->setArg(4, cc.getPeriodicBoxVecZ());
    }
    kernel->execute(cc.getNumAtoms(), 256);
    atomSorter->sort(atomKeys);
    vector<ArrayInterface*> args = {&atomKeys, &order, &inverseOrder};
    execute("spatialMaps", args, cc.getPaddedNumAtoms());

    remapExclusions();
    cc.invalidateReorderedArrays();
    reorderCount++;
    cc.setStepsSinceReorder(0);
}
void OpenCLSpatialNonbonded::remapExclusions() {
    execute("spatialExclusionKeys", {&originalExclusions, &inverseOrder, &exclusionRecords}, numRecords);
    exclusionSorter->sort(exclusionRecords);
    execute("spatialMarkTiles", {&exclusionRecords, &tileIds}, numRecords);
    scan(tileIds, numRecords, exclusionCount);
    cc.clearBuffer(rowCounts);
    execute("spatialBuildTiles", {&exclusionRecords, &tileIds, &scanOffsets, &nb.exclusionTiles, &rowCounts}, numRecords);
    scan(rowCounts, cc.getNumAtomBlocks(), scanTotal);
    execute("spatialRows", {&rowCounts, &scanOffsets, &nb.exclusionRowIndices}, cc.getNumAtomBlocks());
    execute("spatialInitMasks", {&exclusionCount, &nb.exclusions}, numRecords);
    execute("spatialMasks", {&exclusionRecords, &tileIds, &nb.exclusions}, numRecords);
    cc.clearBuffer(rowCursors);
    execute("spatialAdjacency", {&nb.exclusionTiles, &exclusionCount, &nb.exclusionRowIndices, &rowCursors, &nb.exclusionIndices}, numRecords);
    nb.forceRebuildNeighborList = true;
}
bool OpenCLSpatialNonbonded::usesBoundsGather() const {
    return inverseMerge && nb.useNeighborList && nb.numTiles > 0;
}
void OpenCLSpatialNonbonded::prepare() {
    forcesPending = false;
    if (reorderCount == 0 || cc.getStepsSinceReorder() >= reorderInterval) reorder();
    if (usesBoundsGather()) return;
    // These bindings remain constant over the Context lifetime.
    if (!boundArgs.count("spatialPositions")) bind("spatialPositions", {&cc.getPosq(), &positions, &order, &forces});
    kernels.at("spatialPositions")->execute(cc.getPaddedNumAtoms(), 256);
}
void OpenCLSpatialNonbonded::invalidateParameters() {
    cc.invalidateReorderedArrays();
}

void OpenCLSpatialNonbonded::gatherParameters(bool includeForces) {
    if (includeForces) forcesPending = true;
    if (parameterArrays->isDirty()) {
        parameterArrays->validate();
        kernels.at("spatialParameters")->execute(cc.getNumAtoms(), 256);
        parameterArrays->markGathered();
    }
}
bool OpenCLSpatialNonbonded::mergeForces() {
    if (!forcesPending) return false;
    if (inverseMerge) {
        if (!mergeReductionKernel) {
            map<string, string> defines;
            defines["SPATIAL_FORCE_REDUCTION"] = "1";
            mergeReductionKernel = cc.compileProgram(OpenCLKernelSources::utilities, defines)->createKernel("reduceForces");
            mergeReductionKernel->addArg(cc.getLongForceBuffer());
            mergeReductionKernel->addArg(cc.getForceBuffers());
            mergeReductionKernel->addArg(cc.getPaddedNumAtoms());
            mergeReductionKernel->addArg(cc.getNumForceBuffers());
            mergeReductionKernel->addArg(forces);
            mergeReductionKernel->addArg(inverseOrder);
        }
        // The caller has joined PME post-computations. Reduce before virtual
        // sites and integrators consume original-order forces.
        mergeReductionKernel->execute(cc.getPaddedNumAtoms(), 128);
        forcesPending = false;
        return true;
    }
    if (!boundArgs.count("spatialMerge")) bind("spatialMerge", {&cc.getLongForceBuffer(), &forces, inverseMerge ? &inverseOrder : &order});
    kernels.at("spatialMerge")->execute(cc.getNumAtoms(), 256);
    forcesPending = false;
    return false;
}
