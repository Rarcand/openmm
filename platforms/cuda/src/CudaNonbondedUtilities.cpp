/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenMMException.h"
#include "CudaNonbondedUtilities.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaKernelSources.h"
#include "CommonKernelSources.h"
#include "CudaExpressionUtilities.h"
#include "CudaSpatialNonbonded.h"
#include "openmm/common/ContextSelector.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<errorMessage<<": "<<context.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }


class CudaNonbondedUtilities::BlockSortTrait : public ComputeSortImpl::SortTrait {
public:
    BlockSortTrait() {}
    int getDataSize() const {return sizeof(int);}
    int getKeySize() const {return sizeof(int);}
    const char* getDataType() const {return "unsigned int";}
    const char* getKeyType() const {return "unsigned int";}
    const char* getMinKey() const {return "0";}
    const char* getMaxKey() const {return "0xFFFFFFFFu";}
    const char* getMaxValue() const {return "0xFFFFFFFFu";}
    const char* getSortKey() const {return "value";}
};

CudaNonbondedUtilities::CudaNonbondedUtilities(CudaContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
        pinnedCountBuffer(NULL), forceRebuildNeighborList(true), groupFlags(0), canUsePairList(true), canOmitExcludedPairs(true), tilesAfterReorder(0) {
    // Decide how many thread blocks to use.

    string errorMessage = "Error initializing nonbonded utilities";
    int multiprocessors;
    CHECK_RESULT(cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, context.getDevice()));
    CHECK_RESULT(cuEventCreate(&downloadCountEvent, context.getEventFlags()));
    CHECK_RESULT(cuMemHostAlloc((void**) &pinnedCountBuffer, 2*sizeof(unsigned int), CU_MEMHOSTALLOC_PORTABLE));
    numForceThreadBlocks = 4*multiprocessors;
    forceThreadBlockSize = (context.getComputeCapability() < 2.0 ? 128 : 256);

    // When building the neighbor list, we can optionally use large blocks (1024 atoms) to
    // accelerate the process.  This makes building the neighbor list faster, but it prevents
    // us from sorting atom blocks by size, which leads to a slightly less efficient neighbor
    // list.  We guess based on system size which will be faster.

    useLargeBlocks = (context.getNumAtoms() > 90000);
    setKernelSource(CudaKernelSources::nonbonded);
    string mode = context.getPlatformData().propertyValues[CudaPlatform::CudaAtomReordering()];
    phaseTiming = context.getPlatformData().propertyValues.at("AtomReorderingPhaseTiming") == "true";
    diagnostics = context.getPlatformData().propertyValues.at("AtomReorderingDiagnostics") == "true";
    if (mode != "baseline")
        spatial.reset(new CudaSpatialNonbonded(context, *this, mode == "inverse"));
}

CudaNonbondedUtilities::~CudaNonbondedUtilities() {
    for (auto& sample : phaseEvents) {
        if (sample.start) cuEventDestroy(sample.start);
        if (sample.end) cuEventDestroy(sample.end);
    }
    if (pinnedCountBuffer != NULL)
        cuMemFreeHost(pinnedCountBuffer);
    cuEventDestroy(downloadCountEvent);
}

void CudaNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance,
        const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList, bool supportsPairList, bool supportsExclusionOmission) {
    if (groupCutoff.size() > 0) {
        if (usesCutoff != useCutoff)
            throw OpenMMException("All Forces must agree on whether to use a cutoff");
        if (usesPeriodic != usePeriodic)
            throw OpenMMException("All Forces must agree on whether to use periodic boundary conditions");
        if (usesCutoff && groupCutoff.find(forceGroup) != groupCutoff.end() && groupCutoff[forceGroup] != cutoffDistance)
            throw OpenMMException("All Forces in a single force group must use the same cutoff distance");
    }
    if (usesExclusions)
        requestExclusions(exclusionList);
    useCutoff = usesCutoff;
    usePeriodic = usesPeriodic;
    this->useNeighborList |= (useNeighborList && useCutoff);
    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    canUsePairList &= supportsPairList;
    canOmitExcludedPairs &= supportsExclusionOmission;
    if (kernel.size() > 0) {
        if (groupKernelSource.find(forceGroup) == groupKernelSource.end())
            groupKernelSource[forceGroup] = "";
        map<string, string> replacements;
        replacements["CUTOFF"] = "CUTOFF_"+context.intToString(forceGroup);
        replacements["CUTOFF_SQUARED"] = "CUTOFF_"+context.intToString(forceGroup)+"_SQUARED";
        groupKernelSource[forceGroup] += context.replaceStrings(kernel, replacements)+"\n";
    }
}

void CudaNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
    parameters.push_back(parameter);
}

void CudaNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
    arguments.push_back(parameter);
}

string CudaNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
    // See if the parameter has already been added.

    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void CudaNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
    if (anyExclusions) {
        bool sameExclusions = (exclusionList.size() == atomExclusions.size());
        for (int i = 0; i < (int) exclusionList.size() && sameExclusions; i++) {
             if (exclusionList[i].size() != atomExclusions[i].size())
                 sameExclusions = false;
            set<int> expectedExclusions;
            expectedExclusions.insert(atomExclusions[i].begin(), atomExclusions[i].end());
            for (int j = 0; j < (int) exclusionList[i].size(); j++)
                if (expectedExclusions.find(exclusionList[i][j]) == expectedExclusions.end())
                     sameExclusions = false;
        }
        if (!sameExclusions)
            throw OpenMMException("All Forces must have identical exceptions");
    }
    else {
        atomExclusions = exclusionList;
        anyExclusions = true;
    }
}

static bool compareInt2(int2 a, int2 b) {
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void CudaNonbondedUtilities::initialize(const System& system) {
    string errorMessage = "Error initializing nonbonded utilities";
    if (atomExclusions.size() == 0) {
        // No exclusions were specifically requested, so just mark every atom as not interacting with itself.

        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    // Create the list of tiles.

    numAtoms = context.getNumAtoms();
    int numAtomBlocks = context.getNumAtomBlocks();
    int numContexts = context.getPlatformData().contexts.size();
    setAtomBlockRange(context.getContextIndex()/(double) numContexts, (context.getContextIndex()+1)/(double) numContexts);

    // Build a list of tiles that contain exclusions.

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    vector<int2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(make_int2(iter->first, iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), compareInt2);
    exclusionTiles.initialize<int2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    diagnosticExclusionTiles = exclusionTilesVec.size();
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        int2 tile = exclusionTilesVec[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    vector<vector<int> > exclusionBlocksForBlock(numAtomBlocks);
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter) {
        exclusionBlocksForBlock[iter->first].push_back(iter->second);
        if (iter->first != iter->second)
            exclusionBlocksForBlock[iter->second].push_back(iter->first);
    }
    vector<unsigned int> exclusionRowIndicesVec(numAtomBlocks+1, 0);
    vector<unsigned int> exclusionIndicesVec;
    for (int i = 0; i < numAtomBlocks; i++) {
        exclusionIndicesVec.insert(exclusionIndicesVec.end(), exclusionBlocksForBlock[i].begin(), exclusionBlocksForBlock[i].end());
        exclusionRowIndicesVec[i+1] = exclusionIndicesVec.size();
    }
    maxExclusions = 0;
    for (int i = 0; i < (int) exclusionBlocksForBlock.size(); i++)
        maxExclusions = (maxExclusions > exclusionBlocksForBlock[i].size() ? maxExclusions : exclusionBlocksForBlock[i].size());
    exclusionIndices.initialize<unsigned int>(context, exclusionIndicesVec.size(), "exclusionIndices");
    exclusionRowIndices.initialize<unsigned int>(context, exclusionRowIndicesVec.size(), "exclusionRowIndices");
    exclusionIndices.upload(exclusionIndicesVec);
    exclusionRowIndices.upload(exclusionRowIndicesVec);

    // Record the exclusion data.

    exclusions.initialize<tileflags>(context, tilesWithExclusions.size()*CudaContext::TileSize, "exclusions");
    tileflags allFlags = (tileflags) -1;
    vector<tileflags> exclusionVec(exclusions.getSize(), allFlags);
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/CudaContext::TileSize;
            int offset2 = atom2-y*CudaContext::TileSize;
            if (x > y) {
                int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
                exclusionVec[index+offset1] &= allFlags-(1<<offset2);
            }
            else {
                int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
                exclusionVec[index+offset2] &= allFlags-(1<<offset1);
            }
        }
    }
    if (!spatial)
        atomExclusions.clear(); // Spatial views retain original-ID exclusions.
    exclusions.upload(exclusionVec);

    // Create data structures for the neighbor list.

    maxCutoff = getMaxCutoffDistance();
    if (useCutoff) {
        // Select a size for the arrays that hold the neighbor list.  We have to make a fairly
        // arbitrary guess, but if this turns out to be too small we'll increase it later.

        maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        maxSinglePairs = 5*numAtoms;
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, CudaContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 2, "interactionCount");
        singlePairs.initialize<int2>(context, maxSinglePairs, "singlePairs");
        int elementSize = (context.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize<unsigned int>(context, numAtomBlocks, "sortedBlocks");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        int boundsTileLanes = (spatial && spatial->gathersInBounds() ? spatial->getBoundsTileLanes() : 1);
        int tilesPerBoundsBlock = 64/boundsTileLanes;
        numBlockSizes = min((context.getNumAtomBlocks()+tilesPerBoundsBlock-1)/tilesPerBoundsBlock, context.getNumThreadBlocks());
        blockSizeRange.initialize(context, numBlockSizes, 2*elementSize, "blockSizeRange");
        largeBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockCenter");
        largeBlockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 1, "rebuildNeighborList");
        blockSorter = context.createSort(new BlockSortTrait(), numAtomBlocks, false);
        vector<unsigned int> count(2, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(&count[0]);
    }

    // Record arguments for kernels.

    forceArgs.push_back(&context.getForce().getDevicePointer());
    forceArgs.push_back(&context.getEnergyBuffer().getDevicePointer());
    forceArgs.push_back(&context.getPosq().getDevicePointer());
    forceArgs.push_back(&exclusions.getDevicePointer());
    forceArgs.push_back(&exclusionTiles.getDevicePointer());
    forceArgs.push_back(&startTileIndex);
    forceArgs.push_back(&numTiles);
    if (useCutoff) {
        forceArgs.push_back(&interactingTiles.getDevicePointer());
        forceArgs.push_back(&interactionCount.getDevicePointer());
        forceArgs.push_back(context.getPeriodicBoxSizePointer());
        forceArgs.push_back(context.getInvPeriodicBoxSizePointer());
        forceArgs.push_back(context.getPeriodicBoxVecXPointer());
        forceArgs.push_back(context.getPeriodicBoxVecYPointer());
        forceArgs.push_back(context.getPeriodicBoxVecZPointer());
        forceArgs.push_back(&maxTiles);
        forceArgs.push_back(&blockCenter.getDevicePointer());
        forceArgs.push_back(&blockBoundingBox.getDevicePointer());
        forceArgs.push_back(&interactingAtoms.getDevicePointer());
        forceArgs.push_back(&maxSinglePairs);
        forceArgs.push_back(&singlePairs.getDevicePointer());
    }
    hasInitializedParams = false;
    paramStartIndex = forceArgs.size();
    for (int i = 0; i < parameters.size()+arguments.size(); i++)
        forceArgs.push_back(NULL);
    if (energyParameterDerivatives.size() > 0)
        forceArgs.push_back(&context.getEnergyParamDerivBuffer().getDevicePointer());
    if (useCutoff) {
        findBlockBoundsArgs.push_back(&numAtoms);
        findBlockBoundsArgs.push_back(context.getPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecXPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecYPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecZPointer());
        findBlockBoundsArgs.push_back(&context.getPosq().getDevicePointer());
        findBlockBoundsArgs.push_back(&blockCenter.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockBoundingBox.getDevicePointer());
        findBlockBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockSizeRange.getDevicePointer());
        computeSortKeysArgs.push_back(&blockBoundingBox.getDevicePointer());
        computeSortKeysArgs.push_back(&sortedBlocks.getDevicePointer());
        computeSortKeysArgs.push_back(&blockSizeRange.getDevicePointer());
        computeSortKeysArgs.push_back(&numBlockSizes);
        sortBoxDataArgs.push_back(&sortedBlocks.getDevicePointer());
        sortBoxDataArgs.push_back(&blockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&blockBoundingBox.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            sortBoxDataArgs.push_back(&largeBlockCenter.getDevicePointer());
            sortBoxDataArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getInvPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecXPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecYPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecZPointer());
        }
        sortBoxDataArgs.push_back(&context.getPosq().getDevicePointer());
        sortBoxDataArgs.push_back(&oldPositions.getDevicePointer());
        sortBoxDataArgs.push_back(&interactionCount.getDevicePointer());
        sortBoxDataArgs.push_back(&rebuildNeighborList.getDevicePointer());
        sortBoxDataArgs.push_back(&forceRebuildNeighborList);
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecXPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecYPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecZPointer());
        findInteractingBlocksArgs.push_back(&interactionCount.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingTiles.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingAtoms.getDevicePointer());
        findInteractingBlocksArgs.push_back(&singlePairs.getDevicePointer());
        findInteractingBlocksArgs.push_back(&context.getPosq().getDevicePointer());
        findInteractingBlocksArgs.push_back(&maxTiles);
        findInteractingBlocksArgs.push_back(&maxSinglePairs);
        findInteractingBlocksArgs.push_back(&startBlockIndex);
        findInteractingBlocksArgs.push_back(&numBlocks);
        findInteractingBlocksArgs.push_back(&sortedBlocks.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockCenter.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            findInteractingBlocksArgs.push_back(&largeBlockCenter.getDevicePointer());
            findInteractingBlocksArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
        }
        findInteractingBlocksArgs.push_back(&exclusionIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&exclusionRowIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&oldPositions.getDevicePointer());
        findInteractingBlocksArgs.push_back(&rebuildNeighborList.getDevicePointer());
    }
    if (spatial)
        spatial->initialize(system);
}

double CudaNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double CudaNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? 0.08*cutoff : 0.0);
    return cutoff+padding;
}

void CudaNonbondedUtilities::prepareInteractions(int forceGroups) {
    spatialViewReady = false;
    if ((forceGroups&groupFlags) == 0)
        return;
    if (spatial) {
        spatial->prepare();
        spatialViewReady = true;
    }
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    KernelSet& kernels = groupKernels[forceGroups];
    if (useCutoff && usePeriodic) {
        double4 box = context.getPeriodicBoxSize();
        double minAllowedSize = 1.999999*maxCutoff;
        if (box.x < minAllowedSize || box.y < minAllowedSize || box.z < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    if (!useNeighborList)
        return;
    if (numTiles == 0)
        return;

    // Compute the neighbor list.

    beginPhase("neighbor_list");
    // The allocation above and launch must describe the same number of blocks.
    context.executeKernel(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], numBlockSizes*64);
    context.executeKernel(kernels.computeSortKeysKernel, &computeSortKeysArgs[0], context.getNumAtomBlocks());
    blockSorter->sort(sortedBlocks);
    context.executeKernel(kernels.sortBoxDataKernel, &sortBoxDataArgs[0], context.getNumAtoms());
    context.executeKernel(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtoms(), 256);
    endPhase();
    forceRebuildNeighborList = false;
    interactionCount.download(pinnedCountBuffer, false);
    cuEventRecord(downloadCountEvent, context.getCurrentStream());
}

void CudaNonbondedUtilities::initParamArgs() {
    int index = paramStartIndex;
    for (int i = 0; i < parameters.size(); i++)
        forceArgs[index++] = &(spatial ? spatial->parameters[i]->getDevicePointer() : context.unwrap(parameters[i].getArray()).getDevicePointer());
    for (ComputeParameterInfo& arg : arguments)
        forceArgs[index++] = &context.unwrap(arg.getArray()).getDevicePointer();
    hasInitializedParams = true;
}

void CudaNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces && (includeForces || includeEnergy)) {
        CUfunction& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel == NULL)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        if (!hasInitializedParams)
            initParamArgs();
        if (spatial) {
            getSpatialWorkView(includeForces);
        }
        beginPhase("nonbonded_direct");
        context.executeKernel(kernel, &forceArgs[0], numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
        endPhase();

    }
    if (useNeighborList && numTiles > 0) {
        cuEventSynchronize(downloadCountEvent);
        updateNeighborListSize();
    }
}

bool CudaNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;
    if (diagnostics) {
        diagnosticEvaluations++;
        diagnosticTiles += pinnedCountBuffer[0];
        diagnosticPairs += pinnedCountBuffer[1];
        diagnosticExclusions += diagnosticExclusionTiles;
    }
    if (!spatial) {
        if (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0)
            tilesAfterReorder = pinnedCountBuffer[0];
        else if (context.getStepsSinceReorder() > 25 && pinnedCountBuffer[0] > 1.1*tilesAfterReorder) {
            if (spatial)
                spatial->requestReorder();
            else
                context.forceReorder();
        }
    }
    if (pinnedCountBuffer[0] <= maxTiles && pinnedCountBuffer[1] <= maxSinglePairs)
        return false;

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.

    if (pinnedCountBuffer[0] > maxTiles) {
        maxTiles = (unsigned int) (1.2*pinnedCountBuffer[0]);
        unsigned int numBlocks = context.getNumAtomBlocks();
        int totalTiles = numBlocks*(numBlocks+1)/2;
        if (maxTiles > totalTiles)
            maxTiles = totalTiles;
        interactingTiles.resize(maxTiles);
        interactingAtoms.resize(CudaContext::TileSize*(size_t) maxTiles);
        if (forceArgs.size() > 0)
            forceArgs[7] = &interactingTiles.getDevicePointer();
        findInteractingBlocksArgs[6] = &interactingTiles.getDevicePointer();
        if (forceArgs.size() > 0)
            forceArgs[17] = &interactingAtoms.getDevicePointer();
        findInteractingBlocksArgs[7] = &interactingAtoms.getDevicePointer();
    }
    if (pinnedCountBuffer[1] > maxSinglePairs) {
        maxSinglePairs = (unsigned int) (1.2*pinnedCountBuffer[1]);
        singlePairs.resize(maxSinglePairs);
        if (forceArgs.size() > 0)
            forceArgs[19] = &singlePairs.getDevicePointer();
        findInteractingBlocksArgs[8] = &singlePairs.getDevicePointer();
    }
    forceRebuildNeighborList = true;
    context.setForcesValid(false);
    return true;
}

const string& CudaNonbondedUtilities::getReorderingStatistics() {
    ContextSelector selector(context);
    collectPhaseTimes();
    stringstream out;
    out << "{\"enabled\":" << (diagnostics ? "true" : "false")
        << ",\"evaluations\":" << diagnosticEvaluations
        << ",\"candidate_tiles_total\":" << diagnosticTiles
        << ",\"single_pairs_total\":" << diagnosticPairs
        << ",\"exclusion_tiles_total\":" << diagnosticExclusions
        << ",\"last_exclusion_tiles\":" << diagnosticExclusionTiles
        << ",\"spatial_reorders\":" << (spatial ? spatial->getReorderCount() : 0)
        << ",\"stable_recenters\":" << context.getStableRecenterCount()
        << ",\"phase_timing\":" << (phaseTiming ? "true" : "false") << ",\"phases\":{";
    bool first = true;
    for (const auto& phase : phaseTotals) {
        if (!first) out << ",";
        first = false;
        out << "\"" << phase.first << "\":{\"milliseconds\":" << phase.second.first
            << ",\"calls\":" << phase.second.second << "}";
    }
    out << "}}";
    diagnosticReport = out.str();
    return diagnosticReport;
}

void CudaNonbondedUtilities::setUsePadding(bool padding) {
    usePadding = padding;
}

void CudaNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
    int numAtomBlocks = context.getNumAtomBlocks();
    startBlockIndex = (int) (startFraction*numAtomBlocks);
    numBlocks = (int) (endFraction*numAtomBlocks)-startBlockIndex;
    long long totalTiles = context.getNumAtomBlocks()*((long long)context.getNumAtomBlocks()+1)/2;
    startTileIndex = (int) (startFraction*totalTiles);
    numTiles = (long long) (endFraction*totalTiles)-startTileIndex;
    forceRebuildNeighborList = true;
}

void CudaNonbondedUtilities::createKernelsForGroups(int groups) {
    KernelSet kernels;
    string source;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            source += groupKernelSource[i];
        }
    }
    kernels.hasForces = (source.size() > 0);
    kernels.source = source;
    kernels.forceKernel = kernels.energyKernel = kernels.forceEnergyKernel = NULL;
    if (useCutoff) {
        double paddedCutoff = padCutoff(maxCutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-maxCutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        if (useLargeBlocks)
            defines["USE_LARGE_BLOCKS"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        if (spatial) defines["SPATIAL_ATOM_ORDER"] = "1";
        if (spatial && spatial->gathersInBounds()) {
            defines["GATHER_SPATIAL_BOUNDS"] = "1";
            defines["BOUNDS_TILE_LANES"] = context.intToString(spatial->getBoundsTileLanes());
            defines["CLEAR_SPATIAL_FORCES"] = "1";
        }
        defines["MAX_BITS_FOR_PAIRS"] = (canUsePairList ? (context.getComputeCapability() < 8.0 ? "2" : "3") : "0");
        int binShift = 1;
        while (1<<binShift <= context.getNumAtomBlocks())
            binShift++;
        defines["BIN_SHIFT"] = context.intToString(binShift);
        defines["BLOCK_INDEX_MASK"] = context.intToString((1<<binShift)-1);
        string neighborSource = CudaKernelSources::vectorOps+CudaKernelSources::findInteractingBlocks;
        CUmodule interactingBlocksProgram = context.createModule(neighborSource, defines);
        kernels.findBlockBoundsKernel = context.getKernel(interactingBlocksProgram, "findBlockBounds");
        kernels.computeSortKeysKernel = context.getKernel(interactingBlocksProgram, "computeSortKeys");
        kernels.sortBoxDataKernel = context.getKernel(interactingBlocksProgram, "sortBoxData");
        kernels.findInteractingBlocksKernel = context.getKernel(interactingBlocksProgram, "findBlocksWithInteractions");
    }
    groupKernels[groups] = kernels;
}

CUfunction CudaNonbondedUtilities::createInteractionKernel(const string& source, vector<ComputeParameterInfo>& params, vector<ComputeParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
    map<string, string> replacements;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            localData<<param.getType()<<" "<<param.getName()<<";\n";
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                localData<<param.getComponentType()<<" "<<param.getName()<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += param.getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (const ComputeParameterInfo& param : params) {
        args << ", ";
        if (param.isConstant())
            args << "const ";
        args << param.getType();
        args << "* __restrict__ global_";
        args << param.getName();
    }
    for (const ComputeParameterInfo& arg : arguments) {
        args << ", ";
        if (arg.isConstant())
            args << "const ";
        args << arg.getType();
        args << "* __restrict__ ";
        args << arg.getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", mixed* __restrict__ energyParamDerivs";
    if (spatial)
        args << ", const int* spatialExclusionCount";
    replacements["PARAMETER_ARGUMENTS"] = args.str();

    stringstream load1;
    for (const ComputeParameterInfo& param : params) {
        load1 << param.getType();
        load1 << " ";
        load1 << param.getName();
        load1 << "1 = global_";
        load1 << param.getName();
        load1 << "[atom1];\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();

    // Part 1. Defines for on diagonal exclusion tiles

    stringstream broadcastWarpData;
    broadcastWarpData << "posq2.x = real_shfl(shflPosq.x, j);\n";
    broadcastWarpData << "posq2.y = real_shfl(shflPosq.y, j);\n";
    broadcastWarpData << "posq2.z = real_shfl(shflPosq.z, j);\n";
    broadcastWarpData << "posq2.w = real_shfl(shflPosq.w, j);\n";
    for (const ComputeParameterInfo& param : params) {
        broadcastWarpData << param.getType() << " shfl" << param.getName() << ";\n";
        for (int j = 0; j < param.getNumComponents(); j++) {
            if (param.getNumComponents() == 1)
                broadcastWarpData << "shfl" << param.getName() << "=real_shfl(" << param.getName() <<"1,j);\n";
            else
                broadcastWarpData << "shfl" << param.getName()+"."+suffixes[j] << "=real_shfl(" << param.getName()+"1."+suffixes[j] <<",j);\n";
        }
    }
    replacements["BROADCAST_WARP_DATA"] = broadcastWarpData.str();

    // Part 2. Defines for off-diagonal exclusions, and neighborlist tiles.
    stringstream declareLocal2;
    for (const ComputeParameterInfo& param : params)
        declareLocal2<<param.getType()<<" shfl"<<param.getName()<<";\n";
    replacements["DECLARE_LOCAL_PARAMETERS"] = declareLocal2.str();

    stringstream loadLocal2;
    for (const ComputeParameterInfo& param : params)
        loadLocal2<<"shfl"<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();

    stringstream load2j;
    for (const ComputeParameterInfo& param : params)
        load2j<<param.getType()<<" "<<param.getName()<<"2 = shfl"<<param.getName()<<";\n";
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();

    stringstream load2g;
    for (const ComputeParameterInfo& param : params)
        load2g<<param.getType()<<" "<<param.getName()<<"2 = global_"<<param.getName()<<"[atom2];\n";
    replacements["LOAD_ATOM2_PARAMETERS_FROM_GLOBAL"] = load2g.str();

    stringstream clearLocal;
    for (const ComputeParameterInfo& param : params) {
        clearLocal<<"shfl"<<param.getName()<<" = ";
        if (param.getNumComponents() == 1)
            clearLocal<<"0;\n";
        else
            clearLocal<<"make_"<<param.getType()<<"(0);\n";
    }
    replacements["CLEAR_LOCAL_PARAMETERS"] = clearLocal.str();

    stringstream initDerivs;
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        initDerivs<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    stringstream saveDerivs;
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                saveDerivs<<"energyParamDerivs[GLOBAL_ID*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();

    stringstream shuffleWarpData;
    shuffleWarpData << "shflPosq.x = real_shfl(shflPosq.x, tgx+1);\n";
    shuffleWarpData << "shflPosq.y = real_shfl(shflPosq.y, tgx+1);\n";
    shuffleWarpData << "shflPosq.z = real_shfl(shflPosq.z, tgx+1);\n";
    shuffleWarpData << "shflPosq.w = real_shfl(shflPosq.w, tgx+1);\n";
    shuffleWarpData << "shflForce.x = real_shfl(shflForce.x, tgx+1);\n";
    shuffleWarpData << "shflForce.y = real_shfl(shflForce.y, tgx+1);\n";
    shuffleWarpData << "shflForce.z = real_shfl(shflForce.z, tgx+1);\n";
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            shuffleWarpData<<"shfl"<<param.getName()<<"=real_shfl(shfl"<<param.getName()<<", tgx+1);\n";
        else {
            for (int j = 0; j < param.getNumComponents(); j++) {
                // looks something like shflsigmaEpsilon.x = real_shfl(shflsigmaEpsilon.x,tgx+1);
                shuffleWarpData<<"shfl"<<param.getName()
                    <<"."<<suffixes[j]<<"=real_shfl(shfl"
                    <<param.getName()<<"."<<suffixes[j]
                    <<", tgx+1);\n";
            }
        }
    }
    replacements["SHUFFLE_WARP_DATA"] = shuffleWarpData.str();
    map<string, string> subtileShuffle;
    subtileShuffle["tgx+1"] = "((tgx & ~7u) | ((tgx+1) & 7u))";
    replacements["SHUFFLE_EXCLUSION_SUBTILE_DATA"] = context.replaceStrings(shuffleWarpData.str(), subtileShuffle);

    map<string, string> defines;
    if (useCutoff)
        defines["USE_CUTOFF"] = "1";
    if (usePeriodic)
        defines["USE_PERIODIC"] = "1";
    if (useExclusions)
        defines["USE_EXCLUSIONS"] = "1";
    // Arbitrary spatial reordering can create off-diagonal tiles later.
    addArithmeticGuardSources(source, kernelSource,
            context.getPlatformData().propertyValues.at("NonbondedArithmeticGuard"), replacements, defines);

    if (isSymmetric)
        defines["USE_SYMMETRIC"] = "1";
    if (useNeighborList)
        defines["USE_NEIGHBOR_LIST"] = "1";
    // Reuse the ordinary tile's conservative periodic-copy criterion for
    // exclusion tiles. It needs the current block bounds from a neighbor list.
    if (spatial && usePeriodic && useNeighborList)
        defines["EXCLUSION_LOCALITY"] = "1";

    defines["ENABLE_SHUFFLE"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
    double maxCutoff = 0.0;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            double cutoff = groupCutoff[i];
            maxCutoff = max(maxCutoff, cutoff);
            defines["CUTOFF_"+context.intToString(i)+"_SQUARED"] = context.doubleToString(cutoff*cutoff);
            defines["CUTOFF_"+context.intToString(i)] = context.doubleToString(cutoff);
        }
    }
    defines["MAX_CUTOFF"] = context.doubleToString(maxCutoff);
    defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
    defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
    int numExclusionTiles = exclusionTiles.getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(numExclusionTiles);
    int numContexts = context.getPlatformData().contexts.size();
    int startExclusionIndex = context.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (context.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = context.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = context.intToString(endExclusionIndex);
    if (spatial) {
        defines["NUM_TILES_WITH_EXCLUSIONS"] = "spatialExclusionCount[0]";
        defines["FIRST_EXCLUSION_TILE"] = "0";
        defines["LAST_EXCLUSION_TILE"] = "spatialExclusionCount[0]";
    }
    if ((localDataSize/4)%2 == 0 && !context.getUseDoublePrecision())
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    string kernelTemplate = kernelSource;

    CUmodule program = context.createModule(CudaKernelSources::vectorOps+context.replaceStrings(kernelTemplate, replacements), defines);
    CUfunction kernel = context.getKernel(program, "computeNonbonded");
    return kernel;
}

void CudaNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}

void CudaNonbondedUtilities::finishSpatialForces() {
    spatialViewReady = false;
    if (spatial)
        spatial->mergeForces();
}

ArrayInterface* CudaNonbondedUtilities::getInverseSpatialAtomOrder() {
    return spatial && spatial->getReorderCount() > 0 ? &spatial->getInverseAtomOrder() : nullptr;
}

ArrayInterface* CudaNonbondedUtilities::getReorderedParameterArray(ArrayInterface& original) {
    if (!spatial || spatial->getReorderCount() == 0)
        return nullptr;
    if (!original.isInitialized() || &original.getContext() != &context)
        throw OpenMMException("Spatial parameter source must belong to this ComputeContext");
    for (const auto& entry : context.getReorderedArraySet().getEntries())
        if (&context.unwrap(original) == entry.original)
            return entry.sorted;
    return nullptr;
}

void CudaNonbondedUtilities::invalidateSpatialParameters() {
    if (spatial)
        spatial->invalidateParameters();
}

// Diagnostic stream elapsed times. Reuse a bounded event pool and read it only
// at explicit statistics requests or when full. Normal timing runs disable this.
static void checkPhaseEvent(CUresult result) {
    if (result != CUDA_SUCCESS)
        throw OpenMMException("CUDA phase timer: "+CudaContext::getErrorString(result));
}

void CudaNonbondedUtilities::beginPhase(const char* name) {
    if (!phaseTiming) return;
    if (phaseEvents.empty()) {
        phaseEvents.resize(8192);
        for (auto& sample : phaseEvents) {
            checkPhaseEvent(cuEventCreate(&sample.start, CU_EVENT_DEFAULT));
            checkPhaseEvent(cuEventCreate(&sample.end, CU_EVENT_DEFAULT));
        }
    }
    if (pendingPhaseEvents == phaseEvents.size()) collectPhaseTimes();
    auto& sample = phaseEvents[pendingPhaseEvents];
    sample.name = name;
    checkPhaseEvent(cuEventRecord(sample.start, context.getCurrentStream()));
}

void CudaNonbondedUtilities::endPhase() {
    if (!phaseTiming) return;
    checkPhaseEvent(cuEventRecord(phaseEvents[pendingPhaseEvents].end, context.getCurrentStream()));
    pendingPhaseEvents++;
}

void CudaNonbondedUtilities::collectPhaseTimes() {
    if (!pendingPhaseEvents) return;
    checkPhaseEvent(cuEventSynchronize(phaseEvents[pendingPhaseEvents-1].end));
    for (int i = 0; i < pendingPhaseEvents; i++) {
        auto& sample = phaseEvents[i];
        float milliseconds;
        checkPhaseEvent(cuEventElapsedTime(&milliseconds, sample.start, sample.end));
        phaseTotals[sample.name].first += milliseconds;
        phaseTotals[sample.name].second++;
    }
    pendingPhaseEvents = 0;
}

SpatialNonbondedView CudaNonbondedUtilities::getSpatialWorkView(bool includeForces) {
    if (!spatial || !spatialViewReady)
        throw OpenMMException("Spatial work is only available during an active nonbonded evaluation");
    spatial->gatherParameters(includeForces);
    return {&spatial->positions, &spatial->forces,
        &spatial->getAtomOrder(), &spatial->getInverseAtomOrder(),
        &exclusionTiles, &exclusions, &spatial->exclusionCount,
        &exclusionRowIndices, &exclusionIndices,
        useNeighborList ? &interactingTiles : nullptr,
        useNeighborList ? &interactingAtoms : nullptr,
        useNeighborList ? &interactionCount : nullptr, useNeighborList ? &singlePairs : nullptr,
        context.getNumAtoms(), context.getPaddedNumAtoms(), spatial->getReorderCount(), useNeighborList};
}
