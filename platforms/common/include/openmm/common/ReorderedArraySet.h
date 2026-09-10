#ifndef OPENMM_REORDEREDARRAYSET_H_
#define OPENMM_REORDEREDARRAYSET_H_

// SPDX-License-Identifier: LGPL-3.0-or-later
#include "openmm/common/ComputeArray.h"
#include "openmm/OpenMMException.h"
#include <sstream>
#include <vector>

namespace OpenMM {

/**
 * Experimental registration and code generation for a fused per-atom gather.
 *
 * Sources contain values in stable atom-ID order.  Destinations contain values
 * in sorted order.  This helper copies the storage representation, not a kernel
 * language vector type: a 12-byte element stays 12 bytes even on OpenCL, where
 * sizeof(float3) is 16.  Consumers must independently use the correct layout.
 *
 * Register arrays, seal the set, then compile and bind a kernel using the emitted
 * arguments and body.  Arrays are borrowed and must outlive every bound kernel;
 * they must not be replaced or share underlying storage.  ComputeArray wrappers
 * are resolved at registration.  Call validate() before each gather to detect
 * insufficient capacity or changed element size after a resize.
 *
 * The caller owns the permutation, padding initialization, and
 * scheduling.  It must gather after parameter producers and before consumers,
 * whenever values or the permutation change.  No synchronization or device
 * transfers happen during registration. ComputeContext owns the production set;
 * backends seal it and fuse all registered copies into their parameter gather.
 */
class ReorderedArraySet {
public:
    struct Entry {
        ArrayInterface* original;
        ArrayInterface* sorted;
        int elementSize;
    };

    ReorderedArraySet(ComputeContext& context, int numAtoms, int paddedAtoms) :
            context(context), numAtoms(numAtoms), paddedAtoms(paddedAtoms), sealed(false), dirty(true) {
        if (numAtoms < 0 || paddedAtoms < numAtoms)
            throw OpenMMException("Invalid reordered array atom counts");
    }

    void addArray(ArrayInterface& originalArray, ArrayInterface& sortedArray) {
        if (sealed)
            throw OpenMMException("Cannot register an array after sealing the reordered array set");
        ArrayInterface& original = unwrap(originalArray);
        ArrayInterface& sorted = unwrap(sortedArray);
        Entry entry = {&original, &sorted, original.getElementSize()};
        validateEntry(entry);
        if (&original == &sorted)
            throw OpenMMException("A reordered array cannot alias its source");
        for (const Entry& other : entries) {
            // A source may feed multiple destinations, but no destination may
            // be read or written by another entry in the same parallel gather.
            if (&sorted == other.sorted || &sorted == other.original || &original == other.sorted)
                throw OpenMMException("Aliased arrays in a reordered array set");
        }
        entries.push_back(entry);
    }

    void seal() {
        validate();
        sealed = true;
    }

    void validate() const {
        for (const Entry& entry : entries)
            validateEntry(entry);
    }

    /**
     * Invalidate after a source update, allocation change, or new permutation.
     */
    void invalidate() { dirty = true; }

    bool isDirty() const { return dirty; }

    /**
     * Called only after successfully submitting the gather on the consumer queue.
     */
    void markGathered() {
        requireSealed();
        dirty = false;
    }

    const std::vector<Entry>& getEntries() const {
        requireSealed();
        return entries;
    }

    /**
     * Emit additional arguments, including their leading commas.
     */
    std::string getArguments(const std::string& addressSpace) const {
        requireSealed();
        std::stringstream source;
        for (int i = 0; i < entries.size(); i++) {
            std::string type = getCopyType(entries[i].elementSize);
            source << ", " << addressSpace << " const " << type << "* reorderSource" << i
                   << ", " << addressSpace << " " << type << "* reorderDestination" << i;
        }
        return source.str();
    }

    /**
     * Emit copies for a valid sorted index and its original atom ID.
     */
    std::string getGatherSource(const std::string& sortedIndex, const std::string& originalIndex) const {
        requireSealed();
        std::stringstream source;
        for (int i = 0; i < entries.size(); i++) {
            int units = entries[i].elementSize/getCopySize(entries[i].elementSize);
            source << "for (int component=0; component<" << units << "; component++)\n"
                   << "reorderDestination" << i << "[(size_t)(" << sortedIndex << ")*" << units << "+component] = "
                   << "reorderSource" << i << "[(size_t)(" << originalIndex << ")*" << units << "+component];\n";
        }
        return source.str();
    }

private:
    static ArrayInterface& unwrap(ArrayInterface& array) {
        if (!array.isInitialized())
            throw OpenMMException("Cannot register an uninitialized reordered array");
        ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
        return wrapper == NULL ? array : unwrap(wrapper->getArray());
    }

    void validateEntry(const Entry& entry) const {
        if (!entry.original->isInitialized() || !entry.sorted->isInitialized() ||
                &entry.original->getContext() != &context || &entry.sorted->getContext() != &context)
            throw OpenMMException("Reordered arrays must belong to the same ComputeContext");
        if (entry.elementSize <= 0 || entry.original->getElementSize() != entry.elementSize ||
                entry.sorted->getElementSize() != entry.elementSize)
            throw OpenMMException("Reordered array element sizes must match");
        if (entry.original->getSize() < numAtoms || entry.sorted->getSize() < paddedAtoms)
            throw OpenMMException("Insufficient capacity for a reordered array");
    }

    void requireSealed() const {
        if (!sealed)
            throw OpenMMException("Seal the reordered array set before generating a kernel");
    }

    // Device allocations are suitably aligned.  The chosen width divides the
    // element stride, so every vector access is aligned, including odd strides.
    static int getCopySize(int size) {
        return size%16 == 0 ? 16 : (size%8 == 0 ? 8 : (size%4 == 0 ? 4 : 1));
    }
    static std::string getCopyType(int size) {
        int width = getCopySize(size);
        return width == 16 ? "uint4" : (width == 8 ? "uint2" : (width == 4 ? "unsigned int" : "unsigned char"));
    }

    ComputeContext& context;
    int numAtoms, paddedAtoms;
    bool sealed, dirty;
    std::vector<Entry> entries;
};

} // namespace OpenMM
#endif
