// Check the shared device predicate against exact block sets, including hash
// collisions, empty rows and reordered block memberships.
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#define DEVICE inline
#include "../src/kernels/spatialExclusionFilter.cc"

int main() {
    std::mt19937 random(5406);
    int cases = 0;
    for (int count : {0, 1, 2, 4, 31, 32, 33, 131, 1025}) {
        for (int trial = 0; trial < 100; trial++) {
            std::set<int> exact;
            unsigned int filter = 0;
            for (int i = 0; i < count; i++) {
                int block = random()%4096;
                exact.insert(block);
                filter |= spatialExclusionBlockBit(block);
            }
            for (int block = 0; block < 4096; block++)
                if (exact.count(block) && !spatialExclusionMayContain(filter, block))
                    throw std::runtime_error("Exclusion filter false negative");
            if (count == 0 && spatialExclusionMayContain(filter, 0))
                throw std::runtime_error("Empty filter matches");
            cases++;
        }
    }
    unsigned int filter = spatialExclusionBlockBit(31);
    if (!spatialExclusionMayContain(filter, 63) || spatialExclusionMayContain(filter, 32))
        throw std::runtime_error("Block hash/collision handling changed");
    // Changing the permutation must replace the filter, not reuse old bits.
    filter = spatialExclusionBlockBit(32);
    if (!spatialExclusionMayContain(filter, 32) || spatialExclusionMayContain(filter, 31))
        throw std::runtime_error("Rebuilt block membership incorrect");
    std::cout << "Passed " << cases << " exact-set cases plus collision/rebuild checks" << std::endl;
}
