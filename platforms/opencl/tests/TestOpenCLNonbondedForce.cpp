/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "TestNonbondedForce.h"
#include "OpenCLIncludes.h"

bool canRunHugeTest() {
    // Create a minimal context just to see which platform and device are being used.
    
    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(1.0);
    Context context(system, integrator, platform);
    int platformIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLPlatformIndex()));
    int deviceIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLDeviceIndex()));

    // Find out how much memory the device has.

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    long long memory = devices[deviceIndex].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    // Only run the huge test if the device has at least 8 GB of memory.

    return (memory >= 8*(long long)(1<<30));
}

void testSpatialBoxChange() {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(9, 0, 0), Vec3(0, 9, 0), Vec3(0, 0, 9));
    NonbondedForce* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::PME);
    force->setCutoffDistance(1.0);
    force->setEwaldErrorTolerance(1e-5);
    vector<Vec3> positions;
    for (int i = 0; i < 3001; i++) {
        system.addParticle(40);
        force->addParticle(i%2 == 0 ? 0.05 : -0.05, 0.25, 0.01);
        int j = (73*i)%3001;
        positions.push_back(Vec3(0.2+0.52*(j%15), 0.2+0.52*((j/15)%15), 0.2+0.52*(j/225)));
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001), referenceIntegrator(0.001);
    Context context(system, integrator, platform);
    if (platform.getPropertyValue(context, "AtomReorderingStatus") != "spatial")
        return;
    Context reference(system, referenceIntegrator, Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    reference.setPositions(positions);
    context.getState(State::Forces);
    // Changing the box can introduce neighbors even when no atom has moved.
    for (double size : {8.1, 9.0, 8.55}) {
        context.setPeriodicBoxVectors(Vec3(size, 0, 0), Vec3(0, size, 0), Vec3(0, 0, size));
        reference.setPeriodicBoxVectors(Vec3(size, 0, 0), Vec3(0, size, 0), Vec3(0, 0, size));
        State actual = context.getState(State::Forces | State::Energy);
        State expected = reference.getState(State::Forces | State::Energy);
        double error = 0, norm = 0;
        for (int i = 0; i < system.getNumParticles(); i++) {
            Vec3 delta = actual.getForces()[i]-expected.getForces()[i];
            error += delta.dot(delta);
            norm += expected.getForces()[i].dot(expected.getForces()[i]);
        }
        ASSERT(sqrt(error/norm) < 1e-5);
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), actual.getPotentialEnergy(), 1e-5);
    }
}

void runPlatformTests() {
    testSpatialBoxChange();
    testParallelComputation(NonbondedForce::NoCutoff);
    testParallelComputation(NonbondedForce::Ewald);
    testParallelComputation(NonbondedForce::PME);
    testParallelComputation(NonbondedForce::LJPME);
    testReordering();
    if (canRunHugeTest()) {
        double tol = (platform.getPropertyDefaultValue("Precision") == "single" ? 1e-4 : 1e-5);
        testHugeSystem(tol);
    }
}
