// Native integration test only: not linked into production OpenMM libraries.
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomCentroidBondForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ReorderedArraySet.h"
#include "openmm/common/SpatialNonbondedPolicy.h"
#ifdef DIAGNOSTIC_CUDA
#include "CudaContext.h"
#else
#include "OpenCLContext.h"
#endif
#include "TestReorderedArraySet.h"
#include <sstream>
#include <cmath>
using namespace std;
namespace OpenMM {
static ComputeContext& device(ContextImpl& context) {
#ifdef DIAGNOSTIC_CUDA
    using Data = CudaPlatform::PlatformData;
#else
    using Data = OpenCLPlatform::PlatformData;
#endif
    return *static_cast<Data*>(context.getPlatformData())->contexts.at(0);
}
static int checks;
static void require(bool value, const string& message) {
    if (!value) throw OpenMMException(message);
    checks++;
}
static void rejects(function<void()> action) {
    bool caught=false;
    try { action(); } catch (const OpenMMException&) { caught=true; }
    require(caught,"Expected registration/lifecycle rejection");
}
class ProbeImpl : public NonbondedForceImpl {
public:
    ProbeImpl(const NonbondedForce& owner) : NonbondedForceImpl(owner), cc(nullptr) {}
    void initialize(ContextImpl& context) override {
        NonbondedForceImpl::initialize(context);
        cc=&device(context);
        ContextSelector select(*cc);
        for (int stride : {3,12,32}) {
            sources.emplace_back(new ComputeArray());
            destinations.emplace_back(new ComputeArray());
            outputs.emplace_back(new ComputeArray());
            sources.back()->initialize(*cc, cc->getNumAtoms(), stride, "probeSource");
            destinations.back()->initialize(*cc, cc->getPaddedNumAtoms(), stride, "probeSorted");
            outputs.back()->initialize(*cc, cc->getNumAtoms(), stride, "probeOutput");
            cc->addReorderedArray(*sources.back(), *destinations.back());
        }
        fill(0);
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override {
        double result=NonbondedForceImpl::calcForcesAndEnergy(context,includeForces,includeEnergy,groups);
        if ((groups&1)==0) return result;
        ContextSelector select(*cc);
        auto view=cc->getNonbondedUtilities().getSpatialWorkView(includeForces);
        require(view.numAtoms==cc->getNumAtoms() && view.paddedAtoms==cc->getPaddedNumAtoms(), "View dimensions");
        require(view.orderRevision>0 && view.exclusionCount!=nullptr && view.exclusionRows!=nullptr, "View exclusion contract");
        require(view.originalToSpatial==cc->getNonbondedUtilities().getInverseSpatialAtomOrder(), "View mapping mismatch");
        require(cc->getNonbondedUtilities().getReorderedParameterArray(*sources[0])==&destinations[0]->getArray(), "Independent parameter lookup");
        verify();
        if (includeForces && contribute) {
            if (!writer) {
                writer=cc->compileProgram("KERNEL void contribute(GLOBAL mm_ulong* forces, int count) { for(int i=GLOBAL_ID;i<count;i+=GLOBAL_SIZE) forces[i] += (mm_ulong)42950; }")->createKernel("contribute");
                writer->addArg(*view.forces); writer->addArg(view.numAtoms);
            }
            writer->execute(view.numAtoms,64);
            // A second acquisition/default consumer must not clear our contribution.
            cc->getNonbondedUtilities().getSpatialWorkView(true);
        }
        return result;
    }
    bool contribute=false;
    ComputeKernel writer;
    void fill(int generation) {
        ContextSelector select(*cc);
        expected.clear();
        for (int k=0;k<sources.size();k++) {
            int stride=sources[k]->getElementSize();
            expected.emplace_back(sources[k]->getSize()*stride);
            for (int i=0;i<expected.back().size();i++) expected.back()[i]=(i*31+generation*17+k*53)%251;
            sources[k]->upload(expected.back().data());
            vector<unsigned char> sentinel(destinations[k]->getSize()*stride,0xa5);
            destinations[k]->upload(sentinel.data());
        }
        cc->invalidateReorderedArrays();
    }
    void verify() {
        ContextSelector select(*cc);
        require(!cc->getReorderedArraySet().isDirty(), "Gather did not make registered contents current");
        auto* inverse=cc->getNonbondedUtilities().getInverseSpatialAtomOrder();
        require(inverse!=nullptr,"Inverse map unavailable after nonbonded evaluation");
        for (int k=0;k<sources.size();k++) {
            int stride=sources[k]->getElementSize();
            if (readers.size()<=k) {
                string body="KERNEL void consume(GLOBAL const int* inverse, GLOBAL const unsigned char* sorted, GLOBAL unsigned char* output) { for(int i=GLOBAL_ID;i<"+to_string(cc->getNumAtoms())+";i+=GLOBAL_SIZE) for(int b=0;b<"+to_string(stride)+";b++) output[i*"+to_string(stride)+"+b]=sorted[inverse[i]*"+to_string(stride)+"+b]; }";
                readers.push_back(cc->compileProgram(body)->createKernel("consume"));
                readers.back()->addArg(*inverse); readers.back()->addArg(*destinations[k]); readers.back()->addArg(*outputs[k]);
            }
            readers[k]->execute(cc->getNumAtoms(),64);
            vector<unsigned char> actual(cc->getNumAtoms()*stride), original(expected[k].size()), spatial(destinations[k]->getSize()*stride);
            outputs[k]->download(actual.data());sources[k]->download(original.data());destinations[k]->download(spatial.data());
            require(original==expected[k],"Gather modified original IDs");
            require(equal(actual.begin(),actual.end(),expected[k].begin()),"Independent inverse-map consumer read incorrect values");
            for (int i=cc->getNumAtoms()*stride;i<spatial.size();i++) require(spatial[i]==0xa5,"Gather changed caller-owned padding");
        }
    }
    ComputeContext* cc;
    vector<unique_ptr<ComputeArray>> sources,destinations,outputs;
    vector<vector<unsigned char>> expected;
    vector<ComputeKernel> readers;
};
class ProbeForce : public NonbondedForce {
public:
    mutable ProbeImpl* impl;
protected:
    ForceImpl* createImpl() const override { impl=new ProbeImpl(*this);return impl; }
};
class PackedProbeImpl : public NonbondedForceImpl {
public:
    PackedProbeImpl(const NonbondedForce& owner) : NonbondedForceImpl(owner) {}
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override {
        double result = NonbondedForceImpl::calcForcesAndEnergy(context, includeForces, includeEnergy, groups);
        auto& cc = device(context);
        require(cc.getNonbondedUtilities().getUsesStableAtomOrder(), "Packed probe did not select spatial execution");
        rejects([&]() { cc.getNonbondedUtilities().getSpatialWorkView(includeForces); });
        return result;
    }
};
class PackedProbeForce : public NonbondedForce {
protected:
    ForceImpl* createImpl() const override { return new PackedProbeImpl(*this); }
};
class ContextArrayTestDriver : public Platform {
public:
    const string& getName() const override { static string name="context array test";return name; }
    double getSpeed() const override {return 0;}
    bool supportsDoublePrecision() const override {return true;}
    bool supportsContext(Context& context) {
#ifdef DIAGNOSTIC_CUDA
        return true;
#else
        auto& cc = static_cast<OpenCLContext&>(device(getContextImpl(context)));
        return cc.getSIMDWidth() == 32 && cc.getDevice().getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_CPU;
#endif
    }
    int run(Context& first, Context& second, const string& precision) {
        checks=0;
        CompoundIntegrator compound;
        compound.addIntegrator(new VerletIntegrator(.001));
        require(!SpatialNonbondedPolicy::usesIndependentNeighborList(compound), "Unexpected legacy integrator selection");
        auto* nested = new CompoundIntegrator();
        nested->addIntegrator(new DPDIntegrator(300, 1, 1, .001));
        compound.addIntegrator(nested);
        require(SpatialNonbondedPolicy::usesIndependentNeighborList(compound), "Nested DPD integrator was not detected");
        auto& base=device(getContextImpl(first));
        auto& other=device(getContextImpl(second));
        checks+=testReorderedArraySet(base,other);
        rejects([&](){base.getReorderedArraySet();});
        System system;
        auto* force=new ProbeForce();
        force->setNonbondedMethod(NonbondedForce::CutoffPeriodic);force->setCutoffDistance(1.0);
        vector<Vec3> positions;
        for(int i=0;i<131;i++) {
            system.addParticle(39.9);force->addParticle((i%2?.01:-.01),.2,.01);
            positions.emplace_back(.2+(i%6)*.55,.2+((i/6)%6)*.55,.2+(i/36)*.55);
        }
        for(int i=0;i<130;i+=3)force->addException(i,i+1,0,.2,0);
        system.addForce(force);
        auto* external=new CustomExternalForce("0.01*x*x");external->setForceGroup(1);external->addParticle(0,{});system.addForce(external);
        system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(.2,4,0),Vec3(-.1,.1,4));
        map<string,string> properties={{"Precision",precision},{"DeviceIndex",first.getPlatform().getPropertyValue(first,"DeviceIndex")}};
#ifndef DIAGNOSTIC_CUDA
        properties["OpenCLPlatformIndex"]=first.getPlatform().getPropertyValue(first,"OpenCLPlatformIndex");
#endif
        // Two standard forces select the full-tile interface on CUDA.
        auto* secondForce = new NonbondedForce();
        secondForce->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
        secondForce->setCutoffDistance(1.0);
        for (int i = 0; i < system.getNumParticles(); i++) secondForce->addParticle(0, .2, 0);
        for (int i = 0; i < 130; i += 3) secondForce->addException(i, i+1, 0, .2, 0);
        system.addForce(secondForce);
        VerletIntegrator integrator(.00001);
        Context context(system,integrator,first.getPlatform(),properties);
        context.setPositions(positions);context.setVelocities(vector<Vec3>(131));
        auto snapshot=[&](int flags=State::Forces|State::Energy,int groups=1){return context.getState(flags,false,groups);};
        auto without=snapshot();
        force->impl->contribute=true;
        auto with=snapshot();
        for(int i=0;i<131;i++) {
            require(abs((with.getForces()[i][0]-without.getForces()[i][0])-ldexp(42950.0,-32))<1e-7, "Independent force was cleared or merged incorrectly");
            require(abs(with.getForces()[i][1]-without.getForces()[i][1])<1e-7, "Independent force changed wrong component");
        }
        force->impl->contribute=false;
        rejects([&](){force->impl->cc->getNonbondedUtilities().getSpatialWorkView(false);});
        snapshot();force->impl->verify();
        auto& cc=*force->impl->cc;
        rejects([&](){cc.addReorderedArray(*force->impl->sources[0],*force->impl->destinations[0]);});
        for(int round=1;round<=3;round++) {
            force->impl->fill(round);
            snapshot(State::Energy,2);
            require(cc.getReorderedArraySet().isDirty(),"Unrelated force group consumed pending gather");
            if(round==2) cc.setStepsSinceReorder(250);
            snapshot(State::Energy);force->impl->verify();
            snapshot();force->impl->verify();
        }
        // Reallocate both ends after compilation, without changing array objects.
        {
            ContextSelector select(cc);
            for(auto& array:force->impl->sources) array->resize(cc.getNumAtoms()+13);
            for(auto& array:force->impl->destinations) array->resize(cc.getPaddedNumAtoms()+32);
            force->impl->fill(4);
        }
        snapshot();force->impl->verify();
        {
            ContextSelector select(cc);
            force->impl->destinations[0]->resize(cc.getPaddedNumAtoms()-1);
            rejects([&](){cc.getReorderedArraySet().validate();});
            force->impl->destinations[0]->resize(cc.getPaddedNumAtoms()+32);force->impl->fill(5);
        }
        snapshot();force->impl->verify();
        force->setParticleParameters(0,.02,.2,.01);force->updateParametersInContext(context);
        snapshot();force->impl->verify();
        stringstream checkpoint(ios::in|ios::out|ios::binary);context.createCheckpoint(checkpoint);
        integrator.step(251);snapshot();force->impl->verify();
        checkpoint.seekg(0);context.loadCheckpoint(checkpoint);snapshot();force->impl->verify();
        // New ForceImpl and ComputeContext must register fresh arrays before closure.
        context.reinitialize(true);snapshot();force->impl->verify();
        // Centroid bonds must retain their geometry across periodic recentering.
        {
            System centroidSystem;
            auto* nonbonded = new NonbondedForce();
            nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
            nonbonded->setCutoffDistance(1.0);
            for (int i = 0; i < 4; i++) {
                centroidSystem.addParticle(1);
                nonbonded->addParticle(0, .1, .01);
            }
            centroidSystem.addForce(nonbonded);
            auto* centroid = new CustomCentroidBondForce(2, "distance(g1,g2)^2");
            centroid->addGroup({0, 1});
            centroid->addGroup({2, 3});
            centroid->addBond({0, 1});
            centroid->setForceGroup(1);
            centroidSystem.addForce(centroid);
            centroidSystem.setDefaultPeriodicBoxVectors(Vec3(4,0,0), Vec3(0,4,0), Vec3(0,0,4));
            VerletIntegrator centroidIntegrator(.001);
            Context centroidContext(centroidSystem, centroidIntegrator, first.getPlatform(), properties);
            centroidContext.setPositions({Vec3(3.9,1,1), Vec3(4.1,1,1), Vec3(4.2,1,1), Vec3(4.4,1,1)});
            auto before = centroidContext.getState(State::Energy|State::Forces, false, 2);
            require(abs(before.getPotentialEnergy()-.09) < 1e-5, "Incorrect centroid bond energy");
            // A linked context selects eligibility for its own integrator.
            DPDIntegrator linkedIntegrator(300, 1, 1, .001);
            unique_ptr<Context> linked(getContextImpl(centroidContext).createLinkedContext(centroidSystem, linkedIntegrator));
            linked->setPositions({Vec3(3.9,1,1), Vec3(4.1,1,1), Vec3(4.2,1,1), Vec3(4.4,1,1)});
            auto& linkedCC = device(getContextImpl(*linked));
            require(!linkedCC.getNonbondedUtilities().getUsesStableAtomOrder(), "Linked DPD context did not select legacy execution");
            require(isfinite(linked->getState(State::Energy).getPotentialEnergy()), "Invalid linked context energy");
            auto& centroidCC = device(getContextImpl(centroidContext));
            require(centroidCC.getNonbondedUtilities().getUsesStableAtomOrder(), "Centroid probe did not select spatial execution");
            centroidCC.forceReorder();
            centroidCC.reorderAtoms();
            auto after = centroidContext.getState(State::Energy|State::Forces, false, 2);
            require(abs(before.getPotentialEnergy()-after.getPotentialEnergy()) < 1e-5, "Recentering changed centroid bond energy");
            for (int i = 0; i < 4; i++) {
                Vec3 delta = before.getForces()[i]-after.getForces()[i];
                require(delta.dot(delta) < 1e-10, "Recentering changed centroid bond forces");
            }
        }
        // Packed masks cannot be represented by the full-tile consumer API.
        System packedSystem;
        auto* packedForce = new PackedProbeForce();
        packedForce->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
        packedForce->setCutoffDistance(1.0);
        vector<Vec3> packedPositions;
        for (int i = 0; i < 3001; i++) {
            packedSystem.addParticle(39.9);
            packedForce->addParticle(i%2 ? .01 : -.01, .1, .01);
            packedPositions.emplace_back(.2+.25*(i%15), .2+.25*((i/15)%15), .2+.25*(i/225));
        }
        packedForce->addException(0, 1500, 0, .1, 0);
        packedSystem.addForce(packedForce);
        packedSystem.setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));
        VerletIntegrator packedIntegrator(.001);
        Context packedContext(packedSystem, packedIntegrator, first.getPlatform(), properties);
        packedContext.setPositions(packedPositions);
        require(isfinite(packedContext.getState(State::Energy).getPotentialEnergy()), "Packed probe energy is invalid");
        return checks;
    }
};
}
#ifdef _WIN32
#define SPATIAL_TEST_EXPORT __declspec(dllexport)
#else
#define SPATIAL_TEST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" SPATIAL_TEST_EXPORT const char* testContextArrays(OpenMM::Context* a,OpenMM::Context* b,const char* precision) {
    static thread_local string result;
    try {OpenMM::ContextArrayTestDriver driver;result="{\"passed\":true,\"assertions\":"+to_string(driver.run(*a,*b,precision))+"}";}
    catch(const exception& e){result=string("ERROR: ")+e.what();}
    return result.c_str();
}

#ifdef SPATIAL_TEST_EXECUTABLE
#include <iostream>
int main(int argc, char** argv) {
    try {
        string precision=argc>1 ? argv[1] : "mixed";
#ifdef DIAGNOSTIC_CUDA
        OpenMM::CudaPlatform platform;
#else
        OpenMM::OpenCLPlatform platform;
#endif
        OpenMM::System system;
        for(int i=0;i<131;i++)system.addParticle(1);
        map<string,string> properties={{"Precision",precision},{"DeviceIndex","0"}};
#ifndef DIAGNOSTIC_CUDA
        properties["OpenCLPlatformIndex"]=argc>2 ? argv[2] : "";
        properties["DeviceIndex"]=argc>3 ? argv[3] : "";
#endif
        OpenMM::VerletIntegrator a(.001),b(.001);
        OpenMM::Context first(system,a,platform,properties),second(system,b,platform,properties);
        OpenMM::ContextArrayTestDriver driver;
        if (!driver.supportsContext(first)) return 77;
        cout << driver.run(first,second,precision) << " assertions passed" << endl;
        return 0;
    } catch(const exception& e) {cerr << e.what() << endl;return 1;}
}
#endif
