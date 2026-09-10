#ifndef OPENMM_SPATIALNONBONDEDPOLICY_H_
#define OPENMM_SPATIALNONBONDEDPOLICY_H_
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "openmm/System.h"
#include "openmm/NonbondedForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/OpenMMException.h"
#include <map>
#include <string>
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include "openmm/CMAPTorsionForce.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomAngleForce.h"
#include "openmm/CustomTorsionForce.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/CustomCentroidBondForce.h"

namespace OpenMM {
/** Shared eligibility policy. Unknown independent neighbor consumers use legacy
 * execution in auto mode until explicitly migrated. This is not a claim of
 * spatial support for every Force implementation. */
class SpatialNonbondedPolicy {
public:
    static std::string unsupportedReason(const System& system) {
        bool nonbonded = false;
        for (int i=0; i<system.getNumForces(); i++) {
            const Force* f = &system.getForce(i);
            if (auto nb = dynamic_cast<const NonbondedForce*>(f)) {
                nonbonded = true;
                if (nb->getNonbondedMethod() == NonbondedForce::NoCutoff || nb->getNonbondedMethod() == NonbondedForce::CutoffNonPeriodic)
                    return "spatial execution requires a periodic cutoff";
                continue;
            }
            if (auto nb = dynamic_cast<const CustomNonbondedForce*>(f)) {
                nonbonded = true;
                if (nb->getNonbondedMethod() != CustomNonbondedForce::CutoffPeriodic)
                    return "spatial execution requires a periodic cutoff";
                if (nb->getNumInteractionGroups() != 0)
                    return "custom interaction groups use legacy execution";
                continue;
            }
            if (dynamic_cast<const HarmonicBondForce*>(f) ||
                    dynamic_cast<const HarmonicAngleForce*>(f) ||
                    dynamic_cast<const PeriodicTorsionForce*>(f) ||
                    dynamic_cast<const RBTorsionForce*>(f) ||
                    dynamic_cast<const CMAPTorsionForce*>(f) ||
                    dynamic_cast<const CMMotionRemover*>(f) ||
                    dynamic_cast<const MonteCarloBarostat*>(f) ||
                    dynamic_cast<const CustomBondForce*>(f) ||
                    dynamic_cast<const CustomAngleForce*>(f) ||
                    dynamic_cast<const CustomTorsionForce*>(f) ||
                    dynamic_cast<const CustomExternalForce*>(f) ||
                    dynamic_cast<const CustomCompoundBondForce*>(f) ||
                    dynamic_cast<const CustomCentroidBondForce*>(f))
                continue;
            return "force requires legacy execution: "+f->getName();
        }
        return nonbonded ? "" : "no supported nonbonded interaction";
    }
    static void configure(std::map<std::string, std::string>& properties, const System& system,
            int devices, bool supportedDevice = true) {
        const std::string requested = properties.at("AtomReordering");
        if (requested == "baseline") {
            properties["AtomReorderingStatus"] = "baseline";
            return;
        }
        std::string reason = devices != 1 ? "spatial execution currently requires one device" :
                (!supportedDevice ? "spatial execution requires a supported GPU layout" : unsupportedReason(system));
        if (!reason.empty() && requested != "auto")
            throw OpenMMException(reason);
        properties["AtomReorderingStatus"] = reason.empty() ? "spatial" : "baseline: "+reason;
        if (requested == "auto")
            properties["AtomReordering"] = reason.empty() ? "inverse" : "baseline";
    }
};
}
#endif
