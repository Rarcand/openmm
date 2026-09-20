// Periodic translation without changing atom identity or molecule equivalence.
KERNEL void recenterStableAtoms(int numGroups, GLOBAL const int* RESTRICT atoms,
        GLOBAL const int* RESTRICT starts, GLOBAL real4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL real4* RESTRICT correction,
#endif
        GLOBAL int4* RESTRICT shifts, mixed4 boxX, mixed4 boxY, mixed4 boxZ) {
    for (int group = GLOBAL_ID; group < numGroups; group += GLOBAL_SIZE) {
        int first = starts[group], end = starts[group+1];
        mixed3 center = make_mixed3(0,0,0);
        for (int j = first; j < end; j++) {
            int atom = atoms[j];
            real4 p = posq[atom];
            center.x += (mixed) p.x; center.y += (mixed) p.y; center.z += (mixed) p.z;
#ifdef USE_MIXED_PRECISION
            real4 c = correction[atom];
            center.x += (mixed) c.x; center.y += (mixed) c.y; center.z += (mixed) c.z;
#endif
        }
        mixed scale = 1/(mixed) (end-first);
        center.x *= scale; center.y *= scale; center.z *= scale;
        mixed sz = floor(center.z/boxZ.z);
        mixed sy = floor((center.y-sz*boxZ.y)/boxY.y);
        mixed sx = floor((center.x-sz*boxZ.x-sy*boxY.x)/boxX.x);
        // Avoid undefined integer conversions for nonfinite or enormous inputs.
        if (!(fabs(sx) < (mixed) 2147483000 && fabs(sy) < (mixed) 2147483000 && fabs(sz) < (mixed) 2147483000)) {
            shifts[group] = make_int4(0,0,0,1);
            continue;
        }
        shifts[group] = make_int4((int) sx,(int) sy,(int) sz,0);
        if (sx == 0 && sy == 0 && sz == 0) continue;
        mixed dx = sx*boxX.x+sy*boxY.x+sz*boxZ.x;
        mixed dy = sy*boxY.y+sz*boxZ.y;
        mixed dz = sz*boxZ.z;
        for (int j = first; j < end; j++) {
            int atom = atoms[j];
            real4 p = posq[atom];
            mixed px = (mixed) p.x, py = (mixed) p.y, pz = (mixed) p.z;
#ifdef USE_MIXED_PRECISION
            real4 c = correction[atom];
            px += (mixed) c.x; py += (mixed) c.y; pz += (mixed) c.z;
#endif
            px -= dx; py -= dy; pz -= dz;
            posq[atom] = make_real4((real) px,(real) py,(real) pz,p.w);
#ifdef USE_MIXED_PRECISION
            correction[atom] = make_real4(px-(real) px,py-(real) py,pz-(real) pz,0);
#endif
        }
    }
}
