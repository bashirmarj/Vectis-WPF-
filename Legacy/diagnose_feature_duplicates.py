"""
diagnose_feature_duplicates.py
===============================

DIAGNOSTIC TOOL to understand why features aren't merging.

Usage:
    python diagnose_feature_duplicates.py PULLEY_SPA_67-1_1108.stp

This will show detailed information about each detected feature and
why they are or aren't being merged together.
"""

import sys
import logging
import numpy as np
from OCC.Extend.DataExchange import read_step_file
from production_turning_recognizer import ProductionTurningRecognizer

# Enable ALL logging including DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s: %(message)s'
)

def diagnose_pulley(step_file_path: str):
    """Run detailed diagnostics on pulley feature detection"""
    
    print("="*80)
    print("FEATURE DUPLICATE DIAGNOSTIC TOOL")
    print("="*80)
    print(f"\nAnalyzing: {step_file_path}\n")
    
    # Load STEP file
    shape = read_step_file(step_file_path)
    
    # Run turning recognizer
    recognizer = ProductionTurningRecognizer()
    result = recognizer.recognize_turning_features(shape)
    
    if result['part_type'] != 'rotational':
        print("❌ Not a rotational part - no turning features to diagnose")
        return
    
    features = result['features']
    
    print("\n" + "="*80)
    print("DETECTED FEATURES (Before Merging)")
    print("="*80)
    
    # Group by type
    grooves = [f for f in features if f.feature_type.value == 'groove']
    tapers = [f for f in features if f.feature_type.value == 'taper']
    steps = [f for f in features if f.feature_type.value == 'step']
    bases = [f for f in features if f.feature_type.value == 'base_cylinder']
    
    print(f"\n📊 Summary:")
    print(f"   Grooves: {len(grooves)}")
    print(f"   Tapers: {len(tapers)}")
    print(f"   Steps: {len(steps)}")
    print(f"   Bases: {len(bases)}")
    print(f"   Total: {len(features)}")
    
    # Detailed feature analysis
    if grooves:
        print("\n" + "-"*80)
        print("GROOVES (Detailed)")
        print("-"*80)
        for i, g in enumerate(grooves):
            print(f"\nGroove {i}:")
            print(f"  Location: ({g.location[0]:.2f}, {g.location[1]:.2f}, {g.location[2]:.2f})")
            print(f"  Axis: ({g.axis[0]:.3f}, {g.axis[1]:.3f}, {g.axis[2]:.3f})")
            print(f"  Diameter: Ø{g.diameter:.2f}mm")
            print(f"  Width: {g.groove_width:.2f}mm" if g.groove_width else "  Width: N/A")
            print(f"  Type: {g.groove_type}")
            print(f"  Face indices: {g.face_indices}")
        
        # Check distances between grooves
        if len(grooves) > 1:
            print("\n🔍 Groove Pairwise Analysis:")
            for i in range(len(grooves)):
                for j in range(i+1, len(grooves)):
                    g1, g2 = grooves[i], grooves[j]
                    
                    axis1 = np.array(g1.axis)
                    axis2 = np.array(g2.axis)
                    loc1 = np.array(g1.location)
                    loc2 = np.array(g2.location)
                    
                    # Axis alignment
                    axis_dot = abs(np.dot(axis1 / np.linalg.norm(axis1), 
                                         axis2 / np.linalg.norm(axis2)))
                    
                    # Perpendicular distance between axes
                    loc_vec = loc2 - loc1
                    cross = np.cross(axis1 / np.linalg.norm(axis1), loc_vec)
                    perp_dist = np.linalg.norm(cross)
                    
                    # Axial distance
                    axial_dist = abs(np.dot(loc_vec, axis1 / np.linalg.norm(axis1)))
                    
                    # Diameter difference
                    dia_diff = abs(g1.diameter - g2.diameter)
                    
                    print(f"\n   Groove {i} vs Groove {j}:")
                    print(f"      Axis alignment: {axis_dot:.4f} (need >0.996 for coaxial)")
                    print(f"      Perpendicular distance: {perp_dist:.2f}mm (need <5mm)")
                    print(f"      Axial distance: {axial_dist:.2f}mm (need <10mm)")
                    print(f"      Diameter difference: {dia_diff:.2f}mm (need <3mm)")
                    
                    should_merge = (axis_dot > 0.996 and perp_dist < 5.0 and 
                                  axial_dist < 10.0 and dia_diff < 3.0)
                    print(f"      ➜ Should merge: {'YES ✓' if should_merge else 'NO ✗'}")
    
    if tapers:
        print("\n" + "-"*80)
        print("TAPERS (Detailed)")
        print("-"*80)
        for i, t in enumerate(tapers):
            print(f"\nTaper {i}:")
            print(f"  Location: ({t.location[0]:.2f}, {t.location[1]:.2f}, {t.location[2]:.2f})")
            print(f"  Axis: ({t.axis[0]:.3f}, {t.axis[1]:.3f}, {t.axis[2]:.3f})")
            print(f"  Diameter: Ø{t.diameter:.2f}mm")
            print(f"  Angle: {t.taper_angle:.2f}°" if t.taper_angle else "  Angle: N/A")
            print(f"  Start→End: Ø{t.start_diameter:.2f}mm → Ø{t.end_diameter:.2f}mm" 
                  if t.start_diameter else "  Start→End: N/A")
            print(f"  Face indices: {t.face_indices}")
        
        # Check distances between tapers
        if len(tapers) > 1:
            print("\n🔍 Taper Pairwise Analysis:")
            for i in range(len(tapers)):
                for j in range(i+1, len(tapers)):
                    t1, t2 = tapers[i], tapers[j]
                    
                    axis1 = np.array(t1.axis)
                    axis2 = np.array(t2.axis)
                    loc1 = np.array(t1.location)
                    loc2 = np.array(t2.location)
                    
                    # Axis alignment
                    axis_dot = abs(np.dot(axis1 / np.linalg.norm(axis1), 
                                         axis2 / np.linalg.norm(axis2)))
                    
                    # Perpendicular distance
                    loc_vec = loc2 - loc1
                    cross = np.cross(axis1 / np.linalg.norm(axis1), loc_vec)
                    perp_dist = np.linalg.norm(cross)
                    
                    # Axial distance
                    axial_dist = abs(np.dot(loc_vec, axis1 / np.linalg.norm(axis1)))
                    
                    # Angle difference
                    angle_diff = abs((t1.taper_angle or 0.0) - (t2.taper_angle or 0.0))
                    
                    print(f"\n   Taper {i} vs Taper {j}:")
                    print(f"      Axis alignment: {axis_dot:.4f} (need >0.996)")
                    print(f"      Perpendicular distance: {perp_dist:.2f}mm (need <5mm)")
                    print(f"      Axial distance: {axial_dist:.2f}mm (need <10mm)")
                    print(f"      Angle difference: {angle_diff:.2f}° (need <10°)")
                    
                    should_merge = (axis_dot > 0.996 and perp_dist < 5.0 and 
                                  axial_dist < 10.0 and angle_diff < 10.0)
                    print(f"      ➜ Should merge: {'YES ✓' if should_merge else 'NO ✗'}")
    
    if steps:
        print("\n" + "-"*80)
        print("STEPS (Detailed)")
        print("-"*80)
        for i, s in enumerate(steps):
            print(f"\nStep {i}:")
            print(f"  Location: ({s.location[0]:.2f}, {s.location[1]:.2f}, {s.location[2]:.2f})")
            print(f"  Axis: ({s.axis[0]:.3f}, {s.axis[1]:.3f}, {s.axis[2]:.3f})")
            print(f"  Diameter: Ø{s.diameter:.2f}mm")
            print(f"  Depth: {s.step_depth:.2f}mm" if s.step_depth else "  Depth: N/A")
            print(f"  Face indices: {s.face_indices}")
        
        # Check distances between steps
        if len(steps) > 1:
            print("\n🔍 Step Pairwise Analysis:")
            for i in range(len(steps)):
                for j in range(i+1, len(steps)):
                    s1, s2 = steps[i], steps[j]
                    
                    axis1 = np.array(s1.axis)
                    axis2 = np.array(s2.axis)
                    loc1 = np.array(s1.location)
                    loc2 = np.array(s2.location)
                    
                    # Axis alignment
                    axis_dot = abs(np.dot(axis1 / np.linalg.norm(axis1), 
                                         axis2 / np.linalg.norm(axis2)))
                    
                    # Perpendicular distance
                    loc_vec = loc2 - loc1
                    cross = np.cross(axis1 / np.linalg.norm(axis1), loc_vec)
                    perp_dist = np.linalg.norm(cross)
                    
                    # Axial distance
                    axial_dist = abs(np.dot(loc_vec, axis1 / np.linalg.norm(axis1)))
                    
                    # Diameter difference
                    dia_diff = abs(s1.diameter - s2.diameter)
                    
                    print(f"\n   Step {i} vs Step {j}:")
                    print(f"      Axis alignment: {axis_dot:.4f} (need >0.996)")
                    print(f"      Perpendicular distance: {perp_dist:.2f}mm (need <5mm)")
                    print(f"      Axial distance: {axial_dist:.2f}mm (need <15mm)")
                    print(f"      Diameter difference: {dia_diff:.2f}mm (need <3mm)")
                    
                    should_merge = (axis_dot > 0.996 and perp_dist < 5.0 and 
                                  axial_dist < 15.0 and dia_diff < 3.0)
                    print(f"      ➜ Should merge: {'YES ✓' if should_merge else 'NO ✗'}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    if len(grooves) > 1:
        recommendations.append(f"• {len(grooves)} grooves detected - check if they should merge")
    if len(tapers) > 1:
        recommendations.append(f"• {len(tapers)} tapers detected - verify if V-groove or separate features")
    if len(steps) > 1:
        recommendations.append(f"• {len(steps)} steps detected - check if circular edge split")
    
    if not recommendations:
        print("\n✅ No obvious duplicates detected!")
    else:
        print("\n⚠️  Potential issues:")
        for rec in recommendations:
            print(f"   {rec}")
    
    print("\n💡 If features should merge but aren't:")
    print("   1. Check the pairwise analysis above")
    print("   2. Increase tolerances in turning_feature_merger.py")
    print("   3. Look for face_indices overlap - same faces = definitely duplicates")
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_feature_duplicates.py <step_file>")
        sys.exit(1)
    
    diagnose_pulley(sys.argv[1])
