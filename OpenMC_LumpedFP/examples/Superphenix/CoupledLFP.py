"""CoupledLFP.py — Coupled MG transport + depletion with LFP lumping.

Step-4 validation: same physical pin-cell as the CE reference (Depletion_33G),
same burnup schedule, but using the LFP MGXS library (MGXS_Library) and the
custom depletion chain that replaces all fission products with one fictitious
LFP nuclide per fissile parent (LFP_Pu239, LFP_U238, …).

Any deviation w.r.t. the CE reference (Depletion_33G) accumulates three
distinct contributions:
  1. 33-group energy-group collapse error (same as the AllFP MG reference)
  2. Transport spectral error from using LFP XS instead of individual FP XS
  3. Depletion error from the lumped yield/rate approximation in the LFP chain

Results are stored in run/ for subsequent analysis with Analysis.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openmc
import openmc.model

sys.path.insert(0, "/data/user/pati_f/LFP_Thesis/library")
from WrapClass.coupled_depletion_driver import CoupledDepletionDriver  # noqa: E402
from WrapClass.mgxs_library_manager import MGXSLibraryManager  # noqa: E402

# ============================ paths / config ============================

THESIS = Path("/data/user/pati_f/LFP_Thesis")
CHAIN_FILE = THESIS / "library" / "custom_chain.xml"
LIB_ROOT   = THESIS / "library" / "MGXS_Library"
OUT_DIR = Path(__file__).resolve().parent / "run"

POWER_W            = 201.1    # W — matches CE Depletion_33G reference
FUEL_VOLUME        = 0.4004   # cm³
FUEL_TEMPERATURE   = 1500.0   # K
STRUCT_TEMPERATURE = 673.0    # K

# Cumulative burnup targets [MWd/kgHM] — identical to CE reference
BURNUP_TARGETS = [
    0.05, 0.30, 1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 13.00, 17.00,
    21.00, 26.00, 34.00, 42.00, 51.00, 60.00, 70.00, 80.00, 90.00, 100.00,
]

# Pin-cell geometry (same as CE reference)
FUEL_R = 0.357
GAP_R  = 0.371
CLAD_R = 0.4321
EDGE   = 0.5696


# ============================ materials ============================

def build_initial_fuel() -> openmc.Material:
    m = openmc.Material(material_id=1, name="fuel")
    m.add_nuclide("U235",  9.99687e-05, "ao")
    m.add_nuclide("U238",  1.96426e-02, "ao")
    m.add_nuclide("Pu238", 1.76060e-05, "ao")
    m.add_nuclide("Pu239", 2.42806e-03, "ao")
    m.add_nuclide("Pu240", 7.23938e-04, "ao")
    m.add_nuclide("Pu241", 1.94633e-04, "ao")
    m.add_nuclide("Pu242", 6.75511e-05, "ao")
    m.add_nuclide("Am241", 4.76364e-05, "ao")
    m.add_nuclide("O16",   4.59794e-02, "ao")
    m.depletable = True
    m.volume = FUEL_VOLUME
    m.temperature = FUEL_TEMPERATURE
    return m


def build_structural_materials() -> list[openmc.Material]:
    gap = openmc.Material(material_id=2, name="gap")
    gap.add_nuclide("He4", 0.000361, "ao")
    gap.depletable = False
    gap.temperature = STRUCT_TEMPERATURE

    clad = openmc.Material(material_id=3, name="clad")
    clad.add_element("C",    1.95513e-04, "ao")
    clad.add_nuclide("Si28", 9.28063e-04, "ao")
    clad.add_nuclide("Si29", 4.55200e-05, "ao")
    clad.add_nuclide("Si30", 2.90426e-05, "ao")
    clad.add_nuclide("P31",  4.54479e-05, "ao")
    clad.add_nuclide("Ti46", 3.36969e-05, "ao")
    clad.add_nuclide("Ti47", 2.97418e-05, "ao")
    clad.add_nuclide("Ti48", 2.88578e-04, "ao")
    clad.add_nuclide("Ti49", 2.07448e-05, "ao")
    clad.add_nuclide("Ti50", 1.94665e-05, "ao")
    clad.add_nuclide("Cr50", 6.53123e-04, "ao")
    clad.add_nuclide("Cr52", 1.21112e-02, "ao")
    clad.add_nuclide("Cr53", 1.34737e-03, "ao")
    clad.add_nuclide("Cr54", 3.29182e-04, "ao")
    clad.add_nuclide("Mn55", 1.45198e-03, "ao")
    clad.add_nuclide("Fe54", 3.29082e-03, "ao")
    clad.add_nuclide("Fe56", 4.98158e-02, "ao")
    clad.add_nuclide("Fe57", 1.13025e-03, "ao")
    clad.add_nuclide("Fe58", 1.47824e-04, "ao")
    clad.add_nuclide("Ni58", 7.71919e-03, "ao")
    clad.add_nuclide("Ni60", 2.87440e-03, "ao")
    clad.add_nuclide("Ni62", 3.85551e-04, "ao")
    clad.add_nuclide("Ni64", 9.51043e-05, "ao")
    clad.add_nuclide("Mo92", 1.85458e-04, "ao")
    clad.add_nuclide("Mo94", 1.14304e-04, "ao")
    clad.add_nuclide("Mo95", 1.95789e-04, "ao")
    clad.add_nuclide("Mo96", 2.03902e-04, "ao")
    clad.add_nuclide("Mo97", 1.16211e-04, "ao")
    clad.add_nuclide("Mo98", 2.92234e-04, "ao")
    clad.add_nuclide("Mo100", 1.15302e-04, "ao")
    clad.depletable = False
    clad.temperature = STRUCT_TEMPERATURE

    cool = openmc.Material(material_id=4, name="cool")
    cool.add_nuclide("Na23", 0.0219, "ao")
    cool.depletable = False
    cool.temperature = STRUCT_TEMPERATURE

    return [gap, clad, cool]


# ============================ builders ============================

def geometry_builder(materials: list[openmc.Material]) -> openmc.Geometry:
    fuel_m, gap_m, clad_m, cool_m = materials

    fuel_s = openmc.ZCylinder(r=FUEL_R)
    gap_s  = openmc.ZCylinder(r=GAP_R)
    clad_s = openmc.ZCylinder(r=CLAD_R)
    upper  = openmc.ZPlane(z0=0.5,  boundary_type="reflective")
    lower  = openmc.ZPlane(z0=-0.5, boundary_type="reflective")

    pin_u    = openmc.model.pin(surfaces=[fuel_s, gap_s, clad_s],
                                items=[fuel_m, gap_m, clad_m, cool_m])
    pin_rect = openmc.model.HexagonalPrism(edge_length=EDGE,
                                           boundary_type="reflective")
    region = -pin_rect & +lower & -upper
    main   = openmc.Cell(fill=pin_u, region=region)
    root   = openmc.Universe(name="root universe", universe_id=0)
    root.add_cell(main)
    return openmc.Geometry(root)


def settings_builder() -> openmc.Settings:
    s = openmc.Settings()
    s.energy_mode = "multi-group"
    s.batches  = 450
    s.inactive = 150
    s.particles = 100000

    # Shannon-entropy mesh disabled — for this pin-cell geometry the source
    # converges in a few generations anyway, and the 20×20×1 (=400 cells)
    # mesh adds a per-particle scoring overhead that slows MG transport
    # ~4× (108 k → 24 k particles/s in this workflow). Re-enable with a
    # coarser mesh, e.g. (5, 5, 1), if you really want a convergence check.

    uniform = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Uniform(a=0,       b=FUEL_R),
        phi=openmc.stats.Uniform(a=0,     b=2 * np.pi),
        z=openmc.stats.Uniform(a=-0.5,    b=0.5),
    )
    s.source   = openmc.IndependentSource(space=uniform,
                                          constraints={"fissionable": True})
    s.run_mode = "eigenvalue"
    return s


def tallies_builder(fuel_material: openmc.Material) -> openmc.Tallies:
    groups    = openmc.mgxs.EnergyGroups(group_edges="ECCO-33")
    e_filter  = openmc.EnergyFilter(groups.group_edges)
    mat_filter = openmc.MaterialFilter([fuel_material])

    flux_tally = openmc.Tally(name="flux")
    flux_tally.filters = [mat_filter, e_filter]
    flux_tally.scores  = ["flux"]

    fiss_tally = openmc.Tally(name="fiss_rate")
    fiss_tally.filters = [mat_filter, e_filter]   # restrict scoring to the
                                                  # fuel cell only — matches
                                                  # the old script and avoids
                                                  # per-collision σ_f evaluation
                                                  # on structural materials.
    fiss_tally.scores  = ["fission"]

    return openmc.Tallies([flux_tally, fiss_tally])


# ============================ main ============================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    openmc.config["chain_file"] = str(CHAIN_FILE)

    library_manager = MGXSLibraryManager(library_base_path=LIB_ROOT)

    fuel       = build_initial_fuel()
    structurals = build_structural_materials()

    driver = CoupledDepletionDriver(
        library_manager=library_manager,
        geometry_builder=geometry_builder,
        settings_builder=settings_builder,
        tallies_builder=tallies_builder,
        chain_file=str(CHAIN_FILE),
        power=POWER_W,
        output_dir=OUT_DIR,
       )
    
    driver.setup_initial_materials(fuel, structurals)
    print(f"\n[INFO] Burnup targets: {BURNUP_TARGETS}")
    driver.run_burnup_schedule(BURNUP_TARGETS)


if __name__ == "__main__":
    main()
