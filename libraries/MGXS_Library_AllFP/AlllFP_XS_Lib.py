"""
AllFP_XS_Lib.py

Explicit-FP counterpart of LFP_XS_library.py.

Pre-compute and save cross-section libraries for both depletion and transport.

Unlike LFP_XS_library.py, this script does NOT lump fission products: every
nuclide present in the MGXS fuel material is written directly to the library,
skipping the yield-weighted averaging step entirely. There are no LFP_*
nuclides and no yield files involved.

For each burnup step, this script creates:
1. MicroXS (for depletion) - saved to MGXS_Library_AllFP/depletion/BU_{burnup}/
2. MGXSLibrary with XSdata (for transport) - saved to MGXS_Library_AllFP/transport/BU_{burnup}/

The output directory layout, burnup-value naming (BU_X.XXX), base-materials
handling and library_metadata.json are identical to LFP_XS_library.py, so the
produced library is directly consumable by the WrapClass MGXSLibraryManager.

Usage:
    python AllFP_XS_Lib.py [--steps 1 2 3 ...] [--temperature 1500]

    Default: processes all burnup steps at T=1500K
"""

import os
import json
import argparse
import pickle
import numpy as np
import openmc
import openmc.mgxs
import openmc.deplete
from pathlib import Path
from functools import lru_cache

# ================== CONFIGURATION ==================

# Burnup schedule [MWd/kgHM] — delta steps, matching Depletion_33G.py
BU_STEPS = [0.05, 0.25, 0.7, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0]
NSTEPS = len(BU_STEPS) + 1  # +1 for fresh fuel (BU=0)

# Paths
MGXS_RUN_ROOT = Path("/data/user/pati_f/LFP_Thesis/MGXS_runs_33G")          # TODO: update when data available

# Output directories — separate from the LFP library to avoid clobbering it
OUTPUT_BASE = Path("/data/user/pati_f/LFP_Thesis/library/MGXS_Library_AllFP")
OUTPUT_DEPLETION = OUTPUT_BASE / "depletion"
OUTPUT_TRANSPORT = OUTPUT_BASE / "transport"
OUTPUT_BASE_MATERIALS = OUTPUT_BASE / "base"

# Base materials (clad, gap, coolant) MGXS source
BASE_MGXS_FILE = Path("/data/user/pati_f/LFP_Thesis/CladCool/mgxs/mgxs.pkl")  # TODO: update when data available
BASE_TEMPERATURE = 673.0  # K (temperature for base materials)

# Energy groups
N_ENERGY_GROUPS = 33
ENERGY_GROUPS = openmc.mgxs.EnergyGroups(group_edges='ECCO-33')

# Default temperature
DEFAULT_TEMPERATURE = 1500  # K

# Reactions for depletion (MicroXS)
DEPLETION_REACTIONS = ['fission', '(n,gamma)', '(n,2n)','(n,p)','(n,3n)','(n,a)']

# Reactions for transport (XSdata)
# 'multiplicity matrix' is kept for CladCool old library compatibility.
TRANSPORT_REACTIONS = [
    'total', 'absorption', 'fission', 'nu-fission',
    'chi', 'nu-scatter matrix', 'scatter matrix',
    'multiplicity matrix'
]

# All reactions to extract from MGXS library
ALL_REACTIONS = list(set(DEPLETION_REACTIONS + TRANSPORT_REACTIONS))

# ================== UTILITY FUNCTIONS ==================

def calculate_burnup_points():
    """
    Calculate cumulative burnup [MWd/kgHM] for each step index (0 = fresh fuel).
    Burnup is defined directly from BU_STEPS (no power/mass conversion needed).

    Returns:
        dict: {step_index: burnup_MWdkg}
    """
    burnup_map = {0: 0.0}
    cumulative = 0.0
    for i, dbu in enumerate(BU_STEPS):
        cumulative += dbu
        burnup_map[i + 1] = round(cumulative, 6)
    return burnup_map


@lru_cache(maxsize=20)
def load_mgxs_library(bu_step, temperature):
    """Load MGXS library for a specific burnup step (cached)"""
    mgxs_path = MGXS_RUN_ROOT / f"BU_{bu_step}" / f"T_{temperature}" / "mgxs"

    if not mgxs_path.exists():
        raise FileNotFoundError(f"MGXS library not found: {mgxs_path}")

    mgxs_lib = openmc.mgxs.Library.load_from_file(
        filename="mgxs",
        directory=str(mgxs_path)
    )

    return mgxs_lib


@lru_cache(maxsize=20)
def load_materials_from_run(bu_step, temperature):
    """Load materials.xml from MGXS run (cached)"""
    mat_file = MGXS_RUN_ROOT / f"BU_{bu_step}" / f"T_{temperature}" / "materials.xml"

    if not mat_file.exists():
        raise FileNotFoundError(f"Materials file not found: {mat_file}")

    materials = openmc.Materials.from_xml(str(mat_file))
    fuel_material = materials[0]  # Assume first material is fuel

    return fuel_material


# Identity map (no renaming needed when using direct scatter/nu-scatter)
_CANONICAL_KEY = {}

def extract_xs_for_nuclide(mgxs_lib, fuel_material, nuclide, reactions):
    """
    Extract cross sections for a single nuclide across all reactions.

    Returns:
        dict: {reaction: np.array([33])} or None for matrix reactions
              For 'scatter matrix': returns (33, 33) array
    """
    xs_data = {}

    for reaction in reactions:
        try:
            mgxs_obj = mgxs_lib.get_mgxs(domain=fuel_material, mgxs_type=reaction)

            # Extract microscopic XS
            xs = mgxs_obj.get_xs(
                nuclides=[nuclide],
                xs_type='micro',
                value='mean'
            )

            # Use canonical key for storage (strip 'consistent ' prefix)
            key = _CANONICAL_KEY.get(reaction, reaction)

            if xs is not None:
                xs_array = np.array(xs)

                if key in ('nu-scatter matrix', 'scatter matrix'):
                    # Reshape to (g_in, g_out) = (33, 33) only if size matches
                    if xs_array.size == N_ENERGY_GROUPS * N_ENERGY_GROUPS:
                        xs_data[key] = xs_array.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
                    else:
                        continue
                elif key == 'chi':
                    # Chi is (33,) - fission spectrum
                    xs_data[key] = xs_array.flatten()
                else:
                    # Regular 1D cross section
                    xs_data[key] = xs_array.flatten()

        except (KeyError, ValueError, Exception) as e:
            # Nuclide not present for this reaction - silently skip
            continue

    return xs_data


def extract_all_xs(mgxs_lib, fuel_material, nuclide_list, reactions):
    """
    Extract cross sections for all nuclides and reactions.

    Returns:
        dict: {nuclide: {reaction: np.array}}
    """
    all_xs = {}

    for nuclide in nuclide_list:
        xs_data = extract_xs_for_nuclide(mgxs_lib, fuel_material, nuclide, reactions)
        if xs_data:  # Only store if we got some data
            all_xs[nuclide] = xs_data

    return all_xs


# ================== MICROXS (DEPLETION) ==================

def build_microxs(all_xs):
    """
    Build MicroXS array for depletion.

    Every nuclide for which cross sections were extracted is written directly,
    with no yield-weighted lumping.

    Returns:
        openmc.deplete.MicroXS object
    """
    nuclides = list(all_xs.keys())
    n_nuclides = len(nuclides)
    n_reactions = len(DEPLETION_REACTIONS)

    data = np.zeros((n_nuclides, n_reactions, N_ENERGY_GROUPS), dtype=float)
    idx_rxn = {rxn: j for j, rxn in enumerate(DEPLETION_REACTIONS)}

    n_filled = 0

    for i, nuc in enumerate(nuclides):
        for reaction in DEPLETION_REACTIONS:
            if reaction in all_xs[nuc]:
                j = idx_rxn[reaction]
                data[i, j, :] = all_xs[nuc][reaction]
                n_filled += 1

    print(f"       MicroXS: filled {n_filled} entries for {n_nuclides} nuclides")

    micro_xs = openmc.deplete.MicroXS(
        data=data,
        nuclides=nuclides,
        reactions=DEPLETION_REACTIONS
    )

    return micro_xs


# ================== MGXSLIBRARY (TRANSPORT) ==================

def build_xsdata_for_nuclide(nuclide, xs_dict, temperature):
    """
    Build XSdata object for a single nuclide.

    Parameters:
        nuclide: Nuclide name
        xs_dict: {reaction: np.array} for this nuclide
        temperature: Temperature in K, or list of temperatures.
                     When multiple temperatures are given, the same XS data
                     is replicated at each temperature point (useful for base
                     materials that need to be available at both 673K and 1500K).

    Returns:
        openmc.XSdata object
    """
    # Support single temperature or list of temperatures
    if isinstance(temperature, (int, float)):
        temperatures = [float(temperature)]
    else:
        temperatures = [float(t) for t in temperature]

    xsdata = openmc.XSdata(nuclide, energy_groups=ENERGY_GROUPS, temperatures=temperatures)
    xsdata.order = 0  # P0 scattering

    for temp in temperatures:
        # Set each reaction type
        if 'total' in xs_dict:
            xsdata.set_total(xs_dict['total'], temperature=temp)

        if 'absorption' in xs_dict:
            xsdata.set_absorption(xs_dict['absorption'], temperature=temp)

        if 'fission' in xs_dict:
            xsdata.set_fission(xs_dict['fission'], temperature=temp)

        if 'nu-fission' in xs_dict:
            xsdata.set_nu_fission(xs_dict['nu-fission'], temperature=temp)

        if 'chi' in xs_dict:
            # Chi should sum to 1 (fission spectrum)
            chi = xs_dict['chi']
            if chi.sum() > 0:
                chi = chi / chi.sum()  # Normalize
            xsdata.set_chi(chi, temperature=temp)

        if 'nu-scatter matrix' in xs_dict:
            # OpenMC set_scatter_matrix expects the scattering production
            # matrix (nu-scatter).  Shape: (G_in, G_out, scatt_order).
            nu_scat = xs_dict['nu-scatter matrix']
            if nu_scat.ndim == 1:
                nu_scat = nu_scat.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
            scatter_p0 = nu_scat.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS, 1)
            xsdata.set_scatter_matrix(scatter_p0, temperature=temp)

        # Set multiplicity matrix.
        # Case 1: scatter matrix available → reconstruct mult = nu-scatter / scatter
        #         (used for fuel nuclides from MGXS_runs_33G_finer)
        # Case 2: multiplicity matrix available directly → use as-is
        #         (used for CladCool nuclides from old library)
        if 'nu-scatter matrix' in xs_dict and 'scatter matrix' in xs_dict:
            nu_scat = xs_dict['nu-scatter matrix']
            scat = xs_dict['scatter matrix']
            if nu_scat.ndim == 1:
                nu_scat = nu_scat.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
            if scat.ndim == 1:
                scat = scat.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
            mult = np.ones_like(nu_scat)
            nonzero = scat != 0.0
            mult[nonzero] = nu_scat[nonzero] / scat[nonzero]
            xsdata.set_multiplicity_matrix(mult, temperature=temp)
        elif 'multiplicity matrix' in xs_dict:
            mult = xs_dict['multiplicity matrix']
            if mult.ndim == 1:
                mult = mult.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
            xsdata.set_multiplicity_matrix(mult, temperature=temp)

    return xsdata


def build_mgxs_library(all_xs, temperature):
    """
    Build MGXSLibrary for transport.

    Every nuclide for which cross sections were extracted is written directly,
    with no yield-weighted lumping.

    Parameters:
        all_xs: Dictionary of all XS {nuclide: {reaction: xs_array}}
        temperature: Temperature in K

    Returns:
        openmc.MGXSLibrary object
    """
    mg_lib = openmc.MGXSLibrary(ENERGY_GROUPS)

    n_added = 0

    for nuc, xs_dict in all_xs.items():
        if not xs_dict:
            continue

        # Build XSdata for this nuclide
        try:
            xsdata = build_xsdata_for_nuclide(nuc, xs_dict, temperature)
            mg_lib.add_xsdata(xsdata)
            n_added += 1
        except Exception as e:
            print(f"  [WARN] Could not create XSdata for {nuc}: {e}")
            continue

    print(f"       MGXSLibrary: added {n_added} nuclides")

    return mg_lib


# ================== BASE MATERIALS (CLAD, GAP, COOLANT) ==================

def extract_xs_from_mgxs_lib_for_nuclide(mgxs_lib, material, nuclide, reactions):
    """
    Extract cross sections for a single nuclide from an openmc.mgxs.Library.

    Returns:
        dict: {reaction: np.array}
    """
    xs_data = {}

    for reaction in reactions:
        key = _CANONICAL_KEY.get(reaction, reaction)

        candidates = [reaction]

        xs_array = None
        for candidate in candidates:
            try:
                mgxs_obj = mgxs_lib.get_mgxs(domain=material, mgxs_type=candidate)
                xs = mgxs_obj.get_xs(nuclides=[nuclide], xs_type='micro', value='mean')
                if xs is not None:
                    xs_array = np.array(xs)
                    break
            except (KeyError, ValueError, Exception):
                continue

        if xs_array is None:
            continue

        if key in ('nu-scatter matrix', 'scatter matrix'):
            if xs_array.size == N_ENERGY_GROUPS * N_ENERGY_GROUPS:
                xs_data[key] = xs_array.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
        elif key == 'multiplicity matrix':
            if xs_array.size == N_ENERGY_GROUPS * N_ENERGY_GROUPS:
                xs_data[key] = xs_array.reshape(N_ENERGY_GROUPS, N_ENERGY_GROUPS)
        elif key == 'chi':
            xs_data[key] = xs_array.flatten()
        else:
            xs_data[key] = xs_array.flatten()

    return xs_data


def process_base_materials():
    """
    Process base materials (clad, gap, coolant) and save their XS for transport.

    These materials don't change with burnup, so we only need to process them once.
    """
    print("\n" + "="*60)
    print("Processing BASE MATERIALS (clad, gap, coolant)")
    print("="*60)

    # Create output directory
    OUTPUT_BASE_MATERIALS.mkdir(parents=True, exist_ok=True)

    # Load MGXS library from pickle file
    print("[1/3] Loading base materials MGXS library...")
    if not BASE_MGXS_FILE.exists():
        print(f"  [ERROR] Base MGXS file not found: {BASE_MGXS_FILE}")
        return False

    with open(BASE_MGXS_FILE, 'rb') as f:
        mgxs_lib = pickle.load(f)

    print(f"       Loaded MGXS library with {len(mgxs_lib.domains)} domains")

    # Build MGXSLibrary for base materials
    print("[2/3] Building MGXSLibrary for base materials...")
    mg_lib = openmc.MGXSLibrary(ENERGY_GROUPS)

    base_nuclides = []
    material_info = {}

    for material in mgxs_lib.domains:
        mat_name = material.name
        mat_nuclides = material.get_nuclides()
        material_info[mat_name] = {
            'id': material.id,
            'nuclides': mat_nuclides,
            'temperature': BASE_TEMPERATURE
        }

        print(f"       Processing {mat_name} ({len(mat_nuclides)} nuclides)...")

        for nuclide in mat_nuclides:
            # Skip if already added
            if nuclide in base_nuclides:
                continue

            # Extract XS for this nuclide
            xs_dict = extract_xs_from_mgxs_lib_for_nuclide(
                mgxs_lib, material, nuclide, TRANSPORT_REACTIONS
            )

            if not xs_dict:
                print(f"         [WARN] No XS data for {nuclide}")
                continue

            # Build XSdata at BOTH base and fuel temperatures so that
            # nuclides shared between base materials and fuel (e.g. O16, He4)
            # are available at both temperature points after library merging.
            try:
                xsdata = build_xsdata_for_nuclide(
                    nuclide, xs_dict,
                    [BASE_TEMPERATURE, DEFAULT_TEMPERATURE]
                )
                mg_lib.add_xsdata(xsdata)
                base_nuclides.append(nuclide)
            except Exception as e:
                print(f"         [WARN] Could not create XSdata for {nuclide}: {e}")
                continue

    print(f"       Total nuclides in base library: {len(base_nuclides)}")

    # Save MGXSLibrary
    print("[3/3] Saving base materials library...")
    mg_lib_file = OUTPUT_BASE_MATERIALS / "mgxs_base_materials.h5"
    mg_lib.export_to_hdf5(str(mg_lib_file))
    print(f"       Saved: {mg_lib_file}")

    # Save metadata
    metadata = {
        'temperature': BASE_TEMPERATURE,
        'n_energy_groups': N_ENERGY_GROUPS,
        'n_nuclides': len(base_nuclides),
        'nuclides': base_nuclides,
        'materials': material_info,
        'transport_reactions': TRANSPORT_REACTIONS,
    }

    metadata_file = OUTPUT_BASE_MATERIALS / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"       Saved: {metadata_file}")

    print("✓ Completed base materials processing")
    return True


def load_base_mgxs_library():
    """
    Load pre-computed MGXSLibrary for base materials.

    Returns:
        Path to the base materials HDF5 file
    """
    mg_lib_file = OUTPUT_BASE_MATERIALS / "mgxs_base_materials.h5"
    if not mg_lib_file.exists():
        raise FileNotFoundError(f"Base MGXSLibrary not found: {mg_lib_file}")
    return mg_lib_file


# ================== MAIN PROCESSING ==================

def process_burnup_step(bu_step, burnup, temperature):
    """
    Process a single burnup step: extract XS, build and save libraries.

    Parameters:
        bu_step: Burnup step index (1-NSTEPS-1), used to locate MGXS input data
        burnup: Actual burnup value [MWd/kgHM], used for output directory naming
        temperature: Temperature in K
    """
    bu_str = f"{burnup:.3f}"
    print(f"\n{'='*60}")
    print(f"Processing BU step {bu_step} (BU={burnup:.3f} MWd/kgHM) at T={temperature}K")
    print(f"{'='*60}")

    # Create output directories (named by burnup value, not step index)
    dep_dir = OUTPUT_DEPLETION / f"BU_{bu_str}"
    trans_dir = OUTPUT_TRANSPORT / f"BU_{bu_str}"
    dep_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Load MGXS library and materials
    print("[1/5] Loading MGXS library...")
    try:
        mgxs_lib = load_mgxs_library(bu_step, temperature)
        fuel_material = load_materials_from_run(bu_step, temperature)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return False

    # Get nuclides present in fuel material
    fuel_nuclides = fuel_material.get_nuclides()
    print(f"       Fuel contains {len(fuel_nuclides)} nuclides")

    # Extract XS for all nuclides and reactions
    print("[2/5] Extracting cross sections...")
    all_xs = extract_all_xs(mgxs_lib, fuel_material, fuel_nuclides, ALL_REACTIONS)
    print(f"       Extracted XS for {len(all_xs)} nuclides")

    # Build MicroXS for depletion
    print("[3/5] Building MicroXS for depletion...")
    micro_xs = build_microxs(all_xs)

    # Save MicroXS
    micro_xs_file = dep_dir / f"MicroXS_BU_{bu_str}.h5"
    micro_xs.to_hdf5(micro_xs_file)
    print(f"       Saved: {micro_xs_file}")

    # Build MGXSLibrary for transport
    print("[4/5] Building MGXSLibrary for transport...")
    mg_lib = build_mgxs_library(all_xs, temperature)

    # Save MGXSLibrary
    mg_lib_file = trans_dir / f"mgxs_transport_BU_{bu_str}.h5"
    mg_lib.export_to_hdf5(str(mg_lib_file))
    print(f"       Saved: {mg_lib_file}")

    # Save metadata
    print("[5/5] Saving metadata...")
    metadata = {
        'bu_step': bu_step,
        'burnup_MWdkg': burnup,
        'temperature': temperature,
        'n_energy_groups': N_ENERGY_GROUPS,
        'n_nuclides': len(all_xs),
        'nuclides': list(all_xs.keys()),
        'depletion_reactions': DEPLETION_REACTIONS,
        'transport_reactions': TRANSPORT_REACTIONS,
    }

    metadata_file = dep_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    metadata_file = trans_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Completed BU step {bu_step} (BU={burnup:.3f} MWd/kgHM)")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare explicit-FP XS libraries for depletion and transport")
    parser.add_argument('--steps', type=int, nargs='+', default=list(range(0, NSTEPS)),
                        help=f'Burnup steps to process (default: 0-{NSTEPS-1}, where 0=fresh fuel, 1-{NSTEPS-1}=after depletion)')
    parser.add_argument('--temperature', type=int, default=DEFAULT_TEMPERATURE,
                        help=f'Temperature in K (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--base-only', action='store_true',
                        help='Only process base materials (clad, gap, coolant)')
    parser.add_argument('--skip-base', action='store_true',
                        help='Skip base materials processing')
    args = parser.parse_args()

    print("="*70)
    print("EXPLICIT-FP XS LIBRARY PREPARATION FOR DEPLETION AND TRANSPORT")
    print("="*70)
    print(f"Output directories:")
    print(f"  Depletion: {OUTPUT_DEPLETION}")
    print(f"  Transport: {OUTPUT_TRANSPORT}")
    print(f"  Base:      {OUTPUT_BASE_MATERIALS}")
    print(f"Temperature: {args.temperature} K")
    print(f"Steps to process: {args.steps}")
    print("="*70)

    # Process base materials first (clad, gap, coolant)
    if not args.skip_base:
        process_base_materials()

    # If base-only mode, exit here
    if args.base_only:
        return

    # Create output directories
    OUTPUT_DEPLETION.mkdir(parents=True, exist_ok=True)
    OUTPUT_TRANSPORT.mkdir(parents=True, exist_ok=True)

    # Pre-compute burnup map (step_index → MWd/kgHM)
    burnup_map = calculate_burnup_points()
    print("\nBurnup schedule [MWd/kgHM]:")
    for step, bu in sorted(burnup_map.items()):
        print(f"  Step {step:2d} → BU = {bu:8.3f} MWd/kgHM")
    print()

    # Process each burnup step
    n_success = 0
    n_failed = 0

    for bu_step in args.steps:
        burnup = burnup_map.get(bu_step, 0.0)
        bu_str = f"{burnup:.3f}"
        try:
            # BU_0 is fresh fuel - only process transport (no depletion MicroXS)
            if bu_step == 0:
                print(f"\n{'='*60}")
                print(f"Processing BU step 0 (fresh fuel, BU=0.000 MWd/kgHM) at T={args.temperature}K - TRANSPORT ONLY")
                print(f"{'='*60}")

                # Create transport output directory named by burnup value
                trans_dir = OUTPUT_TRANSPORT / f"BU_{bu_str}"
                trans_dir.mkdir(parents=True, exist_ok=True)

                # Load MGXS library and materials for BU_0
                print("[1/4] Loading MGXS library for fresh fuel...")
                try:
                    mgxs_lib = load_mgxs_library(0, args.temperature)
                    fuel_material = load_materials_from_run(0, args.temperature)
                except FileNotFoundError as e:
                    print(f"  [ERROR] {e}")
                    n_failed += 1
                    continue

                fuel_nuclides = fuel_material.get_nuclides()
                print(f"       Fuel contains {len(fuel_nuclides)} nuclides")

                # Extract XS for all nuclides and reactions
                print("[2/4] Extracting cross sections...")
                all_xs = extract_all_xs(mgxs_lib, fuel_material, fuel_nuclides, ALL_REACTIONS)
                print(f"       Extracted XS for {len(all_xs)} nuclides")

                # Build MGXSLibrary for transport only (no MicroXS for BU_0)
                print("[3/4] Building MGXSLibrary for transport...")
                mg_lib = build_mgxs_library(all_xs, args.temperature)

                # Save MGXSLibrary (file name uses burnup value)
                mg_lib_file = trans_dir / f"mgxs_transport_BU_{bu_str}.h5"
                mg_lib.export_to_hdf5(str(mg_lib_file))
                print(f"       Saved: {mg_lib_file}")

                # Save metadata
                print("[4/4] Saving metadata...")
                metadata = {
                    'bu_step': 0,
                    'burnup_MWdkg': burnup,
                    'description': 'Fresh fuel - transport only (no depletion)',
                    'temperature': args.temperature,
                    'n_energy_groups': N_ENERGY_GROUPS,
                    'n_nuclides': len(all_xs),
                    'nuclides': list(all_xs.keys()),
                    'transport_reactions': TRANSPORT_REACTIONS,
                }

                metadata_file = trans_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"✓ Completed BU step 0 (transport only, BU=0.000 MWd/kgHM)")
                n_success += 1
            else:
                # Normal depletion steps
                success = process_burnup_step(bu_step, burnup, args.temperature)
                if success:
                    n_success += 1
                else:
                    n_failed += 1
        except Exception as e:
            print(f"[ERROR] Failed to process BU step {bu_step}: {e}")
            n_failed += 1
            continue

    # Write library_metadata.json based on successfully processed steps
    if n_success > 0:
        successful_burnups = sorted([
            burnup_map[s] for s in args.steps
            if burnup_map.get(s) is not None
            and (OUTPUT_TRANSPORT / f"BU_{burnup_map[s]:.3f}").exists()
        ])
        metadata = {
            "library_name": "LFP_Thesis Pin Cell MGXS Library (explicit FP)",
            "description": "Multi-group cross sections for SFR fuel depletion analysis with explicit fission products (no LFP lumping)",
            "energy_group_structure": "ECCO-33",
            "n_energy_groups": N_ENERGY_GROUPS,
            "burnup_points_MWdkg": successful_burnups,
            "burnup_steps_MWdkg": BU_STEPS,
            "reference_power_W": 201.1,
            "fuel_type": "MOX (Pu/U oxide)",
            "geometry": "Pin cell",
            "fuel_radius_cm": 0.357,
            "clad_outer_radius_cm": 0.4321,
            "fuel_volume_cm3": 0.4004,
            "temperature_fuel_K": args.temperature,
            "temperature_structural_K": BASE_TEMPERATURE,
            "library_structure": {
                "transport": "transport/BU_X.XXX/mgxs_transport_BU_X.XXX.h5",
                "depletion": "depletion/BU_X.XXX/MicroXS_BU_X.XXX.h5",
                "base_materials": "base/mgxs_base_materials.h5"
            },
            "notes": [
                "Burnup points are cumulative burnup [MWd/kgHM]",
                "Explicit fission products: no LFP lumping, no yield weighting",
                "Only successfully produced steps are listed",
                "BU_0.000 only has transport library (fresh fuel, no depletion)"
            ]
        }
        metadata_out = OUTPUT_BASE / "library_metadata.json"
        with open(metadata_out, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ library_metadata.json written: {metadata_out}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully processed: {n_success} steps")
    print(f"Failed: {n_failed} steps")
    print(f"\nOutput files:")
    print(f"  Base materials:         {OUTPUT_BASE_MATERIALS}/mgxs_base_materials.h5")
    print(f"  Depletion (MicroXS):    {OUTPUT_DEPLETION}/BU_*/MicroXS_BU_*.h5")
    print(f"  Transport (MGXSLibrary): {OUTPUT_TRANSPORT}/BU_*/mgxs_transport_BU_*.h5")
    print(f"  Metadata:               {OUTPUT_BASE}/library_metadata.json")
    print(f"\nNote: BU_0 only has transport libraries (no depletion - fresh fuel)")
    print("="*70)


# ================== HELPER FUNCTIONS FOR LOADING ==================

def load_microxs(burnup):
    """
    Load pre-computed MicroXS for a burnup value [MWd/kgHM].

    Usage:
        from AllFP_XS_Lib import load_microxs
        micro_xs = load_microxs(0.05)  # BU = 0.05 MWd/kgHM
    """
    bu_str = f"{burnup:.3f}"
    micro_xs_file = OUTPUT_DEPLETION / f"BU_{bu_str}" / f"MicroXS_BU_{bu_str}.h5"
    if not micro_xs_file.exists():
        raise FileNotFoundError(f"MicroXS not found: {micro_xs_file}")

    return openmc.deplete.MicroXS.from_hdf5(micro_xs_file)


def load_mgxs_transport_library(burnup):
    """
    Load pre-computed MGXSLibrary for transport.

    Usage:
        from AllFP_XS_Lib import load_mgxs_transport_library
        mg_lib_path = load_mgxs_transport_library(0.05)
        materials.cross_sections = str(mg_lib_path)
    """
    bu_str = f"{burnup:.3f}"
    mg_lib_file = OUTPUT_TRANSPORT / f"BU_{bu_str}" / f"mgxs_transport_BU_{bu_str}.h5"
    if not mg_lib_file.exists():
        raise FileNotFoundError(f"MGXSLibrary not found: {mg_lib_file}")

    return mg_lib_file  # Return path, as MGXSLibrary is loaded via materials.cross_sections


def get_burnup_schedule():
    """
    Return the burnup schedule as a list of cumulative burnup points [MWd/kgHM].
    Useful for setting up MGXSLibraryManager or CoupledDepletionDriver.
    """
    return list(calculate_burnup_points().values())


if __name__ == "__main__":
    main()
