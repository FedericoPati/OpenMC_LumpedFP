"""
Create library_metadata.json for MGXS Library

This script creates the metadata file needed by MGXSLibraryManager.
It should be run once to initialize a new library or update an existing one.
"""

import json
from pathlib import Path


def create_library_metadata(library_base_path, output_path=None):
    """
    Create library metadata file.
    
    Parameters
    ----------
    library_base_path : Path
        Base directory of MGXS library
    output_path : Path, optional
        Where to save metadata. Default: library_base_path/library_metadata.json
    """
    library_base_path = Path(library_base_path)
    
    if output_path is None:
        output_path = library_base_path / "library_metadata.json"
    
    # Extract burnup points from directory names
    # Assuming structure: transport/BU_X/ and depletion/BU_X/
    transport_dir = library_base_path / "transport"
    
    burnup_points = []
    if transport_dir.exists():
        for bu_dir in sorted(transport_dir.iterdir(), key=lambda d: float(d.name.replace("BU_", "")) if d.is_dir() and d.name.startswith("BU_") else -1):
            if bu_dir.is_dir() and bu_dir.name.startswith("BU_"):
                # Extract burnup value from BU_X.XXX (dot notation)
                bu_str = bu_dir.name.replace("BU_", "")
                try:
                    bu_val = float(bu_str)
                    burnup_points.append(bu_val)
                except ValueError:
                    print(f"Warning: Could not parse burnup from {bu_dir.name}")
    
    if not burnup_points:
        raise FileNotFoundError(
            f"No BU_* directories found under {transport_dir}.\n"
            "Run LFP_XS_library.py first to produce the library, then re-run this script."
        )
    
    metadata = {
        "library_name": "LFP_Thesis Pin Cell MGXS Library",
        "description": "Multi-group cross sections for SFR fuel depletion analysis with LFP lumped fission products",
        "energy_group_structure": "ECCO-33",
        "n_energy_groups": 33,
        "burnup_points_MWdkg": burnup_points,
        "reference_power_W": 201.1,
        "fuel_type": "MOX (Pu/U oxide)",
        "geometry": "Pin cell",
        "fuel_radius_cm": 0.357,
        "clad_outer_radius_cm": 0.4321,
        "fuel_volume_cm3": 0.4004,
        "temperature_fuel_K": 1500.0,
        "temperature_structural_K": 673.0,
        "library_structure": {
            "transport": "transport/BU_X.XXX/mgxs_transport_BU_X.XXX.h5",
            "depletion": "depletion/BU_X.XXX/MicroXS_BU_X.XXX.h5",
            "base_materials": "base/mgxs_base_materials.h5"
        },
        "notes": [
            "Burnup points are cumulative burnup [MWd/kgHM], read from directory names",
            "Transport libraries contain MGXS for fuel at each burnup",
            "Depletion libraries contain microscopic XS for depletion calc",
            "Base library contains structural materials (clad, coolant, gap)"
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata file: {output_path}")
    print(f"Burnup points [MWd/kg]: {burnup_points}")
    print(f"Total library points: {len(burnup_points)}")
    
    return metadata


if __name__ == "__main__":
    # Adjust this path to your library location
    library_base = Path("/data/user/pati_f/LFP_Thesis/library/MGXS_Library")
    
    if not library_base.exists():
        print(f"Error: Library base path does not exist: {library_base}")
        exit(1)
    
    metadata = create_library_metadata(library_base)
    
    print("\nMetadata created successfully!")
    print("You can now use MGXSLibraryManager with this library.")
