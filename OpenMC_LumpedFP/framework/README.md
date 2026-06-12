# MGXS Library and Coupled Depletion Framework

Modular Python framework for managing burnup-parametrized MGXS libraries and running coupled multi-group transport-depletion simulations with OpenMC.

## Overview

This framework provides two main classes:

1. **`MGXSLibraryManager`**: Manages MGXS/MicroXS libraries parametrized by burnup [MWd/kgHM]
2. **`CoupledDepletionDriver`**: Orchestrates coupled transport-depletion simulations with proper HM normalization

## Key Features

### MGXSLibraryManager

- ✅ **Burnup-based indexing**: Libraries organized by [MWd/kgHM], not time → portable across different power levels
- ✅ **Linear interpolation (default)**: Element-wise blending of MGXS scalar
  XS, `chi` (with renormalization), and `scatter_data` (with `multiplicity`
  treated as the ratio `nu_s / sigma_s`, consistent with OpenMC's storage
  convention)
- ✅ **Independent depletion grid**: Depletion MicroXS files are auto-detected;
  if a transport-grid point has no depletion file (e.g. `BU=0`), the first
  available depletion point is used as a surrogate
- ✅ **Dual library handling**: Separate transport (MGXS) and depletion (MicroXS) libraries
- ✅ **Nuclide filtering against the actual produced library**: matches the
  union of bracketing burnup points rather than the nearest neighbour
- ✅ **Library merging**: Combines fuel and structural material libraries with temperature handling
- ✅ **Chi=0 sanitization**: Prevents NaN propagation from null fission spectra
- ✅ **MicroXS alignment checks**: Verifies that the two MicroXS endpoints
  share the same nuclide and reaction ordering before blending

### CoupledDepletionDriver

- ✅ **Correct HM normalization**: Calculates and preserves initial heavy metal mass (Z ≥ 90)
- ✅ **Burnup tracking**: Maintains cumulative burnup [MWd/kgHM] throughout simulation
- ✅ **Operator override**: Ensures `operator.heavy_metal` uses initial value, not current
- ✅ **Automatic library loading**: Calls MGXSLibraryManager for burnup-appropriate cross sections
- ✅ **Clean interface**: Builder pattern for geometry/settings/tallies
- ✅ **Results tracking**: JSON output with keff, flux, burnup at each step

## Directory Structure

```
library/
├── __init__.py                       # Package initialization
├── mgxs_library_manager.py           # MGXSLibraryManager class
├── coupled_depletion_driver.py       # CoupledDepletionDriver class
├── create_library_metadata.py        # Script to create metadata file
└── README.md                         # This file

MGXS_Library/                         # Library directory
├── library_metadata.json             # Required metadata file
├── transport/
│   ├── BU_0.000/
│   │   └── mgxs_transport_BU_0.000.h5
│   ├── BU_0.055/
│   │   └── mgxs_transport_BU_0.055.h5
│   └── ...
├── depletion/
│   ├── BU_0.000/
│   │   └── MicroXS_BU_0.000.h5
│   ├── BU_0.055/
│   │   └── MicroXS_BU_0.055.h5
│   └── ...
└── base/
    └── mgxs_base_materials.h5        # Structural materials (clad, coolant, etc.)
```

**IMPORTANT**: Directory names use **dot notation** with 3 decimal places (e.g., `BU_0.055`, `BU_1.370`). If your existing libraries use different naming (e.g., `BU_1`, `BU_2` as step indices), you must rename them to use actual burnup values in MWd/kgHM.

## Installation

No installation needed - just ensure the `library/` directory is in your Python path:

```python
import sys
sys.path.insert(0, '/path/to/OpenMC_33G/library')

from mgxs_library_manager import MGXSLibraryManager
from coupled_depletion_driver import CoupledDepletionDriver
```

## Usage

### 1. Create library metadata (one time)

```bash
cd /data/user/pati_f/OpenMC_33G/library
python create_library_metadata.py
```

This creates `MGXS_Library/library_metadata.json` with:
- Burnup points [MWd/kgHM]
- Energy group structure
- Library paths and organization

### 2. Use in your simulation script

```python
from pathlib import Path
import openmc
from mgxs_library_manager import MGXSLibraryManager
from coupled_depletion_driver import CoupledDepletionDriver

# Initialize library manager
library_manager = MGXSLibraryManager(
    library_base_path="/path/to/MGXS_Library",
    metadata_file="library_metadata.json"  # optional, auto-detected
)

# Create geometry/settings/tallies builder functions
def create_geometry(materials):
    # Your geometry creation code
    return geometry

def create_settings():
    # Your settings creation code
    return settings

def create_tallies(fuel_material):
    # Your tallies creation code
    return tallies

# Initialize depletion driver
driver = CoupledDepletionDriver(
    library_manager=library_manager,
    geometry_builder=create_geometry,
    settings_builder=create_settings,
    tallies_builder=create_tallies,
    chain_file="/path/to/chain.xml",
    power=335.0,  # W
    output_dir="/path/to/output"
)

# Set up initial materials
fuel = create_fresh_fuel()
structural = [gap, clad, coolant]
driver.setup_initial_materials(fuel, structural)

# Run simulation with burnup targets (recommended)
burnup_targets = [0.055, 0.329, 1.424, 6.355, 12.711, ...]  # MWd/kgHM
results = driver.run_burnup_schedule(burnup_targets)

# Alternative: specify timesteps directly
# timesteps_days = [0.055, 0.274, 1.096, ...]  # days
# results = driver.run_coupled_simulation(timesteps_days)
```

See `example_usage_classes.py` for a complete working example.

## Key Concepts

### Burnup Normalization

The framework ensures consistent burnup calculation:

```
BU [MWd/kg] = (P [W] × t [days]) / (HM_initial [kg] × 10^6)
```

**Critical**: `HM_initial` is calculated **once** at initialization and **never updated**, matching standard reactor physics convention and OpenMC's internal behavior.

### Library Interpolation

When requesting cross sections at burnup `BU_target`:

1. Find bracketing burnup points: `BU_low` and `BU_high`
2. Calculate interpolation weight: `α = (BU_target - BU_low) / (BU_high - BU_low)`
3. Interpolate:
   - **Transport libraries (MGXS)**: linear interpolation by default
     (`interpolation='linear'`); legacy nearest-neighbour available via
     `interpolation='nearest'`. The linear path operates element-wise on:
     - the scalar arrays `total`, `absorption`, `fission`, `nu-fission`;
     - the `chi` spectrum, with renormalization to unit sum after the blend;
     - the `scatter_data` group. **Convention**: OpenMC stores `nu_scatter` in
       the dataset named `scatter_matrix` whenever `multiplicity_matrix` is
       set (see `openmc/mgxs/library.py:get_xsdata`). The interpolator
       therefore recovers `sigma_s = nu_scatter / mult` at each library point,
       linearly blends `nu_scatter` and `sigma_s` separately (both are
       physically additive), and reconstructs `mult_new = nu_scatter_interp /
       sigma_s_interp`. This preserves reaction balance and is consistent
       with OpenMC's internal use of the multiplicity matrix.
     - Sparsity patterns (`g_min`, `g_max`) are honoured via a fast path when
       identical between the two library points, with a dense fallback that
       unions the patterns otherwise.
   - **Depletion libraries (MicroXS)**: linear interpolation
     - `XS_interp = (1-α) × XS_low + α × XS_high` element-wise
     - Order of nuclides and reactions is verified between the two endpoints
       to prevent silent mis-alignment.

**Extrapolation / clamping**: If `BU_target` is outside `[BU_min, BU_max]`,
the boundary value is used with a notice. The depletion grid may be a strict
subset of the transport grid (typically missing `BU=0`, where LFP MicroXS
cannot be reconstructed); in that case the first available depletion point
is used as a surrogate.

**Nuclide filtering with interpolation**: the fuel material is filtered
against the nuclides actually present in the produced (interpolated +
merged-with-base) transport HDF5 file, not against the nearest-neighbour
library. This avoids dropping nuclides that exist at one of the two
bracketing burnups but not the other.

### Operator Heavy Metal Override

In custom depletion loops where `IndependentOperator` is recreated each step:

```python
operator = openmc.deplete.IndependentOperator(...)
operator.heavy_metal = HM_initial_grams  # Override with initial value (defensive)
```

**Note**: With `timestep_units='d'` and `power=...` (the default in this framework), this override is not strictly necessary since OpenMC doesn't use `heavy_metal` in the calculation. However, it's included as a defensive measure to ensure consistency if the operator is reused or if you switch to `timestep_units='MWd/kg'` or `power_density` modes.


## Contact

For questions or issues, contact me at federicopati@gmail.com
