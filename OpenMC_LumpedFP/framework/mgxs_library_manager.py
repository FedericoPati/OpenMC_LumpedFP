"""
MGXS Library Manager

Manages multi-group cross-section libraries parametrized by burnup [MWd/kgHM].
Provides interpolation between library points and handles both transport (MGXS)
and depletion (MicroXS) data.

Key features:
- Burnup-based indexing (portable across different power levels)
- Linear interpolation between library points
- Separate handling of transport and depletion XS
- Automatic nuclide filtering
- Merging with base structural material libraries
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py
import openmc
import openmc.deplete


class MGXSLibraryManager:
    """
    Manager for burnup-dependent MGXS libraries.
    
    Parameters
    ----------
    library_base_path : str or Path
        Base directory containing MGXS libraries organized by burnup
    metadata_file : str or Path, optional
        Path to library metadata JSON file. If None, looks for 
        'library_metadata.json' in library_base_path.
    
    Attributes
    ----------
    library_path : Path
        Base path to library directory
    metadata : dict
        Library metadata (burnup points, nuclides, energy groups, etc.)
    burnup_points : np.ndarray
        Available burnup points [MWd/kgHM]
    """
    
    def __init__(self, library_base_path: str | Path,
                 metadata_file: Optional[str | Path] = None):
        self.library_path = Path(library_base_path)

        if metadata_file is None:
            metadata_file = self.library_path / "library_metadata.json"

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.burnup_points = np.array(self.metadata['burnup_points_MWdkg'])
        self.n_groups = self.metadata.get('n_energy_groups', 33)

        # Depletion library grid may be a subset of the transport one: BU=0 in
        # particular is typically present only as a transport library because
        # the LFP MicroXS need depletion-derived auxiliary concentrations,
        # which do not exist for fresh fuel. We scan the depletion/ directory
        # to discover which burnup points actually have a MicroXS file.
        self.depletion_burnup_points = self._scan_depletion_burnups()

        # Persistent disk cache for merged transport libraries, keyed by BU.
        # Avoids repeating the HDF5 copy+merge on re-runs or restarts.
        self._cache_dir = self.library_path / "_transport_cache"
        # In-memory map: cache_key → Path (populated during the current run)
        self._transport_cache: Dict[str, Path] = {}
        # Cached result of the pre-merged check (None = not yet determined)
        self._premerged: Optional[bool] = None

        print(f"MGXSLibraryManager initialized")
        print(f"  Library path: {self.library_path}")
        print(f"  Transport burnup points [MWd/kg]: {self.burnup_points}")
        if not np.array_equal(self.burnup_points, self.depletion_burnup_points):
            print(f"  Depletion burnup points [MWd/kg]: {self.depletion_burnup_points}")
            missing = sorted(set(np.round(self.burnup_points, 3))
                             - set(np.round(self.depletion_burnup_points, 3)))
            if missing:
                print(f"  Note: no MicroXS for BU = {missing} — will fall back to "
                      f"nearest available depletion point.")
        print(f"  Energy groups: {self.n_groups}")

    def _scan_depletion_burnups(self) -> np.ndarray:
        """Scan depletion/ directory for actually-present MicroXS files."""
        available = []
        for bu in self.burnup_points:
            if self._get_depletion_library_path(float(bu)).exists():
                available.append(float(bu))
        return np.array(sorted(available)) if available else self.burnup_points.copy()
    
    def _transport_premerged(self) -> bool:
        """Return True if the transport library files already embed base materials.

        Checks once (result cached) by looking for a representative structural
        nuclide (Na23) in the first available transport file.  If found, the
        libraries were built with merge_base_into_transport() and no runtime
        merge is needed.
        """
        if self._premerged is not None:
            return self._premerged

        probe_bu = float(self.burnup_points[0])
        probe_path = self._get_transport_library_path(probe_bu)
        if probe_path.exists():
            with h5py.File(probe_path, 'r') as f:
                self._premerged = "Na23" in f
        else:
            self._premerged = False

        state = "pre-merged" if self._premerged else "fuel-only (runtime merge needed)"
        print(f"  [MGXSLibraryManager] Transport libraries: {state}")
        return self._premerged

    def get_bracketing_burnups(self, target_burnup: float) -> Tuple[float, float, float]:
        """
        Find burnup points that bracket the target burnup on the transport grid.

        Returns
        -------
        bu_low, bu_high : float
            Bracketing burnup points [MWd/kgHM]
        alpha : float
            Interpolation weight (0 = use bu_low, 1 = use bu_high)
        """
        return self._bracket_on_grid(target_burnup, self.burnup_points,
                                     grid_name='transport')

    def _bracket_on_grid(self, target_burnup: float, grid: np.ndarray,
                         grid_name: str = '') -> Tuple[float, float, float]:
        """Generic bracketing on an arbitrary monotone-increasing grid of BU points."""
        if target_burnup <= grid[0]:
            if target_burnup < grid[0]:
                print(f"  Note: clamping to minimum {grid_name} burnup "
                      f"(target={target_burnup:.3f}, min={grid[0]:.3f} MWd/kg)")
            return float(grid[0]), float(grid[0]), 0.0

        if target_burnup >= grid[-1]:
            if target_burnup > grid[-1]:
                print(f"  Note: clamping to maximum {grid_name} burnup "
                      f"(target={target_burnup:.3f}, max={grid[-1]:.3f} MWd/kg)")
            return float(grid[-1]), float(grid[-1]), 0.0

        # Exact-match short circuit: avoids the spurious "interpolate between
        # bu_low and bu_high with alpha=1.0" branch when the target lies on
        # a grid point. This is essential when the libraries at consecutive
        # grid points carry different nuclide lists (e.g. AllFP MicroXS).
        matches = np.where(np.isclose(grid, target_burnup, rtol=1e-9, atol=1e-9))[0]
        if len(matches):
            bu_exact = float(grid[int(matches[0])])
            return bu_exact, bu_exact, 0.0

        idx = np.searchsorted(grid, target_burnup)
        bu_low = float(grid[idx - 1])
        bu_high = float(grid[idx])
        alpha = (target_burnup - bu_low) / (bu_high - bu_low)
        return bu_low, bu_high, alpha
    
    def _get_transport_library_path(self, burnup: float) -> Path:
        """Get path to transport MGXS library for given burnup."""
        # Directory structure: transport/BU_X.XXX/mgxs_transport_BU_X.XXX.h5
        # Using dot notation with 3 decimal places, e.g., BU_0.055
        bu_str = f"{burnup:.3f}"
        lib_dir = self.library_path / "transport" / f"BU_{bu_str}"
        lib_file = lib_dir / f"mgxs_transport_BU_{bu_str}.h5"
        return lib_file
    
    def _get_depletion_library_path(self, burnup: float) -> Path:
        """Get path to depletion MicroXS library for given burnup."""
        # Directory structure: depletion/BU_X.XXX/MicroXS_BU_X.XXX.h5
        # Using dot notation with 3 decimal places, e.g., BU_0.055
        bu_str = f"{burnup:.3f}"
        lib_dir = self.library_path / "depletion" / f"BU_{bu_str}"
        lib_file = lib_dir / f"MicroXS_BU_{bu_str}.h5"
        return lib_file
    
    # ------------------------------------------------------------------
    # Transport library cache helpers
    # ------------------------------------------------------------------

    def _transport_cache_key(self, bu_low: float, bu_high: float,
                             alpha: float, has_base: bool) -> str:
        """Stable string key for the merged transport library cache."""
        if alpha == 0.0:
            key = f"merged_BU_{bu_low:.3f}"
        else:
            key = f"interp_BU_{bu_low:.3f}_{bu_high:.3f}_a{alpha:.6f}"
        if has_base:
            key += "_wbase"
        return key

    def _cache_is_valid(self, cached_path: Path,
                        source_paths: List[Path]) -> bool:
        """Return True iff cached_path exists and is newer than all source files."""
        if not cached_path.exists():
            return False
        try:
            cached_mtime = cached_path.stat().st_mtime
            return all(cached_mtime > p.stat().st_mtime
                       for p in source_paths if p.exists())
        except OSError:
            return False

    # ------------------------------------------------------------------

    def get_transport_library(self, burnup: float,
                             base_library_path: Optional[Path] = None,
                             output_path: Optional[Path] = None,
                             interpolation: str = 'linear') -> Path:
        """
        Get transport MGXS library for target burnup, with interpolation if needed.

        Results are cached on disk under ``library_path/_transport_cache/`` so
        that repeated calls for the same burnup point (re-runs, restarts) skip
        the HDF5 copy-and-merge entirely.  The ``output_path`` argument is kept
        for API compatibility but is no longer used when a cache hit occurs.

        Parameters
        ----------
        burnup : float
            Target burnup [MWd/kgHM]
        base_library_path : Path, optional
            Path to base structural materials library to merge with fuel library
        output_path : Path, optional
            Ignored when the result is served from cache; kept for compatibility.
        interpolation : {'linear', 'nearest'}
            Strategy when the target burnup falls between library points.
            'linear' (default): element-wise linear interpolation of scalar XS
            and scatter matrices, with chi re-normalized to sum to 1.
            'nearest': pick the closer library point (legacy behaviour).

        Returns
        -------
        library_path : Path
            Path to MGXS library (merged if base_library provided)
        """
        bu_low, bu_high, alpha = self.get_bracketing_burnups(burnup)
        has_base = base_library_path is not None
        cache_key = self._transport_cache_key(bu_low, bu_high, alpha, has_base)

        # 1. In-memory cache hit (fastest path — same Python process)
        if cache_key in self._transport_cache:
            cached = self._transport_cache[cache_key]
            if cached.exists():
                print(f"  [cache] Transport library for BU={burnup:.3f} MWd/kg "
                      f"(memory hit: {cached.name})")
                return cached
            del self._transport_cache[cache_key]

        # 2. On-disk cache hit (survives across runs / restarts)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = self._cache_dir / f"{cache_key}.h5"

        source_paths: List[Path] = [self._get_transport_library_path(bu_low)]
        if alpha > 0.0:
            source_paths.append(self._get_transport_library_path(bu_high))
        if has_base:
            source_paths.append(Path(base_library_path))

        if self._cache_is_valid(cached_path, source_paths):
            print(f"  [cache] Transport library for BU={burnup:.3f} MWd/kg "
                  f"(disk hit: {cached_path.name})")
            self._transport_cache[cache_key] = cached_path
            return cached_path

        # 3. Cache miss — generate the fuel library
        if alpha == 0.0:
            fuel_lib = self._get_transport_library_path(bu_low)
        elif interpolation == 'nearest':
            nearest_bu = bu_low if alpha < 0.5 else bu_high
            fuel_lib = self._get_transport_library_path(nearest_bu)
            print(f"  Using nearest transport library at BU={nearest_bu:.3f} MWd/kg ")
            print(f"  (target={burnup:.3f}, interpolation weight={alpha:.3f})")
        elif interpolation == 'linear':
            lib_low_path = self._get_transport_library_path(bu_low)
            lib_high_path = self._get_transport_library_path(bu_high)
            if not lib_low_path.exists():
                raise FileNotFoundError(f"Transport library not found: {lib_low_path}")
            if not lib_high_path.exists():
                raise FileNotFoundError(f"Transport library not found: {lib_high_path}")

            # Write interpolated fuel lib into the cache dir (not the step dir)
            interp_key = (f"fuel_interp_BU_{bu_low:.3f}_{bu_high:.3f}"
                          f"_a{alpha:.6f}")
            fuel_lib = self._cache_dir / f"{interp_key}.h5"
            self._interpolate_mgxs_libraries(lib_low_path, lib_high_path,
                                             alpha, fuel_lib)
            print(f"  Linearly interpolated transport MGXS between "
                  f"BU={bu_low:.3f} and {bu_high:.3f} MWd/kg (alpha={alpha:.3f})")
        else:
            raise ValueError(
                f"Unknown interpolation strategy: '{interpolation}'. "
                f"Use 'linear' or 'nearest'."
            )

        if not fuel_lib.exists():
            raise FileNotFoundError(f"Transport library not found: {fuel_lib}")

        # 4. Merge with base library into the cache dir (or return fuel directly)
        if not has_base:
            self._transport_cache[cache_key] = fuel_lib
            return fuel_lib

        self._merge_libraries(fuel_lib, base_library_path, cached_path)
        self._transport_cache[cache_key] = cached_path
        return cached_path
    
    def get_depletion_library(self, burnup: float) -> openmc.deplete.MicroXS:
        """
        Get depletion MicroXS for target burnup, with interpolation if needed.

        Uses the depletion-specific burnup grid (which may be a subset of the
        transport grid — typically missing BU=0). If the requested burnup falls
        below the first available depletion point, the first available point is
        used as a surrogate (canonical workaround for the BU=0 case, where
        LFP MicroXS cannot be reconstructed because the auxiliary depletion
        concentrations do not yet exist).

        Parameters
        ----------
        burnup : float
            Target burnup [MWd/kgHM]

        Returns
        -------
        micro_xs : openmc.deplete.MicroXS
            Microscopic cross sections for depletion
        """
        bu_low, bu_high, alpha = self._bracket_on_grid(
            burnup, self.depletion_burnup_points, grid_name='depletion')

        if alpha == 0.0:
            micro_path = self._get_depletion_library_path(bu_low)
            if not micro_path.exists():
                raise FileNotFoundError(f"Depletion library not found: {micro_path}")
            if burnup < bu_low:
                print(f"  Using first available depletion library at BU={bu_low:.3f} "
                      f"MWd/kg as surrogate for BU={burnup:.3f}")
            return openmc.deplete.MicroXS.from_hdf5(micro_path)

        micro_low_path = self._get_depletion_library_path(bu_low)
        micro_high_path = self._get_depletion_library_path(bu_high)

        if not micro_low_path.exists():
            raise FileNotFoundError(f"Depletion library not found: {micro_low_path}")
        if not micro_high_path.exists():
            raise FileNotFoundError(f"Depletion library not found: {micro_high_path}")

        micro_low = openmc.deplete.MicroXS.from_hdf5(micro_low_path)
        micro_high = openmc.deplete.MicroXS.from_hdf5(micro_high_path)

        micro_interp = self._interpolate_microxs(micro_low, micro_high, alpha)

        print(f"  Interpolated MicroXS between BU={bu_low:.3f} and {bu_high:.3f} "
              f"MWd/kg (alpha={alpha:.3f})")

        return micro_interp
    
    def _interpolate_microxs(self, micro_low: openmc.deplete.MicroXS,
                            micro_high: openmc.deplete.MicroXS,
                            alpha: float) -> openmc.deplete.MicroXS:
        """
        Linearly interpolate between two MicroXS objects.
        
        Parameters
        ----------
        micro_low : openmc.deplete.MicroXS
            Lower burnup MicroXS
        micro_high : openmc.deplete.MicroXS
            Higher burnup MicroXS
        alpha : float
            Interpolation weight (0 to 1)
        
        Returns
        -------
        micro_interp : openmc.deplete.MicroXS
            Interpolated MicroXS
        """
        # Reactions must match exactly (same physics layout); nuclides may differ
        # between burnup points (typical of AllFP-style libraries that grow as
        # more fission products appear). In that case we take the union of the
        # nuclide lists and zero-fill the missing entries before blending.
        if list(micro_low.reactions) != list(micro_high.reactions):
            raise ValueError(
                "MicroXS reaction ordering differs between burnup points. "
                "Cannot interpolate element-wise."
            )

        n_groups = micro_low.data.shape[2]
        reactions = list(micro_low.reactions)
        nuc_low = list(micro_low.nuclides)
        nuc_high = list(micro_high.nuclides)

        if nuc_low == nuc_high:
            nuclides = nuc_low
            data_interp = (1 - alpha) * micro_low.data + alpha * micro_high.data
        else:
            # Union of nuclides. For nuclides present at BOTH burnups we do
            # the usual linear blend. For nuclides present at only ONE side
            # we assume σ does not change between the two adjacent grid
            # points and reuse the available value (no blend with a spurious
            # zero). Rationale: σ is a property of the nuclide and the local
            # flux spectrum, both of which vary smoothly in BU; the nuclide
            # is "missing" at one side only because its concentration was
            # negligible there during library generation, not because σ
            # doesn't physically exist.
            seen = set(nuc_low)
            nuclides = list(nuc_low) + [n for n in nuc_high if n not in seen]
            n_rxn = len(reactions)
            data_interp = np.zeros((len(nuclides), n_rxn, n_groups), dtype=float)
            idx_low  = {n: i for i, n in enumerate(nuc_low)}
            idx_high = {n: i for i, n in enumerate(nuc_high)}
            n_only_low = n_only_high = 0
            for i, n in enumerate(nuclides):
                in_low = n in idx_low
                in_high = n in idx_high
                if in_low and in_high:
                    data_interp[i] = ((1 - alpha) * micro_low.data[idx_low[n]]
                                      + alpha * micro_high.data[idx_high[n]])
                elif in_low:
                    data_interp[i] = micro_low.data[idx_low[n]]
                    n_only_low += 1
                else:
                    data_interp[i] = micro_high.data[idx_high[n]]
                    n_only_high += 1
            print(f"  MicroXS interp: union {len(nuc_low)} ∪ {len(nuc_high)} → "
                  f"{len(nuclides)} (only-low={n_only_low}, only-high={n_only_high})")
        
        # Create new MicroXS object with correct signature
        # Signature: MicroXS(data, nuclides, reactions)
        micro_interp = openmc.deplete.MicroXS(
            data=data_interp,
            nuclides=nuclides,
            reactions=reactions
        )
        
        return micro_interp
    
    def _interpolate_mgxs_libraries(self, lib_low: Path, lib_high: Path,
                                    alpha: float, output: Path):
        """
        Element-wise linear interpolation between two MGXS HDF5 libraries.

        Strategy per (nuclide, temperature):
          - total / absorption / fission / nu-fission : straight linear blend
          - chi : linear blend + re-normalization to unit sum
          - scatter_data : fast path if g_min/g_max patterns coincide
            (almost always true for the same nuclide at the same temperature),
            otherwise expand to dense GxG, blend, re-compress on the union pattern.

        Nuclides present at only one of the two burnups are copied as-is.
        Top-level attributes (filetype, group structure, energy_groups) and
        the per-nuclide kTs subgroup are taken from the lower-burnup file.
        """
        with h5py.File(lib_low, 'r') as fl, h5py.File(lib_high, 'r') as fh, \
             h5py.File(output, 'w') as out:

            for key, val in fl.attrs.items():
                out.attrs[key] = val

            nuc_low = set(fl.keys())
            nuc_high = set(fh.keys())
            common = nuc_low & nuc_high

            for nuc in sorted(common):
                self._interpolate_nuclide(fl[nuc], fh[nuc], alpha, out, nuc)

            for nuc in sorted(nuc_low - common):
                fl.copy(nuc, out)
                print(f"    Note: nuclide {nuc} only at BU_low, copied as-is")
            for nuc in sorted(nuc_high - common):
                fh.copy(nuc, out)
                print(f"    Note: nuclide {nuc} only at BU_high, copied as-is")

    def _interpolate_nuclide(self, grp_low, grp_high, alpha: float,
                              out_file, nuc_name: str):
        """Interpolate one nuclide group between two MGXS files."""
        out_grp = out_file.create_group(nuc_name)
        for key, val in grp_low.attrs.items():
            out_grp.attrs[key] = val

        temps_low = {k for k in grp_low.keys() if k != 'kTs'}
        temps_high = {k for k in grp_high.keys() if k != 'kTs'}
        common_temps = temps_low & temps_high

        for temp in sorted(common_temps):
            self._interpolate_temp_group(
                grp_low[temp], grp_high[temp], alpha,
                out_grp.create_group(temp))

        for temp in sorted(temps_low - common_temps):
            grp_low.copy(temp, out_grp)
        for temp in sorted(temps_high - common_temps):
            grp_high.copy(temp, out_grp)

        if 'kTs' in grp_low:
            grp_low.copy('kTs', out_grp)
        elif 'kTs' in grp_high:
            grp_high.copy('kTs', out_grp)

    def _interpolate_temp_group(self, tgrp_low, tgrp_high, alpha: float, out_grp):
        """Interpolate the (G,) datasets and the scatter_data subgroup."""
        for ds_name in ('total', 'absorption', 'fission', 'nu-fission'):
            in_low = ds_name in tgrp_low
            in_high = ds_name in tgrp_high
            if in_low and in_high:
                blended = (1.0 - alpha) * tgrp_low[ds_name][...] \
                          + alpha * tgrp_high[ds_name][...]
                # Guard against tiny negatives from floating-point noise
                np.maximum(blended, 0.0, out=blended)
                out_grp.create_dataset(ds_name, data=blended)
            elif in_low:
                out_grp.create_dataset(ds_name, data=tgrp_low[ds_name][...])
            elif in_high:
                out_grp.create_dataset(ds_name, data=tgrp_high[ds_name][...])

        # chi: blend and renormalize
        if 'chi' in tgrp_low and 'chi' in tgrp_high:
            chi = (1.0 - alpha) * tgrp_low['chi'][...] + alpha * tgrp_high['chi'][...]
            np.maximum(chi, 0.0, out=chi)
            s = chi.sum()
            if s > 0:
                chi /= s
            out_grp.create_dataset('chi', data=chi)
        elif 'chi' in tgrp_low:
            out_grp.create_dataset('chi', data=tgrp_low['chi'][...])
        elif 'chi' in tgrp_high:
            out_grp.create_dataset('chi', data=tgrp_high['chi'][...])

        # scatter_data
        if 'scatter_data' in tgrp_low and 'scatter_data' in tgrp_high:
            out_sd = out_grp.create_group('scatter_data')
            self._interpolate_scatter(tgrp_low['scatter_data'],
                                       tgrp_high['scatter_data'],
                                       alpha, out_sd)
        elif 'scatter_data' in tgrp_low:
            tgrp_low.copy('scatter_data', out_grp)
        elif 'scatter_data' in tgrp_high:
            tgrp_high.copy('scatter_data', out_grp)

    def _interpolate_scatter(self, sd_low, sd_high, alpha: float, out_sd):
        """
        Interpolate compressed scatter_matrix / multiplicity_matrix.

        OpenMC convention (Library.get_xsdata() at openmc/mgxs/library.py:1199-
        1207): when a multiplicity matrix is stored, the dataset named
        'scatter_matrix' in the HDF5 actually holds the nu-scatter (production)
        matrix, while 'multiplicity_matrix' holds the ratio nu_s / sigma_s.

        Both nu_s and sigma_s are physically additive cross sections and are
        the natural targets of linear interpolation. The multiplicity, being a
        ratio, is derived from the two.

        Procedure:
            1. nu_s_low  = scatter_matrix_low,    nu_s_high  = scatter_matrix_high
            2. sigma_s_low = nu_s_low / mult_low, sigma_s_high = nu_s_high / mult_high
            3. nu_s_interp    = (1-a) nu_s_low    + a nu_s_high
            4. sigma_s_interp = (1-a) sigma_s_low + a sigma_s_high
            5. mult_interp    = nu_s_interp / sigma_s_interp
            6. scatter_matrix_out = nu_s_interp

        When 'multiplicity_matrix' is not present (e.g. CladCool-style
        libraries), OpenMC implicitly assumes mult=1, so scatter_matrix is
        treated as both nu_s and sigma_s. Linear blend of scatter_matrix is
        then the natural choice.

        Two structural paths:
        - Fast: identical g_min/g_max → blend the flat arrays directly.
        - Slow: expand to dense G×G, blend, re-compress on the union pattern.
        """
        g_min_low = sd_low['g_min'][...]
        g_max_low = sd_low['g_max'][...]
        g_min_high = sd_high['g_min'][...]
        g_max_high = sd_high['g_max'][...]

        ns_low = sd_low['scatter_matrix'][...]   # OpenMC convention: this is nu_s
        ns_high = sd_high['scatter_matrix'][...]

        has_mm = 'multiplicity_matrix' in sd_low and 'multiplicity_matrix' in sd_high
        if has_mm:
            mm_low = sd_low['multiplicity_matrix'][...]
            mm_high = sd_high['multiplicity_matrix'][...]

        same_pattern = (np.array_equal(g_min_low, g_min_high)
                        and np.array_equal(g_max_low, g_max_high)
                        and ns_low.shape == ns_high.shape)

        if same_pattern:
            ns_interp = (1.0 - alpha) * ns_low + alpha * ns_high
            np.maximum(ns_interp, 0.0, out=ns_interp)
            out_sd.create_dataset('g_min', data=g_min_low)
            out_sd.create_dataset('g_max', data=g_max_low)
            out_sd.create_dataset('scatter_matrix', data=ns_interp)
            if has_mm:
                # Recover sigma_s at each end as nu_s / mult (safe-divide)
                sigma_low = np.divide(ns_low, mm_low,
                                       out=np.zeros_like(ns_low),
                                       where=mm_low > 0)
                sigma_high = np.divide(ns_high, mm_high,
                                        out=np.zeros_like(ns_high),
                                        where=mm_high > 0)
                sigma_interp = (1.0 - alpha) * sigma_low + alpha * sigma_high
                # New multiplicity: where sigma_s_interp > 0 use ratio, else 1
                mm_new = np.divide(ns_interp, sigma_interp,
                                    out=np.ones_like(ns_interp),
                                    where=sigma_interp > 0)
                out_sd.create_dataset('multiplicity_matrix', data=mm_new)
            return

        G = len(g_min_low)
        ns_dense_low = self._scatter_to_dense(ns_low, g_min_low, g_max_low, G)
        ns_dense_high = self._scatter_to_dense(ns_high, g_min_high, g_max_high, G)

        # Band-presence masks. An entry is "really present" only if its (g, g')
        # lies within the original g_min/g_max envelope of that library —
        # entries outside the band are treated as physically absent rather
        # than as zeros to be averaged. Option-B blend: where both bands
        # cover the entry, do (1-α)·low + α·high; where only one covers it,
        # reuse the available value; where neither covers it, leave 0.
        mask_low  = self._band_mask(g_min_low,  g_max_low,  G)
        mask_high = self._band_mask(g_min_high, g_max_high, G)
        both      = mask_low & mask_high
        only_low  = mask_low & ~mask_high
        only_high = ~mask_low & mask_high

        ns_dense = np.zeros_like(ns_dense_low)
        ns_dense[both]      = (1.0 - alpha) * ns_dense_low[both] + alpha * ns_dense_high[both]
        ns_dense[only_low]  = ns_dense_low[only_low]
        ns_dense[only_high] = ns_dense_high[only_high]
        np.maximum(ns_dense, 0.0, out=ns_dense)

        g_min_new = np.minimum(g_min_low, g_min_high).astype(np.int64)
        g_max_new = np.maximum(g_max_low, g_max_high).astype(np.int64)

        if has_mm:
            mm_dense_low = self._scatter_to_dense(mm_low, g_min_low, g_max_low, G)
            mm_dense_high = self._scatter_to_dense(mm_high, g_min_high, g_max_high, G)
            sigma_dense_low = np.divide(ns_dense_low, mm_dense_low,
                                         out=np.zeros_like(ns_dense_low),
                                         where=mm_dense_low > 0)
            sigma_dense_high = np.divide(ns_dense_high, mm_dense_high,
                                          out=np.zeros_like(ns_dense_high),
                                          where=mm_dense_high > 0)
            # Same Option-B treatment on σ_s before recovering multiplicity.
            sigma_dense = np.zeros_like(sigma_dense_low)
            sigma_dense[both]      = ((1.0 - alpha) * sigma_dense_low[both]
                                      + alpha * sigma_dense_high[both])
            sigma_dense[only_low]  = sigma_dense_low[only_low]
            sigma_dense[only_high] = sigma_dense_high[only_high]
            mm_dense_new = np.divide(ns_dense, sigma_dense,
                                      out=np.ones_like(ns_dense),
                                      where=sigma_dense > 0)
            ns_flat, mm_flat = self._scatter_to_compressed(
                ns_dense, mm_dense_new, g_min_new, g_max_new)
        else:
            ns_flat = self._scatter_to_compressed_single(
                ns_dense, g_min_new, g_max_new)

        out_sd.create_dataset('g_min', data=g_min_new)
        out_sd.create_dataset('g_max', data=g_max_new)
        out_sd.create_dataset('scatter_matrix', data=ns_flat)
        if has_mm:
            out_sd.create_dataset('multiplicity_matrix', data=mm_flat)

    @staticmethod
    def _scatter_to_dense(values: np.ndarray, g_min: np.ndarray,
                          g_max: np.ndarray, G: int) -> np.ndarray:
        """Compressed (per-source-group slice from g_min to g_max, 1-indexed) → dense G×G."""
        dense = np.zeros((G, G), dtype=np.float64)
        idx = 0
        for g in range(G):
            gmn = int(g_min[g]) - 1
            gmx = int(g_max[g]) - 1
            n = gmx - gmn + 1
            dense[g, gmn:gmx + 1] = values[idx:idx + n]
            idx += n
        return dense

    @staticmethod
    def _band_mask(g_min: np.ndarray, g_max: np.ndarray, G: int) -> np.ndarray:
        """Boolean (G, G) mask: True where (g, g') is inside the compressed band."""
        mask = np.zeros((G, G), dtype=bool)
        for g in range(G):
            gmn = int(g_min[g]) - 1
            gmx = int(g_max[g]) - 1
            mask[g, gmn:gmx + 1] = True
        return mask

    @staticmethod
    def _scatter_to_compressed(sm_dense: np.ndarray, mm_dense: np.ndarray,
                                g_min: np.ndarray, g_max: np.ndarray):
        """Dense G×G → flat compressed using the supplied g_min/g_max."""
        G = sm_dense.shape[0]
        sm_chunks = []
        mm_chunks = []
        for g in range(G):
            gmn = int(g_min[g]) - 1
            gmx = int(g_max[g]) - 1
            sm_chunks.append(sm_dense[g, gmn:gmx + 1])
            mm_chunks.append(mm_dense[g, gmn:gmx + 1])
        return np.concatenate(sm_chunks), np.concatenate(mm_chunks)

    @staticmethod
    def _scatter_to_compressed_single(sm_dense: np.ndarray,
                                       g_min: np.ndarray,
                                       g_max: np.ndarray) -> np.ndarray:
        """Dense G×G → flat compressed (scatter only, no multiplicity)."""
        G = sm_dense.shape[0]
        chunks = []
        for g in range(G):
            gmn = int(g_min[g]) - 1
            gmx = int(g_max[g]) - 1
            chunks.append(sm_dense[g, gmn:gmx + 1])
        return np.concatenate(chunks)

    def _merge_libraries(self, fuel_lib: Path, base_lib: Path, output: Path):
        """
        Merge fuel and base MGXS libraries.
        
        Handles temperature group merging for overlapping nuclides.
        Also sanitizes chi=0 nuclides to prevent NaN propagation.
        """
        shutil.copy(fuel_lib, output)
        
        with h5py.File(output, 'a') as merged, h5py.File(base_lib, 'r') as base:
            for key in base.keys():
                if key not in merged:
                    base.copy(key, merged)
                else:
                    # Merge temperature groups
                    for subkey in base[key].keys():
                        if subkey not in merged[key]:
                            base.copy(base[key][subkey], merged[key], name=subkey)
                        elif subkey == 'kTs':
                            for kt_key in base[key]['kTs'].keys():
                                if kt_key not in merged[key]['kTs']:
                                    base.copy(base[key]['kTs'][kt_key],
                                            merged[key]['kTs'], name=kt_key)
        
        # Sanitize chi=0 nuclides
        n_fixed = self._sanitize_library(output)
        if n_fixed > 0:
            print(f"  Sanitized {n_fixed} nuclides (chi=0 → uniform, fission→0)")
    
    def _sanitize_library(self, lib_path: Path) -> int:
        """
        Fix nuclides with chi=0 to prevent NaN propagation.
        
        Sets chi to uniform 1/G and zeros fission/nu-fission.
        """
        n_fixed = 0
        with h5py.File(lib_path, 'a') as f:
            for nuc in f.keys():
                for temp_key in f[nuc].keys():
                    if temp_key == 'kTs':
                        continue
                    grp = f[nuc][temp_key]
                    if 'chi' not in grp:
                        continue
                    chi = np.array(grp['chi'])
                    if not np.any(chi > 0):
                        n_groups = len(chi)
                        grp['chi'][...] = np.ones(n_groups) / n_groups
                        if 'fission' in grp:
                            grp['fission'][...] = 0.0
                        if 'nu-fission' in grp:
                            grp['nu-fission'][...] = 0.0
                        n_fixed += 1
        return n_fixed
    
    def get_available_nuclides(self, burnup: float, library_type: str = 'transport') -> set:
        """
        Get set of nuclides available in library at given burnup.
        
        Parameters
        ----------
        burnup : float
            Target burnup [MWd/kgHM]
        library_type : {'transport', 'depletion'}
            Which library type to query
        
        Returns
        -------
        nuclides : set
            Set of nuclide names available
        """
        bu_low, bu_high, alpha = self.get_bracketing_burnups(burnup)
        bu_use = bu_low if alpha < 0.5 else bu_high
        
        if library_type == 'transport':
            lib_path = self._get_transport_library_path(bu_use)
        else:
            lib_path = self._get_depletion_library_path(bu_use)
        
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {lib_path}")
        
        with h5py.File(lib_path, 'r') as f:
            if library_type == 'transport':
                return set(f.keys())
            else:
                # MicroXS format: nuclides stored in 'nuclides' dataset
                return set(n.decode() for n in f['nuclides'][:])
    
    def get_nuclides_in_library_file(self, lib_path: Path) -> set:
        """
        Return the set of nuclide names actually present in a given MGXS HDF5
        library file (top-level group names).

        Prefer this over :meth:`get_available_nuclides` when you have just
        produced an interpolated/merged library and need a filter that is
        consistent with its actual contents (union of bracketing burnups, plus
        base structural nuclides).
        """
        with h5py.File(lib_path, 'r') as f:
            return set(f.keys())

    def filter_material_nuclides(
        self,
        material: openmc.Material,
        burnup: Optional[float] = None,
        library_type: str = 'transport',
        available_nuclides: Optional[set] = None,
    ) -> Tuple[openmc.Material, int]:
        """
        Filter material to only include nuclides available in a library.

        Two usage modes:
        - Pass ``available_nuclides`` explicitly: the filter uses that set as-is.
          Recommended when you have just produced an interpolated/merged
          transport library, so the filter matches the actual file contents.
        - Pass ``burnup`` (and optionally ``library_type``): the filter looks up
          the nearest library on the relevant grid. Legacy behaviour, kept for
          backward compatibility.

        Parameters
        ----------
        material : openmc.Material
            Input material (will not be modified).
        burnup : float, optional
            Target burnup [MWd/kgHM]. Required if ``available_nuclides`` is None.
        library_type : {'transport', 'depletion'}
            Which library grid to consult when looking up by burnup.
        available_nuclides : set of str, optional
            Explicit set of nuclide names to keep. Overrides the burnup lookup.

        Returns
        -------
        filtered_material : openmc.Material
            New material with only available nuclides.
        n_removed : int
            Number of nuclides removed.
        """
        if available_nuclides is None:
            if burnup is None:
                raise ValueError(
                    "Either `available_nuclides` or `burnup` must be provided."
                )
            available_nuclides = self.get_available_nuclides(burnup, library_type)

        filtered = openmc.Material(material_id=material.id, name=material.name)
        n_removed = 0

        for nuclide, atom_density, percent_type in material.nuclides:
            if nuclide in available_nuclides:
                filtered.add_nuclide(nuclide, atom_density, percent_type)
            else:
                n_removed += 1

        filtered.depletable = material.depletable
        filtered.volume = material.volume
        filtered.temperature = material.temperature

        return filtered, n_removed
    
    def create_metadata_file(self, output_path: Optional[Path] = None):
        """
        Create library metadata file from current configuration.
        
        Useful for initializing a new library or documenting an existing one.
        """
        if output_path is None:
            output_path = self.library_path / "library_metadata.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Metadata saved to: {output_path}")
