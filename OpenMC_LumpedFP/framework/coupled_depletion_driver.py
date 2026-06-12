"""
Coupled Depletion Driver

Manages the coupling between transport and depletion calculations with
correct handling of heavy metal mass normalization and burnup tracking.

Key features:
- Calculates and preserves initial heavy metal mass (HM_initial)
- Tracks cumulative burnup [MWd/kgHM]
- Calls MGXSLibraryManager for burnup-appropriate cross sections
- Handles operator.heavy_metal override for custom depletion loops
- Provides clean interface for coupled simulations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import openmc
import openmc.deplete
from .mgxs_library_manager import MGXSLibraryManager


class CoupledDepletionDriver:
    """
    Driver for coupled multi-group transport and depletion simulations.
    
    Manages the iterative loop between transport (to compute flux) and
    depletion (to evolve material compositions), with correct handling
    of burnup normalization based on initial heavy metal inventory.
    
    Parameters
    ----------
    library_manager : MGXSLibraryManager
        Manager for MGXS libraries parametrized by burnup
    geometry_builder : Callable
        Function that creates OpenMC geometry given materials
        Signature: geometry_builder(materials) -> openmc.Geometry
    settings_builder : Callable
        Function that creates OpenMC settings
        Signature: settings_builder() -> openmc.Settings
    tallies_builder : Callable
        Function that creates tallies for flux extraction
        Signature: tallies_builder(fuel_material) -> openmc.Tallies
    chain_file : str or Path
        Path to depletion chain XML file
    power : float
        Constant power [W]
    output_dir : str or Path
        Directory for output files
    
    Attributes
    ----------
    HM_initial_grams : float
        Initial heavy metal inventory [g] (Z >= 90)
    HM_initial_kg : float
        Initial heavy metal inventory [kg]
    burnup_cumulative : float
        Cumulative burnup [MWd/kgHM]
    """
    
    def __init__(self,
                 library_manager: MGXSLibraryManager,
                 geometry_builder: Callable,
                 settings_builder: Callable,
                 tallies_builder: Callable,
                 chain_file: str | Path,
                 power: float,
                 output_dir: str | Path,
                 num_threads: Optional[int] = None):
        
        self.library_manager = library_manager
        self.geometry_builder = geometry_builder
        self.settings_builder = settings_builder
        self.tallies_builder = tallies_builder
        self.chain_file = Path(chain_file)
        self.power = power
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if num_threads is None:
            num_threads = int(os.getenv("OMP_NUM_THREADS", "1"))
        self.num_threads = num_threads
        
        # State variables (initialized in setup)
        self.HM_initial_grams = None
        self.HM_initial_kg = None
        self.burnup_cumulative = 0.0
        
        # Materials (set in setup)
        self.fuel_material = None
        self.structural_materials = []
        
        # Results tracking
        self.results = {
            'steps': [],
            'time_days': [],
            'timestep_days': [],
            'burnup_MWdkg': [],
            'keff': [],
            'keff_std': [],
            'flux': [],
        }
        
        print("CoupledDepletionDriver initialized")
        print(f"  Power: {self.power} W")
        print(f"  Output: {self.output_dir}")
        print(f"  Chain: {self.chain_file}")
    
    def setup_initial_materials(self, fuel_material: openmc.Material,
                               structural_materials: List[openmc.Material]):
        """
        Set up initial materials and calculate initial heavy metal inventory.
        
        This MUST be called before running depletion steps.
        
        Parameters
        ----------
        fuel_material : openmc.Material
            Fresh fuel material (depletable=True, volume set)
        structural_materials : list of openmc.Material
            Non-depletable structural materials (clad, coolant, etc.)
        """
        self.fuel_material = fuel_material
        self.structural_materials = structural_materials
        
        # Calculate initial HM mass (ONE TIME ONLY)
        self.HM_initial_grams = self._calculate_heavy_metal_mass(fuel_material)
        self.HM_initial_kg = self.HM_initial_grams * 1e-3
        
        print(f"\nInitial heavy metal inventory:")
        print(f"  {self.HM_initial_grams:.6f} g")
        print(f"  {self.HM_initial_kg:.6f} kg")
        
        # Save HM_initial for future reference
        metadata = {
            'HM_initial_grams': self.HM_initial_grams,
            'HM_initial_kg': self.HM_initial_kg,
            'fuel_volume_cm3': fuel_material.volume,
            'power_W': self.power,
        }
        with open(self.output_dir / 'initial_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _calculate_heavy_metal_mass(self, material: openmc.Material) -> float:
        """
        Calculate heavy metal mass [g] for a material.
        
        Heavy metals = nuclides with Z >= 90 (actinides).
        """
        if material.volume is None:
            raise ValueError("Material volume must be set")
        
        density_g_cm3 = 0.0
        for nuc, atoms_per_bcm in material.get_nuclide_atom_densities().items():
            Z = openmc.data.zam(nuc)[0]
            if Z >= 90:
                # Convert atom density [atom/b-cm] to mass density [g/cm³]
                density_g_cm3 += (1e24 * atoms_per_bcm * 
                                 openmc.data.atomic_mass(nuc) / 
                                 openmc.data.AVOGADRO)
        
        return density_g_cm3 * material.volume
    
    def run_transport_step(self, step_number: int) -> Tuple[np.ndarray, float, float]:
        """
        Run transport calculation for current material state.
        
        Parameters
        ----------
        step_number : int
            Step number (for directory naming)
        
        Returns
        -------
        flux : np.ndarray
            Multigroup flux [n-cm/src], reversed to ECCO order (fast→thermal)
        keff : float
            Effective multiplication factor
        keff_std : float
            Standard deviation of keff
        """
        step_dir = self.output_dir / f"step_{step_number}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Get transport library for current burnup.
        # If the transport files are pre-merged (base materials already embedded
        # during library generation), no base_library_path is needed and the
        # merge+sanitize overhead at runtime is eliminated entirely.
        base_lib = self.library_manager.library_path / "base" / "mgxs_base_materials.h5"
        transport_premerged = self.library_manager._transport_premerged()

        transport_lib = self.library_manager.get_transport_library(
            burnup=self.burnup_cumulative,
            base_library_path=None if transport_premerged else base_lib,
        )
        
        # Filter fuel against the nuclides actually present in the produced
        # transport library file. This is consistent with linear interpolation,
        # which can include the union of nuclides from the two bracketing
        # libraries — a wider set than the nearest-neighbour one.
        transport_nuclides = self.library_manager.get_nuclides_in_library_file(
            transport_lib
        )
        fuel_filtered, n_removed = self.library_manager.filter_material_nuclides(
            self.fuel_material,
            available_nuclides=transport_nuclides,
        )

        if n_removed > 0:
            print(f"  Filtered {n_removed} nuclides, {len(fuel_filtered.get_nuclides())} remain")

        # Build and export XMLs directly into step_dir so that openmc.run(cwd=step_dir)
        # finds them. Using the `path` argument avoids os.chdir side effects.
        materials = openmc.Materials([fuel_filtered] + self.structural_materials)
        materials.cross_sections = str(transport_lib)
        materials.export_to_xml(path=step_dir / "materials.xml")

        geometry = self.geometry_builder([fuel_filtered] + self.structural_materials)
        geometry.export_to_xml(path=step_dir / "geometry.xml")

        settings = self.settings_builder()
        settings.export_to_xml(path=step_dir / "settings.xml")

        tallies = self.tallies_builder(fuel_filtered)
        tallies.export_to_xml(path=step_dir / "tallies.xml")

        # Run OpenMC
        print(f"  Running transport (threads={self.num_threads})...")
        openmc.run(cwd=str(step_dir), threads=self.num_threads)
        
        # Extract results
        sp_files = sorted(step_dir.glob("statepoint.*.h5"))
        if not sp_files:
            raise FileNotFoundError(f"No statepoint file in {step_dir}")
        
        sp = openmc.StatePoint(str(sp_files[-1]), autolink=False)
        
        flux_tally = sp.get_tally(name='flux')
        flux = np.asarray(flux_tally.mean).ravel()[::-1]  # Reverse to ECCO order
        
        keff = sp.keff.n
        keff_std = sp.keff.s
        
        sp.close()
        
        print(f"  keff = {keff:.5f} ± {keff_std:.5f}")
        print(f"  Total flux = {flux.sum():.4e} n-cm/src")
        
        return flux, keff, keff_std
    
    def run_depletion_step(self, flux: np.ndarray, timestep_days: float,
                          step_number: int) -> Path:
        """
        Run depletion calculation for one timestep.
        
        Parameters
        ----------
        flux : np.ndarray
            Multigroup flux from transport [n-cm/src]
        timestep_days : float
            Time step duration [days]
        step_number : int
            Step number (for directory naming)
        
        Returns
        -------
        depletion_file : Path
            Path to depletion_results.h5 file
        """
        step_dir = self.output_dir / f"step_{step_number}"
        
        # Get depletion library (MicroXS) for current burnup
        micro_xs = self.library_manager.get_depletion_library(
            burnup=self.burnup_cumulative
        )
        
        # Create IndependentOperator
        openmc.config['chain_file'] = str(self.chain_file)
        
        operator = openmc.deplete.IndependentOperator(
            materials=[self.fuel_material],
            fluxes=[flux],  # Multi-group flux array [n·cm/src]
            micros=[micro_xs],
            normalization_mode='fission-q',
        )
        
        # Override heavy_metal to use initial value (defensive)
        # Note: with timestep_units='d' and power=..., this override is not
        # strictly necessary (OpenMC doesn't use it), but ensures consistency
        # if operator is reused or if timestep_units is changed to 'MWd/kg'
        operator.heavy_metal = self.HM_initial_grams
        
        # Calculate source rate from power and flux
        # source_rate [neutrons/s] = power [W] / <E_fission> [J/fission] / q [J/neutron]
        # For simplicity, use power directly as integrator handles this
        
        # Create integrator
        integrator = openmc.deplete.PredictorIntegrator(
            operator=operator,
            timesteps=[timestep_days],
            power=self.power,
            timestep_units='d'
        )
        
        # Run depletion
        depletion_file = step_dir / "depletion_results.h5"
        integrator.integrate(path=str(depletion_file), write_rates=True)
        
        # Calculate burnup increment
        delta_burnup = (self.power * timestep_days) / (self.HM_initial_kg * 1e6)
        self.burnup_cumulative += delta_burnup
        
        print(f"  Δburnup = {delta_burnup:.4f} MWd/kg")
        print(f"  Cumulative burnup = {self.burnup_cumulative:.4f} MWd/kg")
        
        return depletion_file
    
    def update_fuel_from_depletion(self, depletion_file: Path):
        """
        Update fuel material composition from depletion results.
        
        Parameters
        ----------
        depletion_file : Path
            Path to depletion_results.h5 file
        """
        results = openmc.deplete.Results(str(depletion_file))
        last_result = results[-1]
        old_fuel = last_result.get_material(str(self.fuel_material.id))
        
        # Create new fuel material with updated composition
        new_fuel = openmc.Material(
            material_id=self.fuel_material.id,
            name=self.fuel_material.name
        )
        
        for nuclide, atom_density, _ in old_fuel.nuclides:
            new_fuel.add_nuclide(nuclide, atom_density, 'ao')
        
        new_fuel.depletable = True
        new_fuel.volume = self.fuel_material.volume
        new_fuel.temperature = self.fuel_material.temperature
        
        self.fuel_material = new_fuel
        
        print(f"  Updated fuel: {len(new_fuel.get_nuclides())} nuclides")
    
    def save_step_results(self, step_number: int, timestep_days: float,
                         cumulative_time: float, keff: float, keff_std: float,
                         flux: np.ndarray):
        """Save results for a single step."""
        step_dir = self.output_dir / f"step_{step_number}"
        
        step_results = {
            'step': step_number,
            'timestep_days': timestep_days,
            'cumulative_time_days': cumulative_time,
            'burnup_MWdkg': self.burnup_cumulative,
            'keff': float(keff),
            'keff_std': float(keff_std),
            'flux': flux.tolist(),
        }
        
        with open(step_dir / "step_results.json", 'w') as f:
            json.dump(step_results, f, indent=2)
        
        # Update summary
        self.results['steps'].append(step_number)
        self.results['timestep_days'].append(timestep_days)
        self.results['time_days'].append(cumulative_time)
        self.results['burnup_MWdkg'].append(self.burnup_cumulative)
        self.results['keff'].append(float(keff))
        self.results['keff_std'].append(float(keff_std))
        self.results['flux'].append(flux.tolist())
    
    def run_burnup_schedule(self, burnup_targets_MWdkg: List[float],
                           start_from_step: int = 0):
        """
        Run coupled simulation targeting specific burnup values.
        
        Converts burnup targets [MWd/kgHM] to time steps [days] based on
        power and initial HM mass, then runs coupled simulation.
        
        Parameters
        ----------
        burnup_targets_MWdkg : list of float
            Target burnup values [MWd/kgHM] for each depletion step
        start_from_step : int, optional
            Step number to start from (for restarts)
        
        Returns
        -------
        results : dict
            Simulation results summary
        """
        if self.HM_initial_kg is None:
            raise RuntimeError("Must call setup_initial_materials() before running simulation")
        
        # Convert burnup targets to timesteps
        timesteps_days = []
        bu_prev = 0.0
        for bu_target in burnup_targets_MWdkg:
            delta_bu = bu_target - bu_prev
            delta_time = (delta_bu * self.HM_initial_kg * 1e6) / self.power
            timesteps_days.append(delta_time)
            bu_prev = bu_target
        
        print(f"Converted burnup targets {burnup_targets_MWdkg} MWd/kg")
        print(f"  to timesteps {timesteps_days} days")
        
        return self.run_coupled_simulation(timesteps_days, start_from_step)
    
    def run_coupled_simulation(self, timesteps_days: List[float],
                              start_from_step: int = 0):
        """
        Run full coupled transport-depletion simulation.
        
        Parameters
        ----------
        timesteps_days : list of float
            Time steps [days] for each depletion interval
        start_from_step : int, optional
            Step number to start from (for restarts). Default is 0 (fresh fuel).
        
        Returns
        -------
        results : dict
            Simulation results summary
        """
        n_steps = len(timesteps_days)
        cumulative_time = 0.0
        
        print("\n" + "="*70)
        print("COUPLED TRANSPORT-DEPLETION SIMULATION")
        print("="*70)
        print(f"Total steps: {n_steps + 1} (step_0 to step_{n_steps})")
        print(f"Initial HM: {self.HM_initial_kg:.6f} kg")
        print(f"Power: {self.power} W")
        print("="*70)
        
        # STEP 0: Fresh fuel transport only (no depletion)
        if start_from_step == 0:
            print(f"\n{'='*70}")
            print(f"STEP 0 - FRESH FUEL (NO DEPLETION)")
            print(f"{'='*70}")
            
            flux, keff, keff_std = self.run_transport_step(0)
            self.save_step_results(0, 0.0, 0.0, keff, keff_std, flux)
            
            print(f"✓ Completed step 0")
        
        # STEPS 1 to N: Depletion + Transport
        for step_idx in range(n_steps):
            if step_idx + 1 < start_from_step:
                continue
            
            step_number = step_idx + 1
            timestep = timesteps_days[step_idx]
            cumulative_time += timestep
            
            print(f"\n{'='*70}")
            print(f"STEP {step_number}/{n_steps}")
            print(f"Timestep: {timestep} days (cumulative: {cumulative_time:.1f} days)")
            print(f"Burnup: {self.burnup_cumulative:.4f} MWd/kg")
            print(f"{'='*70}")
            
            # Run depletion
            print("[1/3] Depletion...")
            depletion_file = self.run_depletion_step(flux, timestep, step_number)
            
            # Update fuel composition
            print("[2/3] Updating fuel composition...")
            self.update_fuel_from_depletion(depletion_file)
            
            # Run transport
            print("[3/3] Transport...")
            flux, keff, keff_std = self.run_transport_step(step_number)
            
            # Save results
            self.save_step_results(step_number, timestep, cumulative_time,
                                  keff, keff_std, flux)
            
            print(f"✓ Completed step {step_number}")
        
        # Save final summary
        with open(self.output_dir / "simulation_summary.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Final burnup: {self.burnup_cumulative:.4f} MWd/kg")
        print(f"Final time: {cumulative_time:.1f} days")
        print(f"keff evolution: {[f'{k:.5f}' for k in self.results['keff']]}")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
        
        return self.results
