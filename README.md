# Lumped Fission Products for OpenMC

This repository contains the code and data developed during my Master's Thesis on the generation and application of parent-specific Lumped Fission Products (LFPs) for multigroup depletion calculations in OpenMC.

The objective of the work is to reduce the computational cost of Monte Carlo depletion by replacing the explicit treatment of hundreds of fission products with a reduced set of parent-specific lumped pseudo-nuclides while preserving their neutronic impact.

The repository includes:

## Repository Structure

### `custom_chain.xml`
Customized OpenMC depletion chain containing the parent-specific LFPs used in this work.

### `simplified_chain.txt`
Human-readable description of the depletion chain used to generate `custom_chain.xml`.

### `chain_creator.py`
Utility script that generates `custom_chain.xml` starting from `simplified_chain.txt`.

### `LFP_XS_Library.py`
Script used to extract multigroup cross sections from OpenMC tallies and generate the corresponding LFP libraries.

### `MGXS_Library/`
Burnup-dependent ECCO-33 multigroup cross-section libraries used in the thesis calculations.

### `MGXS_Library_AllFP/`
Multigroup cross sections generated using the full explicit fission-product inventory. These data are included for comparison purposes and possible future developments.

### `WrapClass/`
Contains the Python classes used to manage the burnup-dependent libraries and perform coupled multigroup transport-depletion calculations. A more detailed description is provided in the README file inside the folder.

### `Coupled_LFP/`
Example application of the framework to the Superphénix reference pin-cell model.

### `Coupled_LFP_ESFR/`
Example application of the framework to an ESFR-like pin-cell model.

## Reference

The methodology is described in:

Federico Pati, *Generation and Application of Parent-Specific Lumped Fission Products for OpenMC Multigroup Depletion Calculations*, Master's Thesis, Politecnico di Torino / KTH Royal Institute of Technology, 2026.
