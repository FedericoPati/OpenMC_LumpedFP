# OpenMC-LFP

This repository contains the code and data developed for my Master's Thesis on the generation and application of parent-specific Lumped Fission Products (LFPs) for multigroup depletion calculations in OpenMC.

The objective of the work is to reduce the computational cost of Monte Carlo depletion by replacing the explicit treatment of many fission products with a reduced set of parent-specific lumped pseudo-nuclides, while preserving their aggregate neutronic effect.

## Repository Structure

### `chains/`

Contains the depletion-chain files and tools.

- `custom_chain.xml`: customized OpenMC depletion chain including the parent-specific LFPs.
- `simplified_chain.txt`: human-readable chain description used to generate `custom_chain.xml`.
- `chain_creator.py`: script used to generate the customized OpenMC depletion chain from `simplified_chain.txt`.

### `library_generation/`

Contains the scripts used to generate the LFP multigroup cross-section libraries.

- `LFP_XS_Library.py`: extracts multigroup cross sections from OpenMC tally outputs and constructs the LFP libraries according to the customized depletion chain.

### `libraries/`

Contains the multigroup cross-section libraries used in the work.

- `MGXS_Library/`: burnup-dependent ECCO-33 multigroup cross-section libraries containing the parent-specific LFPs.
- `Library_MGXS_AllFP/`: multigroup cross-section libraries generated with explicit fission products, included for comparison and possible future developments.

### `framework/`

Contains the Python classes used to perform coupled multigroup transport and depletion calculations with burnup-dependent libraries.

Main components:

- `MGXSLibraryManager`: manages the burnup-dependent multigroup cross-section libraries and returns the data required at each burnup step.
- `CoupledDepletionDriver`: orchestrates the coupled multigroup transport-depletion loop, including transport execution, flux extraction, depletion, material reconstruction, and library selection.

A more detailed description of the framework is provided inside the folder.

### `examples/`

Contains example applications of the framework.

- `Superphenix/`: application to the Superphénix-like reference pin-cell model.
- `ESFR/`: application to the ESFR-like pin-cell model used for the transferability test.


## Reference

This repository is associated with the following Master's Thesis:

Federico Pati, *Methodology for Generation of Multi-Group Lumped Fission Products for Fast Spectrum*, Master's Thesis, Politecnico di Torino and KTH Royal Institute of Technology, 2025/2026.

