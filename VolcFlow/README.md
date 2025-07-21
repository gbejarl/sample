# VolcFlow Model Usage for Cotopaxi Volcano

This repository provides instructions and adapted input files for using the **VolcFlow** model to simulate pyroclastic flows on Cotopaxi volcano. VolcFlow was developed by Karim Kelfoun at Université Clermont Auvergne.

## About VolcFlow

**VolcFlow** is a numerical model for simulating volcanic flows. Please note that I am not the developer of VolcFlow; I use the model for research purposes.

## Adaptations

To tailor VolcFlow for Cotopaxi volcano, I have adapted the following files:

- `input_COT_PDC.m`: Custom input flow parameters for Cotopaxi pyroclastic density currents.
- `repr_COT_PDC.m`: Modified representation file for simulation outputs.

## How to Use

1. **Obtain VolcFlow**  
    Download the VolcFlow model from its official source or copy the `VolcFlow.p` and `VolcFlowFig.fig` from the provided repository.

2. **Replace Input Files**  
    Copy the adapted `input_COT_PDC.m` and `repr_COT_PDC` files into your VolcFlow directory.

3. **Run the Simulation**  
    Execute VolcFlow using the provided input files to simulate pyroclastic flows on Cotopaxi. For this, make sure the `VolcFlow.p` and `VolcFlowFig.fig` files are in your current directory. Run the `VolcFlow.p` script in MATLAB and select the `input_COT_PDC.m` file when prompted. In the VolcFlow GUI, you may click the Representation button to show the extent of the terrain model from where the pyroclastic flows will be modeled. In the VolcFlow GUI, click the OK button when ready to run the simulation. This will generate an animation and a .dat file with numerical results.

## References

- [VolcFlow Model Documentation](https://lmv.uca.fr/volcflow/)
