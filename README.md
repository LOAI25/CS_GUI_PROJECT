# A Unified Graphical User Interface for Compressed Sensing Reconstruction of STEM Images Using Multiple Algorithms

This project provides a **PyQt5-based GUI** for comparing multiple compressed sensing reconstruction algorithms on STEM (Scanning Transmission Electron Microscopy) images.  
It supports ROI selection, noise control, patch-based processing, advanced algorithm parameters, and automated batch execution across different environments.

---


## Test Data
For quick testing and demonstration, sample STEM data is provided in the **`STEM data/`** folder.  
The test data are stored as **HDF5 files** and were downloaded from [Zenodo](https://zenodo.org/records/2652906). 
You can load these files directly through the GUI using the **"Choose your HDF5 file"** button.  
They are suitable for verifying the workflow without preparing your own STEM dataset.


## How to Use

The typical workflow can be divided into four stages: **load file → set parameters → run compressed sensing → export results**.  
We recommend loading the file first, but you may also set parameters before loading; both orders are valid.

1. **Load HDF5 file**  
   - Click **"Choose your HDF5 file"**.  
   - If the file is 3D, select the **slice index** you want to process.

2. **(Optional) Select Slice Index**  
   - For 3D data, choose the slice index within the valid range.

3. **(Optional) Select ROI**  
   - By default, the full image is used.  
   - If you uncheck **"Use full image"**, you can define the ROI (sub-image) size and coordinates.

4. **Choose Algorithms**  
   - Select one or multiple algorithms from the list.  
   - Click again to deselect. Multi-selection is supported.

5. **Set Sampling Rate**  
   - Configure the sampling rate between **0.0 and 1.0**.

6. **Select Sampling Method**  
   - Choose either:
     - `random`
     - `linehop`

7. **Noise Settings**  
   - By default, images are clean (no noise).  
   - If you uncheck **"Use clean image"**, you can specify an **SNR (dB)** and Gaussian noise will be added.

8. **Set Patch Size and Stride**  
   - Reconstruction is patch-based with overlapping regions.  
   - Adjust **patch size** and **stride** according to your needs.

9. **(Optional) Advanced Settings**  
   - Click **"Advanced settings"** to configure algorithm-specific parameters.

10. **Run Reconstruction**  
    - Click **"Do compressed sensing"** to start the process.  
    - The program will queue and run all selected algorithms.

11. **(Optional) Export Results**  
    - After reconstruction, the **"Export results"** button becomes available.  
    - You can export images (PNG/JPG/PDF) along with JSON files containing metrics.

---

## Dependencies

This project uses multiple environments for the GUI and algorithms:  

- The GUI (`main_gui.py`) runs in an environment named **`gui_env`**.  
- Algorithm-specific environments are defined in each algorithm’s **`env.json`**.  
- All environment configuration files are provided in the **`environments/`** folder.

### Install with Conda

To create an environment from a `.yml` file:

```bash
conda env create -f xxx.yml
```


Here, xxx.yml should be replaced with the name of the corresponding environment file (e.g., gui_env.yml).

### Matlab
Some algorithms in this project rely on **MATLAB** for execution. Please make sure you have MATLAB installed on your system.

- Recommended version: **MATLAB R2025a** (other recent versions may also work).
- You need to update the MATLAB path in `run_algorithm.py` to match your local installation.

For example, in `run_algorithm.py`, replace:

```python
default_matlab_path = "/Applications/MATLAB_R2025a.app/bin/maca64/MATLAB"
```


This project has been tested on **Apple Silicon (MacBook Pro with M4 Pro chip)** and runs successfully.  