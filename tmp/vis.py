import os
import numpy as np
import scipy.io as sio
import pyvista as pv

# --- FILE PATHS ---
image_mat_path = "/Users/aldipiroli/workspace/projects/prnet_from_scratch/data/300W_LP/AFW/AFW_134212_1_0.mat"  # 3DMM parameters for one image
bfm_model_path = "/Users/aldipiroli/workspace/projects/prnet_from_scratch/data/300W_LP/Code/BFM.mat"         # Basel Face Model
bfm_uv_path    = "/Users/aldipiroli/workspace/projects/prnet_from_scratch/data/300W_LP/Code/BFM_UV.mat"      # UV coordinates
save_folder    = "."

os.makedirs(save_folder, exist_ok=True)

# --- LOAD BFM MODEL ---
bfm_struct = sio.loadmat(bfm_model_path, struct_as_record=False, squeeze_me=True)
bfm_data = bfm_struct["model"]




bfm_struct = sio.loadmat(bfm_model_path, struct_as_record=False, squeeze_me=True)
bfm_data = bfm_struct['model']  # or ['BFM'] depending on the file
if isinstance(bfm_data, np.ndarray):
    bfm_data = bfm_data[0,0]

# Access shape mean
shape_mean = bfm_data.shapeMU
# Access shape PCs
shape_pcs = bfm_data.shapePC
# Access triangle indices
triangles = bfm_data.tri
meanshape = bfm_data.shapeMU


# Triangles (0-based)
triangles = bfm_data.tri - 1  # Convert from MATLAB 1-based to 0-based indexing

# Faces for PyVista
faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int64)
faces = faces.flatten()

# Create mesh
mesh3d = pv.PolyData(meanshape, faces)

# Visualize
plotter = pv.Plotter()
plotter.add_mesh(mesh3d, color='lightblue', show_edges=True)
plotter.add_axes()
plotter.show_grid()
plotter.show()
