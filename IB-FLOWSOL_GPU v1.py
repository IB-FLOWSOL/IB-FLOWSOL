import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init
from matplotlib.animation import FuncAnimation
import psutil
from scipy.sparse.linalg import cg
from tqdm import tqdm
import torch
import scipy
import os
import cupyx.scipy.sparse as gpu_sp
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
import scipy.sparse as sp
import cupy as cp
# import cupyx.scipy.sparse as sp
import gc
import cupyx.scipy.sparse as cusparse


output_dir = r"D:/numerical computation/Results/FAH4001"

os.makedirs(output_dir, exist_ok=True)
#=======================================================================================================================================#
#                                                                VERSION CHECK BLOCK
#=======================================================================================================================================#
print("="*60)
print("Environment & Library Versions")
print("="*60)

import sys

print("Python version        :", sys.version.replace("\n", " "))

print("NumPy version         :", np.__version__)
print("Matplotlib version    :", plt.matplotlib.__version__)
print("SciPy version         :", scipy.__version__)
print("CuPy version          :", cp.__version__)
print("PyTorch version       :", torch.__version__)
print("psutil version        :", psutil.__version__)

# CUDA info (PyTorch)
print("CUDA / GPU Info (PyTorch)")
print("CUDA available        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version          :", torch.version.cuda)
    print("GPU name              :", torch.cuda.get_device_name(0))
    print("GPU capability        :", torch.cuda.get_device_capability(0))

# CUDA info (CuPy)
print("CUDA / GPU Info (CuPy)")
try:
    print("CuPy CUDA runtime     :", cp.cuda.runtime.runtimeGetVersion())
    print("CuPy GPU name         :", cp.cuda.Device(0).name)
except Exception as e:
    print("CuPy CUDA info error  :", e)

print("="*60)

# Total and available memory (in GB)
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(Fore.GREEN + f"total space {total} GB" + Style.RESET_ALL)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   IMPORTING DATA
#=======================================================================================================================================#

mesh_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz")
gax_file = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# sorted_first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# second_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
inside_pt = mesh_data["array1"]
ghost_nodes = gax_file["array1"]
# print(":::",len(ghost_nodes))
# time.sleep(900)
sorted_first_interface = gax_file["array7"]
sfi = gax_file["array8"]
ffi = gax_file["array9"]
ghost_nodes_list = ghost_nodes_list = [list(map(tuple, block)) for block in ghost_nodes]
first_interface = set(tuple(point) for point in ffi)
first_interface = np.array(
    sorted(first_interface),
    dtype=float)

second_interface = np.array(list(sfi.item()), dtype=float)
del_h = float(mesh_data["del_h"]) 

number_of_ghost_nodes = int(mesh_data["total_number_of_ghost_nodes"])

r_m = gax_file["array11"]
interpolation_coords = gax_file["array12"]
interpolation_normal_dist = gax_file["array13"]

#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   REBUILDING DOMAIN
#=======================================================================================================================================#

conversion_factor = 1/del_h     # mesh size
# conversion_factor = 
print("grid size ",inside_pt[0][0],inside_pt[0][1])
print(conversion_factor)


def cord_transfer_logic(a):
    x_coord=inside_pt[a][0]
    y_coord=inside_pt[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l1(a):
    x_coord=first_interface[a][0]
    y_coord=first_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l2(a):
    x_coord=second_interface[a][0]
    y_coord=second_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

cn = nx = int(mesh_data["nx"])  #201
rn = ny = int(mesh_data["ny"]) 
print(del_h,nx,ny)


# numeric 2D mesh

u_mat = np.full((rn, cn), np.nan)   # u_velocity

v_mat = np.full((rn, cn), np.nan)   # v_velocity

p_mat = np.full((rn, cn), np.nan)   # pressure


variable_array = []                                 # variable marker mesh (in use for pressure)
for i in range (0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    x = f'x{r}|{c}'       # beware of row and column to x-y coordinate system
    variable_array.append(x)
# print("🐒: ",len(variable_array),variable_array)

print("#-mapping begins...")
index_map = {val: i for i, val in enumerate(variable_array)}
print("#-mapping complete 👍🏼")

# Appending all the initial conditions to respective mesh nodes u, v and p (for fluid nodes)
for i in range(0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    u_mat[r][c] = 0                 # uniform initial velocity (u) condition through out the geometry
    v_mat[r][c] = 0                 # uniform initial velocity (v) condition through out the geometry
    p_mat[r][c] = 0                 # uniform initial pressure condition
    
lowerlimit = 0
upperlimit = 1
Z = u_mat  # Example of initial Z
x = np.linspace(lowerlimit, upperlimit, Z.shape[1]) 
y = np.linspace(lowerlimit, upperlimit, Z.shape[0])
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
plt.title("Quick geometry Check...")
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# print(u_mat[30])
# time.sleep(900)


drich_u = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             # x direction velocity drichilit boundary condition
drich_v = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]            # y direction velocity drichilit boundary condition
drich_p = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]            # pressure drichilit boundary condition


for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        u_mat[r][c] = drich_u[i]
        v_mat[r][c] = drich_v[i]
        p_mat[r][c] = drich_p[i]


#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#
#------------------------- set external functions for BCs--------------------#
u_max = 1.5
# ymid = 120
# H = 98
start_point = int(round((4.05 * conversion_factor),0))
end_point =  int(round((5.95 * conversion_factor),0))
H = end_point-start_point
ymid = start_point + (H/2)
print("LAND",start_point,end_point)
# time.sleep(8444)
#=======================================================================================================================================#
#                                                       SOME EXAMPLES OF BOUNDARY CONDITIONS
#=======================================================================================================================================#

# ==============================================Channel Flow================================================
# drich_u = [0,"NDN",0,1]                         # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                             # v velocity drichilit boundary condition
# drich_p = ["NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]             # v velocity drichilit boundary condition
# Ne_BC_p = [0,"NCN",0,0]                         # p velocity drichilit boundary condition

# Orlanski_BC_u = ["NON","NON","NON","NON","NON","NON"]
# Orlanski_BC_v = ["NON","NON","NON","NON","NON","NON"]
# # #==========================================Lid driven Cavity=======================================
# drich_u = [0,0,1,0]                                 # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN","NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,0,0,0]                                 # p velocity drichilit boundary condition

# Orlanski_BC_u = ["NON","NON","NON","NON","NON","NON"]
# Orlanski_BC_v = ["NON","NON","NON","NON","NON","NON"]

# # #==========================================Trapezoidal channel flow =======================================
drich_u = [0,"NDN",0,1]                                 # u velocity drichilit boundary condition
drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
drich_p = ["NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
Ne_BC_u = ["NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
Ne_BC_p = [0,"NCN",0,0]                                 # p velocity drichilit boundary condition

Orlanski_BC_u = ["NON","NON","NON","NON","NON","NON"]
Orlanski_BC_v = ["NON","NON","NON","NON","NON","NON"]

# # #==========================================CD-Nozzle flow =======================================
# drich_u = [0,"NDN",0,0,1]                                 # u velocity drichilit boundary condition
# drich_v = [0,"NDN",0,0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN",0,"NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN",0,"NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN",0,"NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,"NCN",0,0,0]                                 # p velocity drichilit boundary condition

# Orlanski_BC_u = ["NON","NON","NON","NON","NON","NON"]
# Orlanski_BC_v = ["NON","NON","NON","NON","NON","NON"]


#=========================================================================================================================================#
#                                                        SETTING UP BOUNDARY CONDITIONS
#=========================================================================================================================================#
# drich_u = [0,0,1,0]                                 # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN","NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,0,0,0]                                 # p velocity drichilit boundary condition

# Orlanski_BC_u = ["NON","NON","NON","NON","NON","NON"]
# Orlanski_BC_v = ["NON","NON","NON","NON","NON","NON"]
#========================================================================================================================================#
for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        # print(r,c,"mmmm")
        if (drich_u[i] != "NDN"):
            if (drich_u[i] == "f"):
                i_space = int(round((y * conversion_factor),0))
                print(i,j,r,c)
                u_parabola = u_max * (1- ((i_space-ymid)/(H/2))**2)
                u_mat[r][c] = u_parabola
            else:
                u_mat[r][c] = drich_u[i]

        if (drich_v[i] != "NDN"):
            v_mat[r][c] = drich_v[i]
        if (drich_p[i] != "NDN"):    
            p_mat[r][c] = drich_p[i]
     

#=======================================================================================================================================#
#                                                               END
#=======================================================================================================================================#
#=============Linearizing data structure for Neumann BCs============
# first interface
rf_list = []
cf_list = []
# mirror point 
rm_list = []
cm_list = []
# ghost nodes
rgn_list = []
cgn_list = []
# Neumann BC list
Ne_BC_u_list = []
Ne_BC_v_list = []
Ne_BC_p_list = []

alpha_u = []
alpha_v = []
alpha_p = []

beta_u = []
beta_v = []
beta_p = []


for i in range(len(sorted_first_interface)):
    for j in range(len(sorted_first_interface[i])):
        xf, yf = sorted_first_interface[i][j]
        rf_list.append(int(round(yf * conversion_factor))) 
        cf_list.append(int(round(xf * conversion_factor)))

        if (Ne_BC_u[i] != "NCN"):
            Ne_BC_u_list.append(Ne_BC_u[i])
            alpha_u.append(1)
        else:
            Ne_BC_u_list.append(np.nan)
            alpha_u.append(0)

        if (Ne_BC_v[i] != "NCN"):
            Ne_BC_v_list.append(Ne_BC_v[i])
            alpha_v.append(1)
        else:
            Ne_BC_v_list.append(np.nan)
            alpha_v.append(0)

        if (Ne_BC_p[i] != "NCN"):
            Ne_BC_p_list.append(Ne_BC_p[i])
            alpha_p.append(1)
        else:
            Ne_BC_p_list.append(np.nan)
            alpha_p.append(0)

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        xgn, ygn = ghost_nodes[i][j]
        rgn_list.append(int(round(ygn * conversion_factor))) 
        cgn_list.append(int(round(xgn * conversion_factor)))



# moving linearized list to GPU (common to Drichilet and Neumann)
rf = cp.asarray(rf_list)
cf = cp.asarray(cf_list)

rgn = cp.asarray(rgn_list)
cgn = cp.asarray(cgn_list)
# moving to GPU memory
Ne_BC_u_vector = cp.asarray(Ne_BC_u_list)
Ne_BC_v_vector = cp.asarray(Ne_BC_v_list)
Ne_BC_p_vector = cp.asarray(Ne_BC_p_list)
# masking
Ne_mask_u = ~cp.isnan(Ne_BC_u_vector)
Ne_mask_v = ~cp.isnan(Ne_BC_v_vector)
Ne_mask_p = ~cp.isnan(Ne_BC_p_vector)

#=========Linearizing data structure for Drichilet BCs=========
 
drich_bc_u_list = []
drich_bc_v_list = []
drich_bc_p_list = []

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        #'''''''''''' u_velocity - section ''''''''''''''#
        if (drich_u[i] != "NDN"):
            if (drich_u[i] == "f"):
                i_space = int(round((y * conversion_factor),0))
                u_parabola = u_max* (1- ((i_space - ymid)/(H/2))**2) # the parabolic function at inlet
                drich_bc_u_list.append(u_parabola)
            else:
                drich_bc_u_list.append(drich_u[i])
            beta_u.append(1)
        else:
            drich_bc_u_list.append(np.nan)
            beta_u.append(0)
        
        #'''''''''''' v_velocity - section ''''''''''''''#

        if(drich_v[i] != "NDN"):
            drich_bc_v_list.append(drich_v[i])
            beta_v.append(1)
        else:
            drich_bc_v_list.append(np.nan)
            beta_v.append(0)
        
        #'''''''''''''pressure section''''''''''''''''''#

        if(drich_p[i] != "NDN"):
            drich_bc_p_list.append(drich_p[i])
            beta_p.append(1)
        else:
            drich_bc_p_list.append(np.nan)
            beta_p.append(0)

# moving to GPU memory
drich_bc_u_vector = cp.asarray(drich_bc_u_list)
drich_bc_v_vector = cp.asarray(drich_bc_v_list)
drich_bc_p_vector = cp.asarray(drich_bc_p_list)
# creating mask
drich_mask_u = ~cp.isnan(drich_bc_u_vector)
drich_mask_v = ~cp.isnan(drich_bc_v_vector)
drich_mask_p = ~cp.isnan(drich_bc_p_vector) 

#=========Linearizing data structure for Orlanski BCs=========
Orlanski_bc_u_list = []
Orlanski_bc_v_list = []

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):

        if (Orlanski_BC_u[i] != "NON"):
            Orlanski_bc_u_list.append(Orlanski_BC_u[i])
        else:
            Orlanski_bc_u_list.append(np.nan)
        if(Orlanski_BC_v[i] != "NON"):
            Orlanski_bc_v_list.append(Orlanski_BC_v[i])
        else:
            Orlanski_bc_v_list.append(np.nan)
        # if(drich_p[i] != "NDN"):
        #     drich_bc_p_list.append(drich_p[i])
        # else:
        #     drich_bc_p_list.append(np.nan)
# moving to GPU memory
Orlanski_bc_u_vector = cp.asarray(Orlanski_bc_u_list)
Orlanski_bc_v_vector = cp.asarray(Orlanski_bc_v_list)
# Orlanski_bc_p_vector = cp.asarray(Orlanski_bc_p_list)
# creating mask
Orlanski_mask_u = ~cp.isnan(Orlanski_bc_u_vector)
Orlanski_mask_v = ~cp.isnan(Orlanski_bc_v_vector)
# Orlanski_mask_p = ~cp.isnan(drich_bc_p_vector) 

# Store the length of the linear array 
Ne_BC_length = len(Ne_BC_u_list)
Drich_BC_length = len(drich_bc_u_list)
Orlanski_BC_lenght = len(Orlanski_bc_u_list)

#=========================================================== Geometry check ============================================================#
# Change the variable Z to see the bc's appended on u (u_mat), v (v_mat) and p (p_mat)
lowerlimit = 0
upperlimit = 1
Z = u_mat  # Example of initial Z
x = np.linspace(lowerlimit, upperlimit, Z.shape[1]) 
y = np.linspace(lowerlimit, upperlimit, Z.shape[0])
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
plt.title("Quick BC Check...")
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#=======================================================================================================================================#
#                                                       Some other important setups
#=======================================================================================================================================#

total_time_steps = 10000
del_t = 0.00001

del_h = 1.0/(nx-1)
print("△h = ",del_h)

Re = 500        # Set Reynolds number here
etr = 1000        # Set etr here (every time iteration)
RESTART = 0   # 1 (restart) 0 (no restart)
#========================================================================================================================================#
#                                                               END
#========================================================================================================================================#
#====== MOVING TO GPU=====#
u_old = cp.asarray(u_mat)
v_old = cp.asarray(v_mat)
p_old = cp.asarray(p_mat)

B_vector_sequence = []

u_p_cpu = u_mat.copy()           # u_velocity copy mesh
v_p_cpu = v_mat.copy()           # v_velocity copy mesh

u_p = cp.asarray(u_p_cpu)       # u_p moved to GPU MEMORY
v_p = cp.asarray(v_p_cpu)       # v_p moved to GPU MEMORY
    
u_copy = u_mat.copy()
v_copy = v_mat.copy() 

u_copy = cp.asarray(u_copy)     # u_copy/u_new moved to GPU MEMORY
v_copy = cp.asarray(v_copy)     # v_copy/v_new movd to GPU MEMORY

p_prime = p_mat.copy()
p_prime = cp.asarray(p_prime)   # p_prime grid moved to GPU MEMORY

#=======================================================================================================================================#
#                                                           SETUP RESTRAT FILES HERE
#=======================================================================================================================================#
if (RESTART == 1):

    start = 18501
    # =====================================
    # USER: Put file path here manually
    # =====================================

    file_path_u = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u/Time_stack_u_t0018500.npz"
    file_path_v = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v/Time_stack_v_t0018500.npz"
    file_path_p = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_p/Time_stack_p_t0018500.npz"

    # =====================================
    # Load NPZ file
    # =====================================

    data_u = np.load(file_path_u)
    data_v = np.load(file_path_v)
    data_p = np.load(file_path_p)

    print("Available arrays:", data_u.files)
    print("Available arrays:", data_v.files)
    print("Available arrays:", data_p.files)

    key_u = data_u.files[0]       # only one array expected
    key_v = data_v.files[0]       # only one array expected
    key_p = data_p.files[0]       # only one array expected

    U = data_u[key_u]
    V = data_v[key_v]
    P = data_p[key_p]

    u_old = U
    v_old = V
    p_old = P
    # moving to GPU memory 
    u_old = cp.asarray(u_mat)
    v_old = cp.asarray(v_mat)
    p_old = cp.asarray(p_mat)

else:
    start = 1
    pass

#======================================================================================================================================#
#                                                   3D - A matrix for batched system Interpolation
#======================================================================================================================================#
print("Building 3D interpolation matrix.....")
Interpolation_matrix_A = []
std = time.time()
for edge in range(0, len(interpolation_normal_dist), 1):
    for GhostNode in range(0, len(interpolation_normal_dist[edge]), 1):
        matrix =  np.eye(6)
        for idx in range(0, len(interpolation_normal_dist[edge][GhostNode]), 1):
            value = interpolation_normal_dist[edge][GhostNode][idx]
            for j in range(0,len(interpolation_normal_dist[edge][GhostNode]), 1):
                print(edge,GhostNode,idx,j, value,len(interpolation_normal_dist[edge][GhostNode]))
                matrix[idx][j] = value**j
        # print(matrix)
        # time.sleep(900)
        Interpolation_matrix_A.append(matrix)
etd = time.time()
print("size of 3D A matrix = ",len(Interpolation_matrix_A))
print("Time Taken to build 3D A matrix = ", etd - std)

#======================================================================================================================================#
#                                                   2D - B matrix for batched system Interpolation
#======================================================================================================================================#

#=======================================Linearizing Interpolation point coordinates============================#
coords_row = []
coords_col = []
for edge in range(0,len(interpolation_coords),1):
    for GhostNode in range(0,len(interpolation_coords[edge]),1):
        for InterplationIndex in range(0,6,1):
            i = InterplationIndex
            j = len(interpolation_coords[edge][GhostNode])-1
            if (i > j):
                coords_row.append(np.nan)
                coords_col.append(np.nan)
            else:
                print(edge,GhostNode,InterplationIndex,j)
                x_coord = interpolation_coords[edge][GhostNode][InterplationIndex][0]
                y_coord = interpolation_coords[edge][GhostNode][InterplationIndex][1]
                row_io = int(round((y_coord * conversion_factor),0))
                col_jo = int(round((x_coord * conversion_factor),0))

                coords_row.append(row_io)
                coords_col.append(col_jo)

#======================================Lineariing B row and coloumn ===========================#
b_row = []
b_col = []
for i in range(0, number_of_ghost_nodes ,1):
    for j in range(0,6,1): 
        b_row.append(i)
        b_col.append(j)
#==================================== converting the data into apprpriate form ===============#
b_row = np.array(b_row)
b_col = np.array(b_col)

coords_row = np.array(coords_row)
coords_col = np.array(coords_col)

coords_row_mask = ~np.isnan(coords_row)
coords_col_mask = ~np.isnan(coords_col)

# mask everything once at setup and convert float to int data type
coords_row_valid = coords_row[coords_row_mask].astype(np.int32)
coords_col_valid = coords_col[coords_col_mask].astype(np.int32)
print("length of coords_col_mask = ",cp.shape(coords_col_mask))
print("length of b_col = ", cp.shape(b_col))

b_row_valid  = b_row[coords_row_mask].astype(np.int32)
b_col_valid  = b_col[coords_col_mask].astype(np.int32)

b_row_valid = cp.asarray(b_row_valid)
b_col_valid = cp.asarray(b_col_valid)

coords_row_valid = cp.asarray(coords_row_valid)
coords_col_valid = cp.asarray(coords_col_valid)

# =============================================== 2D - B matrix built function ===========================#
Bzoo = cp.zeros((number_of_ghost_nodes,6), dtype = cp.float32)


def B_matrix_2d(phi):

    Bzoo[b_row_valid, b_col_valid] = phi[coords_row_valid ,coords_col_valid]
    
    return Bzoo
#===============================Linearizing r_mirror ======================#
r_mirror_vector = []
for edge in range (0, len(r_m), 1):
    for idr in range (0, len(r_m[edge]), 1):
        r_mirror_vector.append(r_m[edge][idr])

# ================ Transfering necessary vectors to GPU memory ===========#  
Interpolation_matrix_A = cp.asarray(Interpolation_matrix_A, dtype=cp.float32)
r_mirror_vector = cp.asarray(r_mirror_vector)
alpha_vector_u = cp.asarray(alpha_u)
alpha_vector_v = cp.asarray(alpha_v)
alpha_vector_p = cp.asarray(alpha_p)
beta_vecotr_u = cp.asarray(beta_u)
beta_vector_v = cp.asarray(beta_v)
beta_vector_p = cp.asarray(beta_p)
# ========================================================================#

def IB_GPU_interpolater (phi_field, r_mirror, drich_bc_val_vector, alpha_vector, beta_vector):

    
    matrix_A = Interpolation_matrix_A       # 3D matrix (collection of 2D 6x6 matrix stacked over one and another)
    phi_B = B_matrix_2d(phi_field)          # 2D matrix that is collection of phi field vector corresponding to each co-effecient matrix 
    # print(matrix_A[0])
    polynomial_vector_1 = cp.linalg.solve(matrix_A,phi_B)
    # print(polynomial_vector_1[0])
    # vlaue at mirror point; mirror_val is an 1D array of mirror values at all mirror points
    mirror_val = polynomial_vector_1[:,0] + polynomial_vector_1[:,1]*r_mirror + polynomial_vector_1[:,2]*(r_mirror**2) + polynomial_vector_1[:,3]*(r_mirror**3) + polynomial_vector_1[:,4]*(r_mirror**4) + polynomial_vector_1[:,5]*(r_mirror**5)

    mirror_val_grad = polynomial_vector_1[:,1] + 2*polynomial_vector_1[:,2]*(r_mirror**1) + 3*polynomial_vector_1[:,3]*(r_mirror**2) + 4*polynomial_vector_1[:,4]*(r_mirror**3) + 5*polynomial_vector_1[:,5]*(r_mirror**4)

    
    q_val  = -beta_vector * drich_bc_val_vector

    d = r_mirror

    D = d*(2*alpha_vector - beta_vector*d)

    a0 = ( -alpha_vector * (d**2) * mirror_val_grad  +  2 * d* alpha_vector * mirror_val  +  (d**2) * q_val) / D
        
    a1 = ( +beta_vector * (d**2) * mirror_val_grad  -  2 * beta_vector * d * mirror_val -  2 * d * q_val) / D

    a2 = ( (-beta_vector * d + alpha_vector) * mirror_val_grad  +  beta_vector * mirror_val  +  q_val) / D


    # for the moment let r_g = r_mirror
    r_g = r_mirror
    # phi_ghost = a2*(r_g**2) + a1*(r_g**1) + a0  # value at ghost node 
    phi_ghost = a2*(r_g**2) - a1*(r_g**1) + a0  # value at ghost node 

    Interpolated_Drich_BC_vector = phi_ghost    # Interpolated Drichilet BC vector
    Interpolated_Ne_BC_vector = mirror_val      # Interpolated Neuman BC vector
    
    return Interpolated_Drich_BC_vector, Interpolated_Ne_BC_vector, polynomial_vector_1,phi_B

# ============================================================================================#
# Note:  Currently in the solver only Neumann BCs are approximated, the drichilet BCs are not.
# ============================================================================================#

L_inf_change_list = []
iteration = []
#=======================================================================================================================================#
#                                                           SMAC BEGINS
#=======================================================================================================================================#

for t in range(start, total_time_steps, 1):
    
    print("=================================================================================================")
    print("itn = ",t,"/",total_time_steps)
    st = time.time() 
    # --- first & second interface indices generation ---
    if(t == start):
        coords_f = cp.asarray(first_interface, dtype=cp.float64)
        coords_s = cp.asarray(second_interface, dtype=cp.float64)
        coords_i = cp.asarray(inside_pt, dtype=cp.float64)
        io = cp.rint(coords_i[:,1] * conversion_factor).astype(cp.int32)
        jo = cp.rint(coords_i[:,0] * conversion_factor).astype(cp.int32)

        ipf = cp.rint(coords_f[:,1] * conversion_factor).astype(cp.int32)
        jpf = cp.rint(coords_f[:,0] * conversion_factor).astype(cp.int32)

        ips = cp.rint(coords_s[:,1] * conversion_factor).astype(cp.int32)
        jps = cp.rint(coords_s[:,0] * conversion_factor).astype(cp.int32)

    # --- velocities ---
    u = u_old[ipf, jpf]
    v = v_old[ipf, jpf]
    # print(u)

    # convection (u)
    du_dx_back = (u_old[ipf, jpf] - u_old[ipf, jpf-1]) / del_h
    du_dx_forw = (u_old[ipf, jpf+1] - u_old[ipf, jpf]) / del_h
    u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

    du_dy_back = (u_old[ipf, jpf] - u_old[ipf-1, jpf]) / del_h
    du_dy_forw = (u_old[ipf+1, jpf] - u_old[ipf, jpf]) / del_h
    v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

    Hu_conv_f = u_du_dx + v_du_dy

    # convection (v)
    dv_dx_back = (v_old[ipf, jpf] - v_old[ipf, jpf-1]) / del_h
    dv_dx_forw = (v_old[ipf, jpf+1] - v_old[ipf, jpf]) / del_h
    u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

    dv_dy_back = (v_old[ipf, jpf] - v_old[ipf-1, jpf]) / del_h
    dv_dy_forw = (v_old[ipf+1, jpf] - v_old[ipf, jpf]) / del_h
    v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

    Hv_conv_f = u_dv_dx + v_dv_dy

    # diffusion
    Hu_diffusion_f = (1/Re)*(u_old[ipf, jpf+1] + u_old[ipf, jpf-1] + u_old[ipf+1, jpf] + u_old[ipf-1, jpf] - 4*u_old[ipf, jpf])/(del_h**2)

    Hv_diffusion_f = (1/Re)*(v_old[ipf, jpf+1] + v_old[ipf, jpf-1] + v_old[ipf+1, jpf] + v_old[ipf-1, jpf] - 4*v_old[ipf, jpf])/(del_h**2)

    # update
    Up = u_old[ipf, jpf] + del_t*(-Hu_conv_f + Hu_diffusion_f)
    Vp = v_old[ipf, jpf] + del_t*(-Hv_conv_f + Hv_diffusion_f)

    u_p[ipf, jpf] = Up
    v_p[ipf, jpf] = Vp

    #---second interface---
    u = u_old[ips, jps]
    v = v_old[ips, jps]

    du_dx_back = (3*u_old[ips, jps] - 4*u_old[ips, jps-1] + u_old[ips, jps-2])/(2*del_h)
    du_dx_forw = (-3*u_old[ips, jps] + 4*u_old[ips, jps+1] - u_old[ips, jps+2])/(2*del_h)
    u_du_dx = u * cp.where(u >= 0, du_dx_back, du_dx_forw)

    du_dy_back = (3*u_old[ips, jps] - 4*u_old[ips-1, jps] + u_old[ips-2, jps])/(2*del_h)
    du_dy_forw = (-3*u_old[ips, jps] + 4*u_old[ips+1, jps] - u_old[ips+2, jps])/(2*del_h)
    v_du_dy = v * cp.where(v >= 0, du_dy_back, du_dy_forw)

    Hu_conv_s = u_du_dx + v_du_dy

    dv_dx_back = (3*v_old[ips, jps] - 4*v_old[ips, jps-1] + v_old[ips, jps-2])/(2*del_h)
    dv_dx_forw = (-3*v_old[ips, jps] + 4*v_old[ips, jps+1] - v_old[ips, jps+2])/(2*del_h)
    u_dv_dx = u * cp.where(u >= 0, dv_dx_back, dv_dx_forw)

    dv_dy_back = (3*v_old[ips, jps] - 4*v_old[ips-1, jps] + v_old[ips-2, jps])/(2*del_h)
    dv_dy_forw = (-3*v_old[ips, jps] + 4*v_old[ips+1, jps] - v_old[ips+2, jps])/(2*del_h)
    v_dv_dy = v * cp.where(v >= 0, dv_dy_back, dv_dy_forw)

    Hv_conv_s = u_dv_dx + v_dv_dy

    Hu_diffusion_s = (1/Re)*(u_old[ips,jps+1] + u_old[ips,jps-1] + u_old[ips+1,jps] + u_old[ips-1,jps] - 4*u_old[ips,jps])/(del_h**2)
    Hv_diffusion_s = (1/Re)*(v_old[ips,jps+1] + v_old[ips,jps-1] + v_old[ips+1,jps] + v_old[ips-1,jps] - 4*v_old[ips,jps])/(del_h**2)

    Up = u_old[ips,jps] + del_t*(-Hu_conv_s + Hu_diffusion_s)
    Vp = v_old[ips,jps] + del_t*(-Hv_conv_s + Hv_diffusion_s)

    u_p[ips,jps] = Up
    v_p[ips,jps] = Vp

    et = time.time()
    # print("time: ",et-st)
    
    u_interpolation_drich, u_interpolation_Ne, pv, phi_b = IB_GPU_interpolater (u_p, r_mirror_vector, drich_bc_u_vector, alpha_vector_u, beta_vecotr_u)
    v_interpolation_drich, v_interpolation_Ne, pv, phi_b = IB_GPU_interpolater (v_p, r_mirror_vector, drich_bc_v_vector, alpha_vector_v, beta_vector_v)
    

    # Applying interpolated neumann BC
    u_p[rgn[Ne_mask_u], cgn[Ne_mask_u]] = u_interpolation_Ne[Ne_mask_u]
    v_p[rgn[Ne_mask_v], cgn[Ne_mask_v]] = v_interpolation_Ne[Ne_mask_v]
    
    # Applying interpolated drichilet BC 
    # u_p[rgn[drich_mask_u],cgn[drich_mask_u]] = u_interpolation_drich[drich_mask_u]
    # v_p[rgn[drich_mask_v],cgn[drich_mask_v]] = v_interpolation_drich[drich_mask_v]

    # Applying drichilt bc as it is
    u_p[rgn[drich_mask_u],cgn[drich_mask_u]] = drich_bc_u_vector[drich_mask_u]
    v_p[rgn[drich_mask_v],cgn[drich_mask_v]] = drich_bc_v_vector[drich_mask_v]

    # Applying Orlanski BC
    u_p[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] = u_old[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] - (1*del_t/del_h)*(u_old[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] - u_old[rf[Orlanski_mask_u], cf[Orlanski_mask_u]])
    v_p[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] = v_old[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] - (1*del_t/del_h)*(v_old[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] - v_old[rf[Orlanski_mask_v], cf[Orlanski_mask_v]])
    print("===Up complete====")

#========================================================================================================================================#
#                                                Building [A][p'] = [B]  & ([A] is stored in csr format)
#========================================================================================================================================#
    # for moving and deforming bodies
    # inside_pt = time_based_inside_pt[t]     # here inside points which are going to be function of time are stored in time_based_inside_pt
    # ghost_node = time_based_ghost_node[t]   # here ghost points which are going to be function of time are stored in time_based_ghost_node
    if(t == start):
        u_p_cpu = cp.asnumpy(u_p)
        v_p_cpu = cp.asnumpy(v_p)
        p_old_cpu = cp.asnumpy(p_old) 

        sat = time.time()
        # Building co-efficient matrix A here

        row_csr = []
        col_csr = []
        value_csr = []

        B = []
        Total_number_of_nodes = len(variable_array)
        for i in range(0,Total_number_of_nodes,1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]

            # row, col transfrmation
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0)) 

            io_cpu=row
            jo_cpu=col 

            #================================================================building 5 point stencil==================================================================#
            # Find the indices of the neighboring points
            east = col+1
            west = col-1
            south = row-1
            north = row+1  
                    
            # Neighbor handling with safe check
            key_p = f'x{row}|{col}'
            key_east = f'x{row}|{east}'
            key_west = f'x{row}|{west}'
            key_south = f'x{south}|{col}'
            key_north = f'x{north}|{col}'
            #==========================================================================================================================================================#
            #================================================================local coefficient storage=================================================================#
            a_north =[]
            a_west = []
            a_p = [-4]
            a_east = []
            a_south = []

            a = [-4]
            b_e = []
            b_vector_data = []

            row_north = np.full(Total_number_of_nodes, np.nan)
            col_north = np.full(Total_number_of_nodes, np.nan)

            row_south = np.full(Total_number_of_nodes, np.nan)
            col_south = np.full(Total_number_of_nodes, np.nan)
            
            row_east = np.full(Total_number_of_nodes, np.nan)
            col_east = np.full(Total_number_of_nodes, np.nan)

            row_west = np.full(Total_number_of_nodes, np.nan)
            col_west = np.full(Total_number_of_nodes, np.nan)

            b_wall = []
            #==========================================================================================================================================================#
            if key_east in index_map:
                # append i once
                # append e/w/s/n varaible_ array index once
                # append the value once  
                a_east.append(1)
                b_vector_data.append(0)
                # print("east1")
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t =  round((east/conversion_factor),4)
                    y_t = round((row/conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            # print("easts2")
                            # a_p.append(1)
                            a_east.append(0)
                            #=================#
                            row_east[i] = row
                            col_east[i] = east
                            #=================#
                            expression = -1 * (del_h/Re) * (((u_p_cpu[io_cpu+1, jo_cpu] - u_p_cpu[io_cpu-1, jo_cpu])/(del_h**2)) + (u_p_cpu[io_cpu, jo_cpu+1] - u_p_cpu[io_cpu, jo_cpu-1])/(del_h**2))
                            b_wall.append(expression)
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            # print("east3")
                            pass

                        break
                    else:
                        pass
               
            if key_west in index_map:
                a_west.append(1)
                b_vector_data.append(0)
                print("west1")
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((west / conversion_factor),4)
                    y_t = round((row/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            print("west2")
                            # a_p.append(1)
                            a_west.append(0)
                            #=================#
                            row_west[i] = row
                            col_west[i] = west
                            #=================#
                            expression = +1 * (del_h/Re) * (((u_p_cpu[io_cpu+1, jo_cpu] - u_p_cpu[io_cpu-1, jo_cpu])/(del_h**2)) + (u_p_cpu[io_cpu, jo_cpu+1] - u_p_cpu[io_cpu, jo_cpu-1])/(del_h**2))
                            b_wall.append(expression)
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            # print("west3")
                            pass

                        break
                    else:
                        pass
                      
            if key_south in index_map:
                a_south.append(1)
                b_vector_data.append(0)
                # print("south1")
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((south/ conversion_factor),4)
                    target = (x_t,y_t)
                    # need to convert target back into x-y coordinate
                    # print("down",target,key_south,key_west)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        # print(";;;;")
                        ne_pos = ghost_nodes_list.index(current_sub_gn)      # this line tells which edge we are dealng with
                        if (Ne_BC_p[ne_pos] != "NCN"):                       # this implies a neuman condition exist 
                            b_e.append(Ne_BC_p[ne_pos])                      # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            print("south2")
                            # a_p.append(1)
                            a_south.append(0) 
                            #=================#
                            row_south[i] = south
                            col_south[i] = col
                            #=================#
                            expression = +1 * (del_h/Re) * (((v_p_cpu[io_cpu+1, jo_cpu] - v_p_cpu[io_cpu-1, jo_cpu])/(del_h**2)) + (v_p_cpu[io_cpu, jo_cpu+1] - v_p_cpu[io_cpu, jo_cpu-1])/(del_h**2))  
                            b_wall.append(expression)                     
                        else: 
                            b_e.append(0)
                            b_vector_data.append(0)
                            # print("south3")
                            pass
                         
                        break
                        
                    else:
                        pass
        
            if key_north in index_map:
                a_north.append(1)
                b_vector_data.append(0)
                # print("north1")
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((north/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            print("north2")
                            # a_p.append(1)
                            a_north.append(0)
                            #=================#
                            row_north[i] = north
                            col_north[i] = col
                            #=================#
                            expression = -1 * (del_h/Re) * (((v_p_cpu[io_cpu+1, jo_cpu] - v_p_cpu[io_cpu-1, jo_cpu])/(del_h**2)) + (v_p_cpu[io_cpu, jo_cpu+1] - v_p_cpu[io_cpu, jo_cpu-1])/(del_h**2))
                            b_wall.append(expression)
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            # print("north3")
                            pass

                        break
                    else:
                        pass
            

            if key_south in index_map:
                south_m = index_map[key_south]
                S = np.sum(a_south)
                if (S > 1e-2):
                    row_csr.append(i)   # row 
                    col_csr.append(south_m)  #col
                    value_csr.append(S) # value
            if key_west in index_map:
                west_m = index_map[key_west]
                W = np.sum(a_west)
                if (W > 1e-2):
                    row_csr.append(i)   # row 
                    col_csr.append(west_m)  #col
                    value_csr.append(W) # value
            if key_p in index_map:
                p_m = index_map[key_p]
                P = np.sum(a_p)
                if (-5 < P  < 1e-2):
                    row_csr.append(i)   # row 
                    col_csr.append(p_m)  #col
                    value_csr.append(P) # value
                    
            if key_east in index_map:         
                east_m = index_map[key_east]
                E = np.sum(a_east)
                if (E > 1e-2):
                    row_csr.append(i)   # row 
                    col_csr.append(east_m)  #col
                    value_csr.append(E) # value
            if key_north in index_map:
                north_m = index_map[key_north]
                N = np.sum(a_north)
                if (N > 1e-2):
                    row_csr.append(i)   # row 
                    col_csr.append(north_m)  #col
                    value_csr.append(N) # value


            Avg_b1 = (u_p_cpu[io_cpu,jo_cpu+1] - u_p_cpu[io_cpu,jo_cpu-1])/(2)
            Avg_b2 = (v_p_cpu[io_cpu+1,jo_cpu] - v_p_cpu[io_cpu-1,jo_cpu])/(2) 

            zeta = del_t/del_h

            const = (del_h**2/(del_t*del_h)) * ((Avg_b1  +  2*zeta*p_old_cpu[io_cpu,jo_cpu] - zeta*p_old_cpu[io_cpu,jo_cpu-1] - zeta*p_old_cpu[io_cpu,jo_cpu+1]) + (Avg_b2 + 2*zeta*p_old_cpu[io_cpu,jo_cpu] - zeta*p_old_cpu[io_cpu-1,jo_cpu] - zeta*p_old_cpu[io_cpu+1,jo_cpu]) ) + np.sum(b_wall)
                
            b_e.append(const)
            b_final = np.sum(b_e)
            B.append(b_final)
            B_vector_sequence.append(b_vector_data)

        size = (len(variable_array))
        value_csr = np.array(value_csr,dtype=np.float64)
        row_csr = np.array(row_csr,dtype=np.int32)
        col_csr = np.array(col_csr,dtype=np.int32)

        # For a more readable version (Kilobytes)
        print(f"Memory size: {row_csr.nbytes / 1024:.2f} KB")
        # For a more readable version (Kilobytes)
        print(f"Memory size: {col_csr.nbytes / 1024:.2f} KB")
        # For a more readable version (Kilobytes)
        print(f"Memory size: {value_csr.nbytes / 1024:.2f} KB")
        # time.sleep(900)
    
        B_vector_sequence_gpu = cp.asarray(B_vector_sequence)
        B_np = np.array(B, dtype=np.float32)
        print("B = ",B_np)
        print(np.any(B_np != 0))
        # time.sleep(900)
        row_north_gpu = cp.asarray(row_north)
        col_north_gpu = cp.asarray(col_north)

        row_south_gpu = cp.asarray(row_south)
        col_south_gpu = cp.asarray(col_south)

        row_east_gpu = cp.asarray(row_east)
        col_east_gpu = cp.asarray(col_east)

        row_west_gpu = cp.asarray(row_west)
        col_west_gpu = cp.asarray(col_west)

        row_north_mask = ~cp.isnan(row_north_gpu)
        col_north_mask = ~cp.isnan(col_north_gpu)

        row_south_mask = ~cp.isnan(row_south_gpu)
        col_south_mask = ~cp.isnan(col_south_gpu)

        row_east_mask = ~cp.isnan(row_east_gpu)
        col_east_mask = ~cp.isnan(col_east_gpu)

        row_west_mask = ~cp.isnan(row_west_gpu)
        col_west_mask = ~cp.isnan(col_west_gpu)


        row_north_gpu = row_north_gpu[~cp.isnan(row_north_gpu)].astype(cp.int32)
        col_north_gpu = col_north_gpu[~cp.isnan(col_north_gpu)].astype(cp.int32)

        row_south_gpu = row_south_gpu[~cp.isnan(row_south_gpu)].astype(cp.int32)
        col_south_gpu = col_south_gpu[~cp.isnan(col_south_gpu)].astype(cp.int32)

        row_east_gpu = row_east_gpu[~cp.isnan(row_east_gpu)].astype(cp.int32)
        col_east_gpu = col_east_gpu[~cp.isnan(col_east_gpu)].astype(cp.int32)

        row_west_gpu = row_west_gpu[~cp.isnan(row_west_gpu)].astype(cp.int32)
        col_west_gpu = col_west_gpu[~cp.isnan(col_west_gpu)].astype(cp.int32)

        zeros = cp.zeros(Total_number_of_nodes)
        zero_north = zeros
        zero_south = zeros
        zero_west = zeros
        zero_east = zeros 

        # time.sleep(900)
    if(t > start):

        zero_north[row_north_mask] = v_p[row_north_gpu, col_north_gpu]
        zero_south[row_south_mask] = v_p[row_south_gpu, col_south_gpu]
        zero_west[row_west_mask] = u_p[row_west_gpu, col_west_gpu]
        zero_east[row_east_mask] = u_p[row_east_gpu, col_east_gpu]

        beta_north_v = -1 * (del_h/Re) * zero_north
        beta_south_v = +1 * (del_h/Re) * zero_south
        beta_west_u = +1 * (del_h/Re) * zero_west
        beta_east_u = -1 * (del_h/Re) * zero_east

        # print(beta_north_v,beta_east_u,beta_west_u,beta_south_v)

        sbt = time.time()
        Avg_b1 = (u_p[io,jo+1] - u_p[io,jo-1])/(2)  # (Up[j+1] - Up[j-1])/2
        Avg_b2 = (v_p[io+1,jo] - v_p[io-1,jo])/(2)  # (Vp[i-1] - Vp[i-1])/2
        zeta = del_t/del_h
        const = (del_h**2/(del_t*del_h)) * ((Avg_b1  +  2*zeta*p_old[io,jo] - zeta*p_old[io,jo-1] - zeta*p_old[io,jo+1]) + (Avg_b2 + 2*zeta*p_old[io,jo] - zeta*p_old[io-1,jo] - zeta*p_old[io+1,jo])) + beta_north_v + beta_south_v + beta_east_u + beta_west_u
        b = B_vector_sequence_gpu
        b_sum = const + cp.sum(b)
        B_gpu_m = cp.asarray(b_sum, dtype=cp.float64)

    #===================================================================================================================================#
    #                                                    Using GMRES to solve [A][p'] = [B]
    #===================================================================================================================================#

    #============================================================
    # MOVE MATRIX TO GPU ONLY ONCE
    # ============================================================
    
    if t == start:
        # The matrix is converted into CSR form and moved ont GPU only once
        A_csr = sp.csr_matrix((value_csr, (row_csr, col_csr)), shape=(size, size))
        print(">>>>")
        print(A_csr.data.size)
        print(A_csr.indices.size)
        print(A_csr.indptr.size)
        print(len(B_np))
        # time.sleep(900)
        A_gpu = cusparse.csr_matrix(A_csr)
        B_gpu = cp.asarray(B_np)
    if t > start:
        B_gpu = cp.asarray(B_gpu_m,dtype=cp.float64)

    sgmres = time.time()
    solution_vector, info = splinalg.gmres(A_gpu,B_gpu, tol = 1e-3, restart=20, maxiter=10)     

    egmres = time.time()

    #==================================================================================================================================#
    #                                                               END
    #===================================================================================================================================#
    # if(t%etr < 1e-3):
    # print("itn = ",t,"/",total_time_steps)
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    print(solution_vector)
    print("INFO = ",info)

    p_prime[io,jo] = solution_vector

    p_interpolation_drich, p_interpolation_Ne, pv,phi_b = IB_GPU_interpolater(p_prime, r_mirror_vector, drich_bc_p_vector, alpha_vector_p, beta_vector_p)
    # Applying interpolated Neumann BCs
    p_prime[rgn[Ne_mask_p], cgn[Ne_mask_p]] = p_interpolation_Ne[Ne_mask_p]

    if (t % etr < 1e-3):
        a = p_interpolation_Ne[Ne_mask_p]
        b = p_prime[rf[Ne_mask_p], cf[Ne_mask_p]]
        
        change_Linf = cp.linalg.norm(a - b, ord=cp.inf)
        print("L_inf CHANGE = ",change_Linf)
        L_inf_change_list.append(change_Linf.item())
        iteration.append(t)

    # #====================================== To study interpolation results or backtrack error ======================================#
    # print(interpolation_normal_dist[2][0])
    # print("Local-Boundary Matrix A: ")
    # print(Interpolation_matrix_A[165])
    # print("phi_B: ",phi_b[165],len(phi_b[165]))
    # print("polynomial_vector: ",pv[165],len(pv[165]))
    # mask = cp.isnan(pv).any(axis=1)
    # indices = cp.where(cp.isnan(pv).any(axis=1))[0]
    # print(" ")
    # print(mask)
    # print("True Indices = ",indices)
    # # print("P_prime: ",p_interpolation_Ne[Ne_mask_p])

    # # Drichilet BCs
    # # p_prime[rgn[drich_mask_p],cgn[drich_mask_p]] = p_interpolation_drich[drich_mask_p]

    # print("P_prime: ",p_interpolation_drich[drich_mask_p])
    # if (t>0):
    #     # print(r_mirror_vector)
    #     location = 5
    #     rft = rf[location]
    #     cft = cf[location]
    #     rgr = rgn[location]
    #     cgc = cgn[location]
    #     print("?",rft,cft,rgr,cgc)
    #     y_real, x_real = cp.round(rft/conversion_factor, 4), cp.round(cft/conversion_factor,4)
    #     y_realg, x_realg = cp.round(rgr/conversion_factor, 4), cp.round(cgc/conversion_factor,4)
    #     print("Solver co-ordinates: ", rf[location], cf[location], " Actual co-ordiantes: ",x_real, y_real, x_realg, y_realg)
    #     print("p_at_first_interface: ",p_prime[rf[location], cf[location]])
    #     print("p_from_Interpolator: ",p_interpolation_Ne[location])
    #     print("")
    #     # print("phi_b = ", phi_b[1])
    #     # print("a_vector =", pv[18])
    #     # # Define your target value
    #     # target = 0.029580652713775635

    #     # # Find the indices where the values are close to the target
    #     # x = cp.where(cp.isclose(p_interpolation_Ne, target, atol=1e-18))
    #     # print("==> ",x)
    #     # print(p_interpolation_Ne[x])
    #     print("===========")
    #     # print(p_interpolation_Ne)
    #     # time.sleep(1)
    #=================================================================================================================================#

    # Applying Drichilet BCs as it is
    p_prime[rgn[drich_mask_p],cgn[drich_mask_p]] = drich_bc_p_vector[drich_mask_p]

    p_new = p_prime + p_old        #corrected pressure copy mesh     # p[n+1] = p' + p[n]

    # =================================================================================================================================#
    #                                                         Updating u[n+1] and v[n+1]
    # =================================================================================================================================#
    u_copy[ipf, jpf] = u_old[ipf, jpf] + del_t*(-Hu_conv_f + Hu_diffusion_f - ((p_new[ipf, jpf+1] - p_new[ipf, jpf-1])/(2*del_h)))
    v_copy[ipf, jpf] = v_old[ipf, jpf] + del_t*(-Hv_conv_f + Hv_diffusion_f - ((p_new[ipf+1, jpf] - p_new[ipf-1, jpf])/(2*del_h)))

    u_copy[ips, jps] = u_old[ips,jps] + del_t*(-Hu_conv_s + Hu_diffusion_s - ((p_new[ips, jps+1] - p_new[ips, jps-1])/(2*del_h)))
    v_copy[ips, jps] = v_old[ips,jps] + del_t*(-Hv_conv_s + Hv_diffusion_s - ((p_new[ips+1, jps] - p_new[ips-1, jps])/(2*del_h)))

    et = time.time()
  
    u_copy_interpolation_drich, u_copy_interpolation_Ne, pv, phi_b = IB_GPU_interpolater (u_copy, r_mirror_vector, drich_bc_u_vector, alpha_vector_u, beta_vecotr_u)
    v_copy_interpolation_drich, v_copy_interpolation_Ne, pv, phi_b = IB_GPU_interpolater (v_copy, r_mirror_vector, drich_bc_v_vector, alpha_vector_v, beta_vector_v)

    # Applying neumann BC
    u_copy[rgn[Ne_mask_u], cgn[Ne_mask_u]] = u_copy_interpolation_Ne[Ne_mask_u]
    v_copy[rgn[Ne_mask_v], cgn[Ne_mask_v]] = v_copy_interpolation_Ne[Ne_mask_v]
    
    # # Applying drichilet BC
    u_copy[rgn[drich_mask_u],cgn[drich_mask_u]] = drich_bc_u_vector[drich_mask_u]
    v_copy[rgn[drich_mask_v],cgn[drich_mask_v]] = drich_bc_v_vector[drich_mask_v]

    # # Orlanski BC
    # u_copy[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] = u_old[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] - (1*del_t/del_h)*(u_old[rgn[Orlanski_mask_u], cgn[Orlanski_mask_u]] - u_old[rf[Orlanski_mask_u], cf[Orlanski_mask_u]])
    # v_copy[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] = v_old[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] - (1*del_t/del_h)*(v_old[rgn[Orlanski_mask_v], cgn[Orlanski_mask_v]] - v_old[rf[Orlanski_mask_v], cf[Orlanski_mask_v]])

    p_old = p_new
    u_old = u_copy
    v_old = v_copy

    #===================================================================================================================================#
    #                                                                   END
    #===================================================================================================================================#

    #===========================SAVING DATA===============================#
    if t % etr < 1e-3:

        u_old_xt = cp.asnumpy(u_old)
        v_old_xt = cp.asnumpy(v_old)
        p_old_xt = cp.asnumpy(p_old)

        base_dir = r"D:/numerical computation/geometry meshing/Meshes"

        # ----------------------------
        # Create folders once (safe)
        # ----------------------------
        u_dir = os.path.join(base_dir, "Time_stack_u")
        v_dir = os.path.join(base_dir, "Time_stack_v")
        p_dir = os.path.join(base_dir, "Time_stack_p")

        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)

        print("@",t)
        tag = f"{t:07d}"   # zero padded timestep
        print("@",tag)

        # ----------------------------
        # Force numeric arrays (important)
        # ----------------------------
        u_clean = np.asarray(u_old_xt, dtype=np.float32)
        v_clean = np.asarray(v_old_xt, dtype=np.float32)
        p_clean = np.asarray(p_old_xt, dtype=np.float32)

        # ----------------------------
        # Save with UNIQUE filenames
        # ----------------------------
        u_path = os.path.join(u_dir, f"Time_stack_u_t{tag}.npz")
        v_path = os.path.join(v_dir, f"Time_stack_v_t{tag}.npz")
        p_path = os.path.join(p_dir, f"Time_stack_p_t{tag}.npz")

        np.savez(u_path, u=u_clean)
        np.savez(v_path, v=v_clean)
        np.savez(p_path, p=p_clean)

        print(f"✅ Saved timestep {t}")

#==============================================#
#               L_inf CHANGE CHECK 
#==============================================#
print("change_table",L_inf_change_list)
print(type(L_inf_change_list))
L_inf_change = cp.asnumpy(L_inf_change_list)
plt.plot(iteration,L_inf_change )
plt.scatter(iteration,L_inf_change)
plt.show()
#==============================================#

#=======================================================================================================================================#
#                                                               POST-PROCESSING
#=======================================================================================================================================#
u_old_xt = cp.asnumpy(u_old,dtype=cp.float32)
v_old_xt = cp.asnumpy(v_old,dtype=cp.float32)
p_old_xt = cp.asnumpy(p_old,dtype=cp.float32)

lowerlimit = 0
upperlimit = 1

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = u_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
ax.streamplot(X, Y, u_old, v_old, color= 'k', density = 1.5, linewidth=1)
plt.title("Lid Driven Cavity: Velocity Streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = v_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = p_old_xt  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.title("Lid Driven Cavity: Pressure plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    

