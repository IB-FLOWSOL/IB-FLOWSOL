import numpy as np
import matplotlib.pyplot as plt

# =====================================
# USER: Put file path here manually
# =====================================
# -----------------To access main storage files (in mesh folder)-------------------------------------
file_path_u = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u/Time_stack_u_t0200000.npz"
file_path_v = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v/Time_stack_v_t0200000.npz"
file_path_p = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_p/Time_stack_p_t0200000.npz"
# #------------------To access files in result folder--------------------------------------------------
# file_path_u = r"D:/numerical computation/Results/Hyper FLOWSOL_with GMRES/Lid driven cavity/Re_400/u/Time_stack_u_t181000.npz"
# file_path_v = r"D:/numerical computation/Results/Hyper FLOWSOL_with GMRES/Lid driven cavity/Re_400/v/Time_stack_v_t181000.npz"
# file_path_p = r"D:/numerical computation/Results/Hyper FLOWSOL_with GMRES/Lid driven cavity/Re_400/p/Time_stack_p_t181000.npz"
# "D:\numerical computation\Results\SNH23001\Lid driven cavity\Re_400 SNH23001\v\Time_stack_v_t820200.npz"
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

# print("Loaded:", key)
# print("Raw shape:", F.shape)
# print("Dtype:", F.dtype)

# =====================================
# Handle stacked or 2D data automatically
# =====================================

if U.ndim == 3:
    print("Detected stacked data → using first snapshot.")
    U = U[0]              # shape becomes (ny, nx)

elif U.ndim != 2:
    raise ValueError(f"Unsupported array shape: {U.shape}")

ny, nx = U.shape
print("Final field shape:", U.shape)

# =====================================
# Grid (edit physical domain if needed)
# =====================================

# Grid index coordinates
x = np.arange(nx)
y = np.arange(ny)

# If you want physical coordinates instead, uncomment and edit:
# xmin, xmax = 0.0, 36.0
# ymin, ymax = 0.0, 21.0
# x = np.linspace(xmin, xmax, nx)
# y = np.linspace(ymin, ymax, ny)

# =====================================
# Velocity Plotting
# =====================================
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(8, 6))
# -------- Filled contours --------
cf = ax.contourf(X, Y, U, 25, cmap="coolwarm")      # some cmaps = 'viridis' , 'coolwarm' , 'PuBu' , 'jet'.
plt.colorbar(cf, ax=ax, label=key_u)
ax.streamplot(X, Y, U, V, color= 'k', density = 3, linewidth= 0.5)
# -------- Formatting --------
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.grid(False)
# plt.tight_layout()
plt.show()

# =====================================
# Pressure Plotting
# =====================================
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(8, 6))
# -------- Filled contours --------
cf = ax.contourf(X, Y, P, 25, cmap="jet")
plt.colorbar(cf, ax=ax, label=key_u)
# -------- Isobars (contour lines) --------
cs = ax.contour(X, Y, P, 120, colors="black", linewidths=0.6)
ax.clabel(cs, fontsize=8)

# -------- Formatting --------
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.grid(False)
# plt.tight_layout()
plt.show()

# print(U)
u_velocity_1 = []
u_velocity_2 = []
u_velocity_3 = []
u_ana = []
y = np.arange(0,1+0.0125,0.0125)
for row in range(0,81,1):
    u_velocity_1.append(U[row][40])

# for pipe flow
for row in range(0,51,1):
    u_analytical = (6 * 1 / (50**2)) * (50 * row - row**2)
    u_ana.append(u_analytical)

#======== Re=100 =========#
x_ghia= np.array([
    1.00000,
    0.84123,
    0.78871,
    0.73722,
    0.68717,
    0.23151,
    0.00332,
   -0.13641,
   -0.20581,
   -0.21090,
   -0.15662,
   -0.10150,
   -0.06434,
   -0.04775,
   -0.04192,
   -0.03717,
    0.00000
])
#=========== Re 400 ==============#
x_ghia = np.array([
    1.00000,
    0.75837,
    0.68439,
    0.61756,
    0.55892,
    0.29093,
    0.16256,
    0.02135,
   -0.11477,
   -0.17119,
   -0.32726,
   -0.24299,
   -0.14612,
   -0.10338,
   -0.09266,
   -0.08186,
    0.00000
])

#============ Re 1000 =========#

#============ Re 3200 =========#

y_ghia = np.array([
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000
])

plt.plot(u_velocity_1,y,label = "Our Work")
plt.plot(x_ghia,y_ghia,label="Ghia & Ghia 1982")
# plt.plot(u_ana,y,label = "Analytical result")
plt.title("Re = 400 and Grid-size = 81 x 81")
plt.legend()
plt.grid(True)
plt.show()


