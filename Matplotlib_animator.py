import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_cfd(u_folder=None,
                v_folder=None,
                mode="u",          # "u", "stream", or "both"
                interval=50):

    # -----------------------------
    # 1. Collect files
    # -----------------------------
    def extract_time(filename):
        match = re.search(r't(\d+)', filename)
        return int(match.group(1)) if match else -1

    if mode in ["u", "both"]:
        if u_folder is None:
            raise ValueError("u_folder required for mode 'u' or 'both'")
        u_files = sorted(
            [f for f in os.listdir(u_folder) if f.endswith(".npz")],
            key=extract_time
        )

    if mode in ["stream", "both"]:
        if u_folder is None or v_folder is None:
            raise ValueError("Both u_folder and v_folder required for streamlines.")
        u_files = sorted(
            [f for f in os.listdir(u_folder) if f.endswith(".npz")],
            key=extract_time
        )
        v_files = sorted(
            [f for f in os.listdir(v_folder) if f.endswith(".npz")],
            key=extract_time
        )

    if mode == "both":
        nframes = min(len(u_files), len(v_files))
    else:
        nframes = len(u_files)

    # -----------------------------
    # 2. Load first frame
    # -----------------------------
    u0 = np.load(os.path.join(u_folder, u_files[0]))["u"]

    if mode in ["stream", "both"]:
        v0 = np.load(os.path.join(v_folder, v_files[0]))["v"]

    ny, nx = u0.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y)

    # -----------------------------
    # 3. Setup Figure
    # -----------------------------
    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(15, 5))

    # ---- U Plot ----
    if mode in ["u", "both"]:
        if mode == "both":
            im = ax1.imshow(u0, origin='lower', cmap='jet')
            fig.colorbar(im, ax=ax1)
            ax1.set_title("u field")
        else:
            im = ax1.imshow(u0, origin='lower', cmap='jet')
            fig.colorbar(im)
            ax1.set_title("u field")

    # ---- Streamlines ----
    if mode in ["stream", "both"]:
        if mode == "both":
            ax_stream = ax2
        else:
            ax_stream = ax1

        ax_stream.streamplot(X, Y, u0, v0,
                             density=1.2,
                             linewidth=0.8)
        ax_stream.set_title("Streamlines")

    fig.suptitle(u_files[0])

    # -----------------------------
    # 4. Update function
    # -----------------------------
    def update(frame):

        u = np.load(os.path.join(u_folder, u_files[frame]))["u"]

        if mode in ["stream", "both"]:
            v = np.load(os.path.join(v_folder, v_files[frame]))["v"]

        # Update u
        if mode in ["u", "both"]:
            im.set_array(u)

        # Update streamlines
        if mode in ["stream", "both"]:
            ax_stream.cla()
            ax_stream.streamplot(X, Y, u, v,
                                 density=2.2,
                                 linewidth=0.8)
            ax_stream.set_title("Streamlines")

        fig.suptitle(u_files[frame])

        return []

    # -----------------------------
    # 5. Animate
    # -----------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=interval,
        blit=False
    )

    plt.tight_layout()
    plt.show()


# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    #===================================== From Result folder============================================#
    # u_folder = r"D:/numerical computation/Results/Hyper FLOWSOL_with GMRES/Backward facing step/Re_500/u"
    # v_folder = r"D:/numerical computation/Results/Hyper FLOWSOL_with GMRES/Backward facing step/Re_500/v"
    #===================================== From main storage ============================================#
    u_folder = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u"
    v_folder = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v"

    # Choose mode: "u", "stream", "both"
    animate_cfd(u_folder=u_folder,
                v_folder=v_folder,
                mode="u",
                interval=50)
