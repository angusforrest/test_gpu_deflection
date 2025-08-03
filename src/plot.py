import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("../particle0_steps_with_time.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(df["x"], df["y"], df["z"])
ax.set_title("3D Orbit of p0")
ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_zlabel("z (AU)")
ax.grid(True)
plt.savefig("orbit_test_3d.png")

# 2d projection
# fig2, ax2 = plt.subplots(figsize=(6, 6))
# sc = ax2.scatter(df["x"].values, df["y"].values, c=df["time"].values, s=3, linewidths=0)
# cb = plt.colorbar(sc, ax=ax2)
# cb.set_label("time")
# ax2.set_title("2D Projection")
# ax2.set_xlabel("x (AU)")
# ax2.set_ylabel("y (AU)")
# ax2.set_aspect("equal", adjustable="box")
# ax2.grid(True)
# plt.savefig("orbit_test_2d.png", dpi=300)


# # ---------- Energy ----------
# df_energy = pd.read_csv("../energy.csv")

# # relative error vs initial energy
# e0 = df_energy["energy"].iloc[0]
# rel_error = abs((df_energy["energy"] - e0) / e0)
# plt.figure()
# plt.semilogy(df_energy["step"], rel_error)
# plt.xlabel("Step")
# plt.ylabel("|(E(t)-E0)/E0|")
# plt.title("Relative energy error over time")
# plt.grid(True)
# plt.savefig("energy_error.png")


# # Parallel computation check
# df_energy_parallel = pd.read_csv("../energy_parallel.csv")
# assert (df_energy["step"] == df_energy_parallel["step"]).all(), (
#     "Steps don't align"
# )

# # overlaid
# e0 = df_energy["energy"].iloc[0]
# rel_seq = abs((df_energy["energy"] - e0) / e0)
# rel_par = abs((df_energy_parallel["energy"] - e0) / e0)
# plt.figure()
# plt.semilogy(df_energy["step"], rel_seq, label="Sequential", linewidth=1.2)
# plt.semilogy(
#     df_energy_parallel["step"],
#     rel_par,
#     label="Parallel",
#     linestyle="--",
#     linewidth=1.2,
# )
# plt.xlabel("Step")
# plt.ylabel("|(E(t)-E0)/E0|")
# plt.title("Relative Energy Error")
# plt.legend()
# plt.grid(True)
# plt.savefig("energy_comparison.png")

# # abs. diff
# energy_diff = abs(df_energy["energy"] - df_energy_parallel["energy"])
# plt.figure()
# plt.plot(df_energy["step"], energy_diff)
# plt.xlabel("Step")
# plt.ylabel("|E_seq - E_par|")
# plt.title("Absolute Energy Difference")
# plt.grid(True)
# plt.savefig("energy_diff.png")
