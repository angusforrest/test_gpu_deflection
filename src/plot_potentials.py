import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

bulge = pd.read_csv("../sphericalcutoff.csv")
disk = pd.read_csv("../miyamoto_nagai.csv")
halo = pd.read_csv("../navarro_frenk_white.csv")

R = bulge["x"].to_numpy(dtype=float)
aRb = bulge["ax"].to_numpy(dtype=float)
aRd = disk["ax"].to_numpy(dtype=float)
aRh = halo["ax"].to_numpy(dtype=float)

mask = R > 0.0
R = R[mask]
aRb = aRb[mask]
aRd = aRd[mask]
aRh = aRh[mask]

# v_c = sqrt(R * |a_R|))
vc_b = np.sqrt(R * np.abs(aRb))
vc_d = np.sqrt(R * np.abs(aRd))
vc_h = np.sqrt(R * np.abs(aRh))

R0 = 1.0
# i0 = np.argmin(np.abs(R - R0))
# vc0_total = np.sqrt(R[i0] * np.abs(aRb[i0] + aRd[i0] + aRh[i0]))

x_n = R / R0
# y_b = vc_b / vc0_total
# y_d = vc_d / vc0_total
# y_h = vc_h / vc0_total

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(x_n, vc_d, "--", linewidth=2.0, color="black", label="Disk")
plt.plot(x_n, vc_h, ":", linewidth=2.0, color="green", label="Halo")
# plt.plot(x_n, y_b,  '-', linewidth=2.0, color='blue', label='Bulge')
plt.xlim(0, 4)
plt.xlabel(r"$R/R_0$")
plt.ylabel(r"$v_c(R)/v_c(R_0)$")
plt.legend(loc="upper right")
plt.savefig("potentials.png")
plt.show()
