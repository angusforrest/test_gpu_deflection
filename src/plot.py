import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("../particle_orbit.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'])
ax.set_title("3D Orbit of p0")
ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_zlabel("z (AU)")
ax.grid(True)
plt.savefig("orbit_test_3d.png")
