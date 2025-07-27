import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../particle_orbit.csv")
plt.plot(df['x'], df['y'])
plt.axis('equal')
plt.title("Orbit of Particle 0")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.grid(True)
plt.savefig("orbit_test.png")