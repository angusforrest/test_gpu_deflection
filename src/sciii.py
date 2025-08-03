import scipy.special as special
import numpy


def _mass(R):
    R = numpy.array(R)
    out = numpy.ones_like(R)
    out = (
        2.0
        * numpy.pi
        * R ** (3.0 - 1.8)
        / (1.5 - 1.8 / 2.0)
        * special.hyp1f1(
            1.5 - 1.8 / 2.0,
            2.5 - 1.8 / 2.0,
            -((R / (1.9 / 8.0)) ** 2.0),
        )
    )
    return out


rs = numpy.linspace(0, 4, 201)
print(rs[50])
c2 = (1.9 / 8.0) ** 2
r2c2s = [r**2 / c2 for r in rs]
acc = _mass(rs) / rs**2
print(f"unnormalised acceleration {acc}")
print(f"normalised acceleration {0.05 * acc / acc[50]}")
print(f"vel {numpy.sqrt(rs * (0.05 * acc / acc[50]))}")
print(0.05 / acc[50])
nacc = 0.05 * acc / acc[50]
output = numpy.stack([rs, nacc])
numpy.savetxt("../pysphericalcutoff.csv", output.T, delimiter=",")
# print(numpy.sqrt(0.05 * _mass(rs) / rs / (numpy.sqrt(_mass(rs) / rs**2)[50])))
# print(2 * np.pi * sc.gamma(0.6) * (1 - sc.gammaincc(0.6, r2c2s)))
# print(
#     0.15
#     * np.sqrt(rs * 2 * np.pi * sc.gamma(0.6) * (1 - sc.gammaincc(0.6, r2c2s)))
#     / np.sqrt(rs * 2 * np.pi * sc.gamma(0.6) * (1 - sc.gammaincc(0.6, r2c2s)))[
#         50
#     ]
# )
