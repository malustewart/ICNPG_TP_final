import numpy as np
from soa_model import SOAParameters, calc_gain_matrix, solve_gain
import numba as nb
from scipy.optimize import root_scalar
import PSO

WAVELENGTH = 1.55e-6

# Reasonable-ish SOA parameters
params = SOAParameters(
    tau_sp=1e-9,
    n0=1e24,
    Gamma=0.3,
    a=3e-20,
    alpha_int=1000,      # 1000 1/m
    vg=8e7,
    W=2e-6,
    d=0.2e-6,
    L=500e-6,
)

gain_map = np.load('measurement/processed/soa_2d_gain_map.npz')



Is = gain_map["soa_current_mA"] * 1e-3
Pins = gain_map["input_power_W"]
gain = gain_map["gain_linear"]

# fix dataset with numerical errors at edges that include nan
Pin_range = slice(1,len(Pins) - 1)

gain = gain[:, Pin_range]
Pins = Pins[Pin_range]
S0s = params.get_S0_from_P(Pins, WAVELENGTH)

Gs = calc_gain_matrix(S0s, Is, params)

print(PSO.cost(gain, Gs))
