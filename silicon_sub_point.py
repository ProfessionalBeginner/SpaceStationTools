from scipy.optimize import brentq 
import numpy as np 

def vapor_pressure_eq(T,A,B,C,D,target_log_p):
    if T <= 0:
        return float('inf')
    return -(A/T) + B + (C*np.log10(T)) + 10**-3*D*T 

target_log_p = -13  # Lunar vacuum: 10^-13 atm

data = {'A': 17250, 'B': -15.97, 'C': 6.403, 'D': -0.5281,}
T_min = 1
T_max = 3533 #T_boil
try:
    T_sub = brentq(vapor_pressure_eq, T_min, T_max,
                  args=(data['A'], data['B'], data['C'], data['D'], target_log_p))
    print(f"Silicon: {T_sub:.1f} K ({T_sub-273.15:.1f}°C)")
except ValueError:
    print(f"Silicon: No solution in range!")