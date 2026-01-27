from scipy.optimize import brentq 
import numpy as np 

def vapor_pressure_eq(T,A,B,C,D,target_log_p):
    if T <= 0:
        return float('inf')
    return A + (B/T) + (C*np.log10(T)) + D/(T**3) - target_log_p

elements = {
    'Na': {'A': 5.298, 'B': -5603, 'C': 0, 'D': 0, 'T_melt': 371},
    'Mg': {'A': 8.489, 'B': -7813, 'C': -0.8253, 'D': 0, 'T_melt': 923},
    'Al': {'A': 9.459, 'B': -17342, 'C': -0.7927, 'D': 0, 'T_melt': 933},
    'Ca': {'A': 10.127, 'B': -9517, 'C': -1.403, 'D': 0, 'T_melt': 1112},
    'Ti': {'A': 11.925, 'B': -24991, 'C': -1.3376, 'D': 0, 'T_melt': 1930},
    'Mn': {'A': 12.805, 'B': -15097, 'C': -1.7896, 'D': 0, 'T_melt': 1519},
    'Fe': {'A': 7.1, 'B': -21723, 'C': 0.4536, 'D': -0.5846, 'T_melt': 1808},
    
}

target_log_p = -13  # Lunar vacuum: 10^-13 atm

for element, data in elements.items():
    T_min = 1
    T_max = data['T_melt']  
    try:
        T_sub = brentq(vapor_pressure_eq, T_min, T_max,
                      args=(data['A'], data['B'], data['C'], data['D'], target_log_p))
        print(f"{element}: {T_sub:.1f} K ({T_sub-273.15:.1f}°C)")
    except ValueError:
        print(f"{element}: No solution in range!")