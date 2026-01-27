import pandas as pd

def simulate_sequential_distillation(t_initial_k: float = 300.0):
    # Masses in kg
    batch_masses = {
        "Na": 0.0033,
        "Mg": 0.0480,
        "Ca": 0.0860,
        "Mn": 0.0016,
        "Al": 0.0712,
        "Fe": 0.1178,
        "Ti": 0.0462
        # Si (0.2266) is excluded as it is scraped off at the end
    }

    # 2. Properties Dictionary
    element_data = {
        "Na": {"T_sub": 306.2,  "dH_sub_kJ": 4713,  "Cp": 1227.925},
        "Mg": {"T_sub": 404.0,  "dH_sub_kJ": 6166,  "Cp": 1023.205},
        "Ca": {"T_sub": 491.8,  "dH_sub_kJ": 4450,  "Cp": 647.000},
        "Mn": {"T_sub": 730.0,  "dH_sub_kJ": 4944,  "Cp": 477.300},
        "Al": {"T_sub": 861.4,  "dH_sub_kJ": 12231, "Cp": 921.096},
        "Fe": {"T_sub": 1012.1, "dH_sub_kJ": 7449,  "Cp": 460.548},
        "Ti": {"T_sub": 1201.2, "dH_sub_kJ": 9874,  "Cp": 544.284}
    }

    sorted_elements = sorted(batch_masses.keys(), key=lambda x: element_data[x]["T_sub"])

    current_temp = t_initial_k
    total_energy_j = 0.0
    results = []

    print(f"{'Step':<10} | {'Temp Range (K)':<15} | {'Mass Heated (kg)':<15} | {'Energy (J)':<15}")
    print("-" * 65)

    # 4. Sequential Loop
    for target in sorted_elements:
        props = element_data[target]
        target_mass = batch_masses[target]
        target_temp = props["T_sub"]
        
        # Calculate current active mass (everything remaining in the pot)
        current_alloy_mass = sum(batch_masses.values())
        
        # Calculate current C_avg (Weighted average of what is currently in the pot)
        current_c_avg = 0.0
        for elem, mass in batch_masses.items():
            if current_alloy_mass > 0:
                frac = mass / current_alloy_mass
                current_c_avg += frac * element_data[elem]["Cp"]
        
        # A. Sensible Heat (Heating the WHOLE remaining mixture to next T)
        delta_t = target_temp - current_temp
        if delta_t < 0: delta_t = 0 # Should not happen if sorted correctly
        
        e_sensible = current_alloy_mass * current_c_avg * delta_t
        
        # B. Latent Heat (Subliming ONLY the target)
        e_latent = (target_mass * props["dH_sub_kJ"]) * 1000
        
        step_energy = e_sensible + e_latent
        total_energy_j += step_energy
        
        # Log results
        results.append({
            "Element": target,
            "T_start": current_temp,
            "T_end": target_temp,
            "Mass_Alloy": current_alloy_mass,
            "Step_Energy_J": step_energy
        })
        
        print(f"{target:<10} | {current_temp:.1f} -> {target_temp:.1f} | {current_alloy_mass:.4f}          | {step_energy:,.0f}")

        # Update System State for next step
        # 1. The temp is now at the sublimation point of the element we just removed
        current_temp = target_temp 
        # 2. Remove the element from the batch (it has turned to gas)
        del batch_masses[target]

    return total_energy_j, pd.DataFrame(results)

# --- Run Simulation ---
total_j, df_results = simulate_sequential_distillation()

print("-" * 65)
print(f"Total Distillation Energy: {total_j/1000:,.2f} kJ")