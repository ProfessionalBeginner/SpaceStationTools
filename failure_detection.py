#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import datetime

warnings.filterwarnings('ignore')
plt.style.use('bmh') 

class LifeSupportMonitor:
    def __init__(self):
        self.ml_model = IsolationForest(
            contamination=0.1, 
            random_state=42, 
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
        # Format: (Mean, Std_Dev, Hard_Min, Hard_Max)
        self.specs = {
            'co2_scrubber': (95.0, 2.0, 85.0, 105.0),
            'o2_gen':       (2500.0, 50.0, 2300.0, 2700.0),
            'pressure':     (50.66, 0.5, 48.0, 53.0),
            'voltage':      (1.8, 0.1, 1.4, 2.2),
            'temp':         (45.0, 3.0, 35.0, 60.0)
        }

    def get_simulated_data(self, hours=720):
        data = []
        keys = list(self.specs.keys())
        
        for h in range(hours):
            # Base sensor noise
            row = [np.random.normal(self.specs[k][0], self.specs[k][1]) for k in keys]
            
            # Circadian load simulation
            shift_load = np.sin(2 * np.pi * h / 24) * 0.5
            row[0] += shift_load       
            row[1] += shift_load * 100 
            
            data.append(row)
            
        return np.array(data)

    def prep_features(self, raw_data):
        df = pd.DataFrame(raw_data, columns=list(self.specs.keys()))
        
        delta = df.diff(periods=24).fillna(0)
        smooth_trend = delta.ewm(span=24).mean()
        z_score = (smooth_trend - smooth_trend.mean()) / (smooth_trend.std() + 1e-6)

        return np.hstack([df.values, smooth_trend.values, z_score.values])

    def train(self, history):
        print(f"[*] System Init: Training on {len(history)} hours of historic data...")
        context_data = self.prep_features(history)
        self.scaler.fit(context_data)
        self.ml_model.fit(self.scaler.transform(context_data))

    def detect_ml(self, incoming_stream):
        context = self.prep_features(incoming_stream)
        scaled = self.scaler.transform(context)
        return self.ml_model.predict(scaled)

    def detect_threshold(self, incoming_stream):
        results = []
        keys = list(self.specs.keys())
        
        for row in incoming_stream:
            is_anomaly = 1
            for i, val in enumerate(row):
                if val < self.specs[keys[i]][2] or val > self.specs[keys[i]][3]:
                    is_anomaly = -1
                    break
            results.append(is_anomaly)
            
        return np.array(results)

    def inject_failure(self, clean_data, fail_type, start_idx):
        broken = clean_data.copy()
        mask = np.zeros(len(broken), dtype=bool)
        
        if fail_type == 'slow_leak':
            # Gradual pressure decay
            for i in range(start_idx, len(broken)):
                leak = -0.015 * (i - start_idx) 
                broken[i, 2] += leak
                mask[i] = True
                
        elif fail_type == 'voltage_spike':
            # Intermittent spikes
            for i in range(start_idx, len(broken)):
                if np.random.random() < 0.2:
                    broken[i, 3] += np.random.uniform(1.0, 3.0)
                    mask[i] = True
                    
        return broken, mask

    def get_stats(self, preds, mask, start_idx):
        relevant_preds = preds[start_idx:]
        alarms = np.where(relevant_preds == -1)[0]
        
        if len(alarms) == 0:
            return None, 0.0 # Missed detection
            
        react_time = alarms[0] # Hours to first alarm
        
        caught = np.sum((preds == -1) & mask)
        total_broken = np.sum(mask)
        accuracy_during_failure = (caught / total_broken * 100) if total_broken > 0 else 0
        
        return react_time, accuracy_during_failure

def visualize_reliability(results_db):
    """Generates reliability plots and saves to disk"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Reaction Time Reliability (Box Plot)
    plot_data = []
    labels = []
    
    for scenario, data in results_db.items():
        if data['ml_times']:
            plot_data.append(data['ml_times'])
            labels.append(f"{scenario}\n(AI)")
        if data['leg_times']:
            plot_data.append(data['leg_times'])
            labels.append(f"{scenario}\n(Legacy)")

    ax1.boxplot(plot_data, labels=labels, patch_artist=True)
    ax1.set_title('Reaction Time Variance (Lower is Better)')
    ax1.set_ylabel('Hours to Detect')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Detection Consistency (Bar Chart)
    scenarios = list(results_db.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    ai_rates = [np.mean(results_db[s]['ml_rates']) for s in scenarios]
    leg_rates = [np.mean(results_db[s]['leg_rates']) for s in scenarios]
    
    # Error bars for standard deviation
    ai_err = [np.std(results_db[s]['ml_rates']) for s in scenarios]
    leg_err = [np.std(results_db[s]['leg_rates']) for s in scenarios]

    ax2.bar(x - width/2, ai_rates, width, label='AI Model', yerr=ai_err, capsize=5)
    ax2.bar(x + width/2, leg_rates, width, label='Legacy', yerr=leg_err, capsize=5)
    
    ax2.set_title('Mean Detection Accuracy (+/- SD)')
    ax2.set_ylabel('Accuracy %')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save Logic
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sim_reliability_{ts}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n[*] Graph saved successfully: {filename}")
    
    plt.show()

def monte_carlo_sim(n_iters=150):
    print(f"\n--- MONTE CARLO RELIABILITY DIAGNOSTIC (N={n_iters}) ---")
    
    sys = LifeSupportMonitor()
    
    # Pre-train system once
    history = sys.get_simulated_data(1500)
    sys.train(history)

    scenarios = [
        ('Pressure Leak', 'slow_leak'),
        ('Voltage Surge', 'voltage_spike')
    ]
    
    FAIL_START = 800
    SIM_LEN = 1200
    
    # Database for plotting
    results_db = {} 

    print(f"\n{'SCENARIO':<20} | {'SYS':<6} | {'MEAN (h)':<10} | {'SD (h)':<8} | {'CONFIDENCE':<12}")
    print("-" * 75)
    
    for name, key in scenarios:
        stats = {
            'ml_times': [], 'ml_rates': [], 'ml_misses': 0,
            'leg_times': [], 'leg_rates': [], 'leg_misses': 0
        }
        
        for _ in range(n_iters):
            test_stream = sys.get_simulated_data(SIM_LEN)
            bad_stream, truth_mask = sys.inject_failure(test_stream, key, FAIL_START)
            
            # --- AI Detection ---
            ml_preds = sys.detect_ml(bad_stream)
            t_ml, r_ml = sys.get_stats(ml_preds, truth_mask, FAIL_START)
            
            if t_ml is not None:
                stats['ml_times'].append(t_ml)
            else:
                stats['ml_misses'] += 1
            stats['ml_rates'].append(r_ml)
            
            # --- Legacy Detection ---
            leg_preds = sys.detect_threshold(bad_stream)
            t_leg, r_leg = sys.get_stats(leg_preds, truth_mask, FAIL_START)
            
            if t_leg is not None:
                stats['leg_times'].append(t_leg)
            else:
                stats['leg_misses'] += 1
            stats['leg_rates'].append(r_leg)

        def print_row(sys_name, times, rates, misses):
            if times:
                mean_t = np.mean(times)
                std_t = np.std(times)
                mean_r = np.mean(rates)
                
                # Confidence: (N - Misses) / N
                reliability = ((n_iters - misses) / n_iters) * 100
                conf_str = f"{reliability:.1f}% ({int(mean_r)}% Acc)"
                
                print(f"{name:<20} | {sys_name:<6} | {mean_t:>6.1f}     | {std_t:>6.1f}   | {conf_str}")
            else:
                print(f"{name:<20} | {sys_name:<6} | {'FAILED':>6}     | {'N/A':>6}   | 0.0%")

        print_row("AI", stats['ml_times'], stats['ml_rates'], stats['ml_misses'])
        print_row("LEGACY", stats['leg_times'], stats['leg_rates'], stats['leg_misses'])
        print("-" * 75)
        
        results_db[name] = stats

    visualize_reliability(results_db)

if __name__ == "__main__":
    monte_carlo_sim()