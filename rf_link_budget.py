import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# --- CONSTANTS ---
k_B = 1.380649e-23
c = 2.998e8
AU = 1.496e11

def calculate_link(mode, p_tx, d_tx, d_rx, dist_au, bandwidth_mhz):
    dist_m = dist_au * AU
    
    if mode == 'RF':
        freq = 32e9 
        wavelength = c / freq
        # Efficiency 0.6 for RF dish
        eta = 0.6 
        # Atmospheric loss (Rain/Air) ~3 dB
        atm_loss_db = 3.0
    else: # OPTICAL
        wavelength = 1550e-9 # 1550nm Infrared
        freq = c / wavelength
        # Efficiency 0.3 for Optical 
        eta = 0.3
        # Atmospheric loss (Clouds/Air) ~6 dB 
        atm_loss_db = 6.0

    # 1. Gains
    g_tx_linear = eta * (np.pi * d_tx / wavelength)**2
    g_rx_linear = eta * (np.pi * d_rx / wavelength)**2
    
    # 2. Path Loss
    path_loss_linear = (4 * np.pi * dist_m / wavelength)**2
    
    # 3. Received Power
    atm_loss_linear = 10**(atm_loss_db/10)
    p_rx = (p_tx * g_tx_linear * g_rx_linear) / (path_loss_linear * atm_loss_linear)
    
    if mode == 'RF':
        T_sys = 40.0 # Cryo RF
    else:
        T_sys = 10000.0 

    bandwidth_hz = bandwidth_mhz * 1e6
    noise = k_B * T_sys * bandwidth_hz
    
    snr = p_rx / noise
    
    # 5. Capacity
    if snr > 0:
        cap = bandwidth_hz * np.log2(1 + snr)
    else:
        cap = 0
        
    # 6. Spot Size at Earth (Beam Diameter)
    # Theta = 1.22 * lambda / D
    theta = 1.22 * wavelength / d_tx
    spot_diameter_km = (theta * dist_m) / 1000.0
    
    return p_rx, snr, cap, spot_diameter_km

fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.9)
ax.axis('off')

title_text = ax.text(0.5, 1.05, "RF vs Optical Link Budget", ha='center', fontsize=20, fontweight='bold')
result_text = ax.text(0.5, 0.6, "", ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=1", fc="white"))

axcolor = 'lightgoldenrodyellow'
ax_ptx = plt.axes([0.25, 0.35, 0.5, 0.03], facecolor=axcolor)
ax_dtx = plt.axes([0.25, 0.30, 0.5, 0.03], facecolor=axcolor)
ax_drx = plt.axes([0.25, 0.25, 0.5, 0.03], facecolor=axcolor)
ax_dist = plt.axes([0.25, 0.20, 0.5, 0.03], facecolor=axcolor)
ax_bw = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)

s_ptx = Slider(ax_ptx, 'Tx Power (W)', 1, 500, valinit=50)
s_dtx = Slider(ax_dtx, 'Tx Aperture (m)', 0.1, 5.0, valinit=0.5)
s_drx = Slider(ax_drx, 'Rx Aperture (m)', 0.5, 40.0, valinit=10.0)
s_dist = Slider(ax_dist, 'Dist (AU)', 0.1, 10.0, valinit=5.2)
s_bw = Slider(ax_bw, 'Bandwidth (MHz)', 10, 50000, valinit=1000) # Up to 50 GHz

ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(ax_radio, ('RF (Ka-Band)', 'Optical (Laser)'))

def update(val):
    mode = 'RF' if radio.value_selected == 'RF (Ka-Band)' else 'OPTICAL'
    
    p = s_ptx.val
    dtx = s_dtx.val
    drx = s_drx.val
    d = s_dist.val
    bw = s_bw.val
    
    p_rx, snr, cap, spot = calculate_link(mode, p, dtx, drx, d, bw)
    
    p_rx_dbm = 10*np.log10(p_rx*1000) if p_rx > 0 else -999
    
    if cap > 1e9: rate_str = f"{cap/1e9:.2f} Gbps"
    elif cap > 1e6: rate_str = f"{cap/1e6:.2f} Mbps"
    else: rate_str = f"{cap/1e3:.2f} Kbps"
    
    result_text.set_text(
        f"--- MODE: {mode} ---\n\n"
        f"Tx Power: {p:.1f} W  |  Dish: {dtx:.1f} m\n"
        f"Received Power: {p_rx_dbm:.2f} dBm\n"
        f"Beam Spot Size at Earth: {spot:.0f} km\n\n"
        f"MAX SPEED: {rate_str}\n"
    )
    
    if snr < 1:
        result_text.set_bbox(dict(facecolor='#ffcccc', edgecolor='red', boxstyle='round,pad=1'))
    else:
        result_text.set_bbox(dict(facecolor='#ccffcc', edgecolor='green', boxstyle='round,pad=1'))

    fig.canvas.draw_idle()

s_ptx.on_changed(update)
s_dtx.on_changed(update)
s_drx.on_changed(update)
s_dist.on_changed(update)
s_bw.on_changed(update)
radio.on_clicked(update)

# ooh pretty sliders :)

update(0)
plt.show()