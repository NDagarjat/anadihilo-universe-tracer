import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import imageio
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIGURATION (NASA Dark Theme) ---
st.set_page_config(page_title="Anadihilo 3D Universe", layout="wide", page_icon="🌌")

# Custom CSS for Dark UI
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #00e5ff !important; }
    .stButton>button { border-radius: 20px; border: 1px solid #00e5ff; color: #00e5ff; background-color: transparent; }
    .stButton>button:hover { background-color: #00e5ff; color: #000; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Anadihilo Dynamics: 3D Universe Tracer")
st.markdown("**Strict PDF Implementation:** $a_j = \\frac{K}{P_j} \\sum \\frac{P_k}{r^2 + \\epsilon}$ | **Relation:** $\\Omega(Dg) = n / 0.8$")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("1. Global Settings")
    
    # Input Type Selection
    input_mode = st.radio("Input Mode (Equation 3)", ["Dagar Intensity (Dg)", "Systemic Boundary (n)"])
    
    # Universal Constants
    st.subheader("Scaling Constants")
    K_default = 3.98e14 # Solar System Scale
    K_val = st.number_input("Universal Constant (K)", value=K_default, format="%.2e", help="Adjust for Atomic vs Celestial scales")
    dt = st.number_input("Time Step (Seconds)", value=3600, step=60)
    steps = st.slider("Total Simulation Steps", 500, 20000, 5000)
    
    st.markdown("---")
    st.header("2. Body Configuration")
    
    bodies = []
    # Create inputs for 3 Bodies
    for i in range(1, 4):
        with st.expander(f"Body {i} Configuration", expanded=(i==1)):
            name = st.text_input(f"Name", value=f"Body {i}", key=f"n{i}")
            
            # Logic for Dg vs n
            if input_mode == "Systemic Boundary (n)":
                n_val = st.number_input(f"Boundary 'n' (meters)", value=1.0 if i>1 else 1000.0, format="%.4e", key=f"val{i}")
                # Formula: Dg = n / 0.8 (From PDF Eq 3)
                p_val = n_val / 0.8
                st.caption(f"Calculated Dg: {p_val:.4e}")
            else:
                p_val = st.number_input(f"Intensity 'Dg'", value=1.0 if i>1 else 1480000.0, format="%.4e", key=f"val{i}")
                # Formula: n = Dg * 0.8
                n_cal = p_val * 0.8
                st.caption(f"Implied Boundary n: {n_cal:.4e} m")

            # 3D Coordinates
            c1, c2, c3 = st.columns(3)
            px = c1.number_input(f"Pos X", value=0.0, format="%.2e", key=f"px{i}")
            py = c2.number_input(f"Pos Y", value=0.0, format="%.2e", key=f"py{i}")
            pz = c3.number_input(f"Pos Z", value=0.0, format="%.2e", key=f"pz{i}")
            
            v1, v2, v3 = st.columns(3)
            vx = v1.number_input(f"Vel X", value=0.0, format="%.2e", key=f"vx{i}")
            vy = v2.number_input(f"Vel Y", value=0.0, format="%.2e", key=f"vy{i}")
            vz = v3.number_input(f"Vel Z", value=0.0, format="%.2e", key=f"vz{i}")
            
            bodies.append({
                'name': name,
                'P': p_val, # Dagar Value used for physics
                'pos': np.array([px, py, pz], dtype=float),
                'vel': np.array([vx, vy, vz], dtype=float),
                'hist': []
            })

# --- PHYSICS ENGINE ---
def run_anadihilo_physics():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Epsilon (Singularity Fix) constant
    EPS_BASE = 1.0 
    
    for s in range(steps):
        # 1. Determine Effective Friction (Handover/Assimilation)
        # Copy original P values
        eff_P = [b['P'] for b in bodies]
        parents = [-1] * 3
        
        for i in range(3):
            for j in range(3):
                if i == j: continue
                # Assimilation Logic:
                # If neighbor (j) has higher P and is "close" relative to its influence
                if bodies[j]['P'] > bodies[i]['P']:
                    dist = np.linalg.norm(bodies[i]['pos'] - bodies[j]['pos'])
                    # Influence Zone approximation from P (Scaled)
                    # Using a safe large interaction threshold for handover logic detection
                    threshold = 1.0e10 # Approx 10 Million km (Tunable based on scale)
                    
                    if dist < threshold:
                        eff_P[i] = bodies[j]['P'] # Assimilate: Use Parent's Friction
                        parents[i] = j

        # 2. Calculate Accelerations (Eq 6)
        accs = []
        for j in range(3):
            acc_vec = np.zeros(3) # 3D Vector
            for k in range(3):
                if j == k: continue
                
                r_vec = bodies[k]['pos'] - bodies[j]['pos']
                r_sq = np.sum(r_vec**2)
                dist_val = np.sqrt(r_sq)
                
                if dist_val == 0: continue
                
                # Singularity Killer (Epsilon)
                # epsilon = 1 / (P_j + P_k)
                epsilon = 1.0 / (bodies[j]['P'] + bodies[k]['P'])
                
                # Friction Selection
                # If I have a parent 'k', I use MY intrinsic P for internal interaction
                # If I interact with 'k' (external), I use my Effective P
                if parents[j] == k:
                    friction = bodies[j]['P']
                else:
                    friction = eff_P[j]
                
                # Main Formula: a = (K / Friction) * (P_source / (r^2 + epsilon))
                force_mag = (K_val / friction) * (bodies[k]['P'] / (r_sq + epsilon))
                
                acc_vec += force_mag * (r_vec / dist_val)
            accs.append(acc_vec)
            
        # 3. Update State
        for idx, b in enumerate(bodies):
            b['vel'] += accs[idx] * dt
            b['pos'] += b['vel'] * dt
            
            # Logging (Every nth step to save memory, default every 5th)
            if s % 5 == 0:
                b['hist'].append(b['pos'].copy())
        
        # UI Update
        if s % (steps // 20) == 0:
            progress_bar.progress(s / steps)
            status_text.text(f"Computing Step {s}/{steps} (Anadihilo Grid Update...)")
            
    progress_bar.progress(100)
    status_text.success("Calculation Complete: Systemic Boundaries Resolved.")
    return bodies

# --- EXECUTION & VISUALIZATION ---
if st.button("🚀 EXECUTE SIMULATION", use_container_width=True):
    
    result_bodies = run_anadihilo_physics()
    
    # 1. 3D INTERACTIVE PLOT (Plotly)
    st.subheader("1. 3D Trajectory Visualization")
    
    fig = go.Figure()
    colors = ['#ffcc00', '#0099ff', '#00ff99'] # Gold, Blue, Green
    
    for idx, b in enumerate(result_bodies):
        hist = np.array(b['hist'])
        if len(hist) > 0:
            # Add Line Path
            fig.add_trace(go.Scatter3d(
                x=hist[:,0], y=hist[:,1], z=hist[:,2],
                mode='lines',
                name=f"{b['name']} Path",
                line=dict(color=colors[idx], width=4)
            ))
            # Add Final Position Marker
            fig.add_trace(go.Scatter3d(
                x=[hist[-1,0]], y=[hist[-1,1]], z=[hist[-1,2]],
                mode='markers',
                name=f"{b['name']} Core",
                marker=dict(size=8, color=colors[idx])
            ))

    fig.update_layout(
        scene = dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor="black"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="black",
        font=dict(color="white"),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. DATA DOWNLOAD (Full CSV)
    st.subheader("2. Download Coordinate Logs")
    
    data_dict = {'Step': range(len(result_bodies[0]['hist']))}
    for b in result_bodies:
        hist = np.array(b['hist'])
        data_dict[f"{b['name']}_X"] = hist[:,0]
        data_dict[f"{b['name']}_Y"] = hist[:,1]
        data_dict[f"{b['name']}_Z"] = hist[:,2]
        
    df = pd.DataFrame(data_dict)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Download Full 3D Data (CSV)",
        csv,
        "anadihilo_3d_log.csv",
        "text/csv",
        key='download-csv'
    )
    
    # 3. GIF GENERATION
    st.subheader("3. Generate Simulation GIF")
    st.info("Creating animation frames... This might take a moment.")
    
    # Create frames using Matplotlib (Static frames to GIF)
    frames = []
    max_frames = 60 # Limit frames for GIF to keep size low
    step_size = max(1, len(result_bodies[0]['hist']) // max_frames)
    
    for i in range(0, len(result_bodies[0]['hist']), step_size):
        fig_g, ax_g = plt.subplots(figsize=(5, 5), dpi=100)
        # Dark theme for GIF
        fig_g.patch.set_facecolor('black')
        ax_g.set_facecolor('black')
        
        # Plot trails up to current frame
        for idx, b in enumerate(result_bodies):
            hist = np.array(b['hist'])
            # 2D Projection (X-Y) for clarity in GIF, or specific view
            ax_g.plot(hist[:i,0], hist[:i,1], color=colors[idx], lw=1)
            if i > 0:
                ax_g.plot(hist[i-1,0], hist[i-1,1], 'o', color=colors[idx], markersize=6)
        
        ax_g.set_aspect('equal')
        ax_g.grid(True, alpha=0.2, color='white')
        ax_g.tick_params(colors='white')
        for spine in ax_g.spines.values(): spine.set_edgecolor('white')
        
        # Capture buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig_g)
        buf.seek(0)
        frames.append(imageio.v3.imread(buf))
        
    # Save GIF
    imageio.mimsave("simulation.gif", frames, fps=10, loop=0)
    
    with open("simulation.gif", "rb") as file:
        btn = st.download_button(
            label="🎬 Download Animation (GIF)",
            data=file,
            file_name="anadihilo_sim.gif",
            mime="image/gif"
        )
    
    st.image("simulation.gif", caption="2D Projection of 3D Orbit")

else:
    st.info("👈 Configure the bodies in the sidebar and click 'Execute Simulation'.")

