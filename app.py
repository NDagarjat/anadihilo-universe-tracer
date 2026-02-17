import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio.v3 as iio
import io

# --- PAGE CONFIG (NASA Theme) ---
st.set_page_config(page_title="Anadihilo Dynamics Core", layout="wide", page_icon="🛰️")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    div.stButton > button { 
        background-color: #00e5ff; color: #000; font-weight: bold; border-radius: 5px; border: none;
    }
    div.stButton > button:hover { background-color: #ffffff; }
    h1, h2, h3 { color: #00e5ff !important; font-family: 'Courier New', monospace; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #111; border-radius: 5px; color: #fff; }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ ANADIHILO DYNAMICS: ORBITAL TRACER")
st.markdown("**Status:** Online | **Engine:** High-Precision Physics | **Resolution:** 50k Steps")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    # Physics Controls
    steps = st.slider("Calculation Steps (Accuracy)", 10000, 100000, 50000, step=5000)
    speed_factor = st.select_slider("Animation Speed", options=["Precision (Slow)", "Normal", "Hyper (Fast)"], value="Normal")
    
    st.divider()
    
    # Body 1: Sun
    st.subheader("🟡 Sun (Anchor)")
    p1 = st.number_input("P (Dg)", value=1480000.0, format="%.1f", key="p1")
    
    # Body 2: Earth
    st.subheader("🔵 Earth (Parent)")
    p2 = st.number_input("P (Dg)", value=4.44, format="%.2f", key="p2")
    x2 = st.number_input("X Position (m)", value=1.496e11, format="%.3e", key="x2")
    vy2 = st.number_input("Vy Velocity (m/s)", value=29780.0, format="%.1f", key="vy2")
    
    # Body 3: Moon
    st.subheader("⚪ Moon (Child)")
    p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
    # Auto-set relative to Earth
    def_x3 = x2 + 3.844e8
    def_vy3 = vy2 + 1022.0
    x3 = st.number_input("X Position (m)", value=def_x3, format="%.3e", key="x3")
    vy3 = st.number_input("Vy Velocity (m/s)", value=def_vy3, format="%.1f", key="vy3")

# --- PHYSICS ENGINE (Optimized) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Constants
    K_VAL = 3.98e14
    EPSILON_BASE = 1.0
    DT = 3600 # 1 Hour
    
    # Initialization
    bodies = [
        {'name': 'Sun', 'P': p1, 'p': np.array([0.0, 0.0, 0.0]), 'v': np.array([0.0, 0.0, 0.0]), 'hist': np.zeros((steps, 3))},
        {'name': 'Earth', 'P': p2, 'p': np.array([x2, 0.0, 0.0]), 'v': np.array([0.0, vy2, 0.0]), 'hist': np.zeros((steps, 3))},
        {'name': 'Moon', 'P': p3, 'p': np.array([x3, 0.0, 0.0]), 'v': np.array([0.0, vy3, 0.0]), 'hist': np.zeros((steps, 3))}
    ]
    
    # Pre-calculate constants to save loop time
    P_vals = np.array([b['P'] for b in bodies])
    
    # Main Loop (Vectorized where possible)
    for s in range(steps):
        # Store History (Direct Numpy Assignment is faster)
        for i in range(3):
            bodies[i]['hist'][s] = bodies[i]['p']
            
        # 1. Handover Logic (Assimilation)
        eff_P = P_vals.copy()
        parents = [-1, -1, -1]
        
        # Check pairs for assimilation
        # Earth(1) vs Moon(2)
        r_em = bodies[2]['p'] - bodies[1]['p']
        dist_em = np.linalg.norm(r_em)
        if dist_em < 1.0e10: # 10 Million km Zone
            if P_vals[1] > P_vals[2]: # Earth > Moon
                eff_P[2] = P_vals[1]
                parents[2] = 1
        
        # 2. Acceleration Calculation
        accs = np.zeros((3, 3)) # 3 Bodies, 3 Dimensions
        
        for j in range(3):
            for k in range(3):
                if j == k: continue
                
                r_vec = bodies[k]['p'] - bodies[j]['p']
                r_sq = np.sum(r_vec**2)
                d = np.sqrt(r_sq)
                
                # Formula Application
                eps = 1.0 / (P_vals[j] + P_vals[k])
                
                # Friction: Internal (with Parent) -> Own P, External -> Effective P
                friction = P_vals[j] if parents[j] == k else eff_P[j]
                
                mag = (K_VAL / friction) * (P_vals[k] / (r_sq + eps))
                accs[j] += mag * (r_vec / d)
        
        # 3. Integration (Update)
        for i in range(3):
            bodies[i]['v'] += accs[i] * DT
            bodies[i]['p'] += bodies[i]['v'] * DT
            
    return bodies

# --- VISUALIZATION HELPERS ---
def generate_gif(data_rel, skip=50):
    # Generates GIF automatically in memory
    frames = []
    # Plot setup
    plt.style.use('dark_background')
    
    # Limit frames to keep GIF size manageable (max 100 frames)
    total_len = len(data_rel)
    actual_skip = max(skip, total_len // 80)
    
    for i in range(0, total_len, actual_skip):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.axis('off') # Cleaner look
        
        # Trail
        ax.plot(data_rel[:i, 0], data_rel[:i, 1], color='white', lw=1, alpha=0.6)
        # Earth (Center)
        ax.plot(0, 0, 'o', color='#0099ff', ms=8)
        # Moon (Current)
        if i > 0:
            ax.plot(data_rel[i-1, 0], data_rel[i-1, 1], 'o', color='gray', ms=4)
            
        # Limits (Auto-scale to keep Moon in frame)
        max_range = np.max(np.abs(data_rel)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))
        
    # Convert to GIF bytes
    gif_bytes = io.BytesIO()
    iio.imwrite(gif_bytes, frames, format='GIF', duration=100, loop=0)
    gif_bytes.seek(0)
    return gif_bytes

# --- MAIN EXECUTION ---
if st.button("🚀 INITIALIZE & RUN", use_container_width=True):
    
    with st.spinner(f"Computing {steps} Physics Steps & Rendering Graphics... (Please Wait)"):
        # 1. Run Physics
        results = run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3)
        
        # Prepare Data for Plotting
        sun_h = results[0]['hist']
        earth_h = results[1]['hist']
        moon_h = results[2]['hist']
        rel_moon = moon_h - earth_h # Moon relative to Earth
        
        # 2. Auto-Generate GIF (Background)
        gif_data = generate_gif(rel_moon)
        
        # 3. Create Plots
        
        # Animation Settings (Auto Speed Control)
        # Downsample data for browser performance (Show max 500 frames)
        anim_skip = max(1, steps // 500) 
        if speed_factor == "Precision (Slow)": anim_skip = max(1, steps // 800)
        if speed_factor == "Hyper (Fast)": anim_skip = max(1, steps // 300)
        
        duration = 20 # ms per frame
        
        # --- 2D GRAPH (TOP VIEW) ---
        fig_2d = go.Figure()
        
        # Earth (Center)
        fig_2d.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='#0099ff'), name='Earth'))
        # Moon Path
        fig_2d.add_trace(go.Scatter(x=rel_moon[:,0], y=rel_moon[:,1], mode='lines', line=dict(color='white', width=1), opacity=0.5, name='Orbit Trace'))
        # Moon Marker (Animated)
        fig_2d.add_trace(go.Scatter(x=[rel_moon[0,0]], y=[rel_moon[0,1]], mode='markers', marker=dict(size=8, color='white'), name='Moon'))
        
        frames_2d = [go.Frame(data=[go.Scatter(x=[0], y=[0]), go.Scatter(x=rel_moon[:,0], y=rel_moon[:,1]), go.Scatter(x=[rel_moon[k,0]], y=[rel_moon[k,1]])]) 
                     for k in range(0, steps, anim_skip)]
        
        fig_2d.frames = frames_2d
        fig_2d.update_layout(
            title="2D Top View (Relative Orbit)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
            height=500,
            updatemenus=[dict(type="buttons", buttons=[dict(label="▶ PLAY 2D", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
        )

        # --- 3D GRAPH (NASA STYLE) ---
        fig_3d = go.Figure()
        
        # Static Trails
        fig_3d.add_trace(go.Scatter3d(x=sun_h[:,0], y=sun_h[:,1], z=sun_h[:,2], mode='lines', line=dict(color='#ffcc00', width=2), name='Sun Trace'))
        fig_3d.add_trace(go.Scatter3d(x=earth_h[:,0], y=earth_h[:,1], z=earth_h[:,2], mode='lines', line=dict(color='#0099ff', width=2), name='Earth Trace'))
        # Markers
        fig_3d.add_trace(go.Scatter3d(x=[sun_h[0,0]], y=[sun_h[0,1]], z=[sun_h[0,2]], mode='markers', marker=dict(size=20, color='#ffcc00'), name='Sun'))
        fig_3d.add_trace(go.Scatter3d(x=[earth_h[0,0]], y=[earth_h[0,1]], z=[earth_h[0,2]], mode='markers', marker=dict(size=10, color='#0099ff'), name='Earth'))
        
        frames_3d = []
        for k in range(0, steps, anim_skip):
            frames_3d.append(go.Frame(data=[
                go.Scatter3d(), go.Scatter3d(), # Skip updating lines
                go.Scatter3d(x=[sun_h[k,0]], y=[sun_h[k,1]], z=[sun_h[k,2]]),
                go.Scatter3d(x=[earth_h[k,0]], y=[earth_h[k,1]], z=[earth_h[k,2]])
            ]))
            
        fig_3d.frames = frames_3d
        fig_3d.update_layout(
            title="3D Solar System View (Rotatable)",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='black'),
            paper_bgcolor='black', font=dict(color='white'),
            height=500,
            updatemenus=[dict(type="buttons", buttons=[dict(label="▶ PLAY 3D", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
        )

    # --- LAYOUT DISPLAY ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🪐 2D Top View (Stable)")
        st.plotly_chart(fig_2d, use_container_width=True)
        st.success("✅ GIF Auto-Generated")
        st.download_button("⬇️ Download Animation (GIF)", gif_data, file_name="anadihilo_orbit.gif", mime="image/gif", use_container_width=True)
        
    with col2:
        st.subheader("🌌 3D Interactive View")
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # CSV Data
        df = pd.DataFrame({'Step': range(steps)})
        df['Sun_X'], df['Sun_Y'] = sun_h[:,0], sun_h[:,1]
        df['Earth_X'], df['Earth_Y'] = earth_h[:,0], earth_h[:,1]
        df['Moon_X'], df['Moon_Y'] = moon_h[:,0], moon_h[:,1]
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Trajectory Data (CSV)", csv, "anadihilo_data.csv", "text/csv", use_container_width=True)

else:
    st.info("👋 Ready to Calculate. Set parameters in the sidebar and click 'INITIALIZE & RUN'.")
