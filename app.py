import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo 3D Smooth", layout="centered", page_icon="🌌")

# Dark UI CSS
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #fff; }
    .stButton>button { 
        border: 2px solid #00e5ff; 
        color: #00e5ff; 
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover { 
        background-color: #00e5ff; 
        color: #000; 
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🌌 Anadihilo Universe Tracer")
st.caption("High-Performance 3D Engine | Anadihilo Physics")

# --- CONFIGURATION (EXPANDER) ---
with st.expander("⚙️ Configure Simulation (Click to Expand)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Physics Accuracy (Steps)", 1000, 20000, 10000)
    with col2:
        speed = st.select_slider("Animation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
        
    st.markdown("---")
    # Body Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Sun**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
    with c2:
        st.markdown("🔵 **Earth**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X (m)", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vy (m/s)", value=29780.0, key="vy2")
    with c3:
        st.markdown("⚪ **Moon**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        x3 = st.number_input("X (m)", value=x2 + 3.844e8, format="%.2e", key="x3")
        vy3 = st.number_input("Vy (m/s)", value=vy2 + 1022.0, key="vy3")

# --- PHYSICS ENGINE (Heavy Calculation) ---
@st.cache_data # Cache result to prevent re-calculation on every interaction
def calculate_physics(p1, p2, p3, x2, vy2, x3, vy3, steps):
    dt = 3600
    bodies = [
        {'name': 'Sun', 'P': p1, 'p': np.array([0.0, 0.0, 0.0]), 'v': np.array([0.0, 0.0, 0.0]), 'hist': []},
        {'name': 'Earth', 'P': p2, 'p': np.array([x2, 0.0, 0.0]), 'v': np.array([0.0, vy2, 0.0]), 'hist': []},
        {'name': 'Moon', 'P': p3, 'p': np.array([x3, 0.0, 0.0]), 'v': np.array([0.0, vy3, 0.0]), 'hist': []}
    ]
    EPSILON = 1.0
    
    # Run Loop
    for s in range(steps):
        # Handover Logic
        eff_P = [b['P'] for b in bodies]
        parents = [-1] * 3
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if bodies[j]['P'] > bodies[i]['P']:
                    if np.linalg.norm(bodies[i]['p'] - bodies[j]['p']) < 1.0e10:
                        eff_P[i] = bodies[j]['P']
                        parents[i] = j
        
        # Forces
        accs = []
        for j in range(3):
            a = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r = bodies[k]['p'] - bodies[j]['p']
                d = np.linalg.norm(r)
                if d == 0: continue
                fric = bodies[j]['P'] if parents[j] == k else eff_P[j]
                force = (3.98e14 / fric) * (bodies[k]['P'] / (d**2 + (EPSILON/(bodies[j]['P']+bodies[k]['P']))))
                a += force * (r/d)
            accs.append(a)
            
        # Update & Log
        for idx, b in enumerate(bodies):
            b['v'] += accs[idx] * dt
            b['p'] += b['v'] * dt
            if s % 10 == 0: # Log every 10th step only (Optimization 1)
                b['hist'].append(b['p'].copy())
                
    return bodies

# --- VISUALIZATION ENGINE ---
if st.button("🚀 LAUNCH SIMULATION", type="primary"):
    with st.spinner("Calculating Physics & Preparing Animation..."):
        data = calculate_physics(p1, p2, p3, x2, vy2, x3, vy3, steps)
        
        # Animation Speed Settings
        frame_skip = 10 if speed == "Slow" else 20 if speed == "Normal" else 40
        duration = 50 if speed == "Fast" else 100
        
        tab1, tab2, tab3 = st.tabs(["🪐 Macro View (Sun-Earth)", "🌑 Micro View (Moon-Earth)", "📥 Data & GIF"])
        
        # --- TAB 1: MACRO SYSTEM ---
        with tab1:
            st.write("**Solar System Scale**")
            
            # Prepare Data Arrays
            sun_h = np.array(data[0]['hist'])
            earth_h = np.array(data[1]['hist'])
            moon_h = np.array(data[2]['hist'])
            
            # Create Figure
            fig1 = go.Figure()
            
            # 1. Static Trails (Always Visible)
            fig1.add_trace(go.Scatter3d(x=sun_h[:,0], y=sun_h[:,1], z=sun_h[:,2], mode='lines', line=dict(color='#ffcc00', width=4), name='Sun Orbit'))
            fig1.add_trace(go.Scatter3d(x=earth_h[:,0], y=earth_h[:,1], z=earth_h[:,2], mode='lines', line=dict(color='#0099ff', width=4), name='Earth Orbit'))
            
            # 2. Dynamic Markers (Moving Bodies) - Initial Pos
            fig1.add_trace(go.Scatter3d(x=[sun_h[0,0]], y=[sun_h[0,1]], z=[sun_h[0,2]], mode='markers', marker=dict(color='#ffcc00', size=20), name='Sun'))
            fig1.add_trace(go.Scatter3d(x=[earth_h[0,0]], y=[earth_h[0,1]], z=[earth_h[0,2]], mode='markers', marker=dict(color='#0099ff', size=10), name='Earth'))
            
            # 3. Frames for Animation (Optimized)
            frames = []
            for k in range(0, len(sun_h), frame_skip):
                frames.append(go.Frame(data=[
                    go.Scatter3d(x=[sun_h[k,0]], y=[sun_h[k,1]], z=[sun_h[k,2]]), # Update Sun Pos
                    go.Scatter3d(x=[earth_h[k,0]], y=[earth_h[k,1]], z=[earth_h[k,2]])  # Update Earth Pos
                ]))
            
            fig1.frames = frames
            fig1.update_layout(
                scene=dict(bgcolor="black", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                paper_bgcolor="black", margin=dict(l=0,r=0,t=0,b=0), height=500,
                updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
            )
            st.plotly_chart(fig1, use_container_width=True)

        # --- TAB 2: MICRO SYSTEM ---
        with tab2:
            st.write("**Moon Orbit (Relative to Earth)**")
            
            # Relative Calculations
            rel_path = moon_h - earth_h
            
            fig2 = go.Figure()
            
            # Earth (Static Center)
            fig2.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=25, color='blue'), name='Earth'))
            
            # Moon Trail (Static)
            fig2.add_trace(go.Scatter3d(x=rel_path[:,0], y=rel_path[:,1], z=rel_path[:,2], mode='lines', line=dict(color='white', width=5), name='Orbit Path'))
            
            # Moon Marker (Dynamic)
            fig2.add_trace(go.Scatter3d(x=[rel_path[0,0]], y=[rel_path[0,1]], z=[rel_path[0,2]], mode='markers', marker=dict(size=12, color='white'), name='Moon'))
            
            # Frames
            frames_micro = []
            for k in range(0, len(rel_path), frame_skip):
                frames_micro.append(go.Frame(data=[
                    go.Scatter3d(x=[0], y=[0], z=[0]), # Earth stays
                    go.Scatter3d(x=rel_path[:,0], y=rel_path[:,1], z=rel_path[:,2]), # Path stays
                    go.Scatter3d(x=[rel_path[k,0]], y=[rel_path[k,1]], z=[rel_path[k,2]]) # Update Moon
                ], traces=[0, 1, 2])) # Update specific traces

            fig2.frames = frames_micro
            fig2.update_layout(
                scene=dict(bgcolor="black", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                paper_bgcolor="black", margin=dict(l=0,r=0,t=0,b=0), height=500,
                updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play Orbit", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.info("Agar animation ruk jaye, to 'Animation Speed' ko 'Fast' karke Run karein.")

        # --- TAB 3: GIF & CSV ---
        with tab3:
            # CSV Download
            df = pd.DataFrame({'Step': range(len(sun_h))})
            df['Sun_X'], df['Sun_Y'], df['Sun_Z'] = sun_h[:,0], sun_h[:,1], sun_h[:,2]
            df['Earth_X'], df['Earth_Y'], df['Earth_Z'] = earth_h[:,0], earth_h[:,1], earth_h[:,2]
            df['Moon_X'], df['Moon_Y'], df['Moon_Z'] = moon_h[:,0], moon_h[:,1], moon_h[:,2]
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Data", csv, "anadihilo_sim.csv", "text/csv")
            
            # Optimized GIF Generator
            st.divider()
            if st.button("🎬 Generate Smooth GIF"):
                with st.spinner("Rendering Frames (This takes ~10s)..."):
                    plt.switch_backend('Agg')
                    frames = []
                    # Use relative path for better visual
                    step_sz = max(1, len(rel_path) // 60) # Limit to 60 frames for GIF
                    
                    for i in range(0, len(rel_path), step_sz):
                        f, ax = plt.subplots(figsize=(4,4), facecolor='black')
                        ax.set_facecolor('black')
                        # Draw Trail
                        ax.plot(rel_path[:i,0], rel_path[:i,1], color='white', lw=1.5, alpha=0.7)
                        # Draw Earth
                        ax.plot(0,0, 'o', color='#0099ff', ms=12)
                        # Draw Moon
                        if i > 0:
                            ax.plot(rel_path[i-1,0], rel_path[i-1,1], 'o', color='white', ms=6)
                        
                        ax.axis('off')
                        ax.set_aspect('equal')
                        
                        # Buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close(f)
                        buf.seek(0)
                        frames.append(imageio.v3.imread(buf))
                    
                    imageio.mimsave("orbit.gif", frames, fps=15, loop=0)
                    st.success("GIF Ready!")
                    st.image("orbit.gif", width=300)
                    with open("orbit.gif", "rb") as file:
                        st.download_button("⬇️ Download GIF", file, "orbit.gif", "image/gif")
