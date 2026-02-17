import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Server Backend Fix
import matplotlib.pyplot as plt
import imageio
import io

# --- PAGE CONFIG (NASA Theme) ---
st.set_page_config(page_title="Anadihilo Core", layout="centered", page_icon="🛰️")

# Custom CSS for UI
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #fff; }
    /* Button Styling */
    div.stButton > button { 
        width: 100%; border-radius: 8px; border: 2px solid #00e5ff; 
        color: #00e5ff; background: transparent; font-weight: bold;
    }
    div.stButton > button:hover { background-color: #00e5ff; color: #000; }
    /* Hide Streamlit Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ ANADIHILO DYNAMICS")
st.caption("Status: Online | Logic: Handover Resolution")

# --- INPUTS (MAIN SCREEN - EXPANDER) ---
# Sidebar hata diya, ab button samne dikhenge
with st.expander("⚙️ CONFIGURE SIMULATION (CLICK HERE)", expanded=True):
    
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Accuracy (Steps)", 1000, 30000, 10000)
    with col2:
        speed = st.select_slider("Animation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
    
    st.markdown("---")
    
    # Body 1: Sun
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Sun**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
        # Sun stationary at 0,0,0
    
    # Body 2: Earth
    with c2:
        st.markdown("🔵 **Earth**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X (m)", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vy (m/s)", value=29780.0, key="vy2")
    
    # Body 3: Moon
    with c3:
        st.markdown("⚪ **Moon**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        # Auto-calc relative to Earth
        x3 = st.number_input("X (m)", value=x2 + 3.844e8, format="%.2e", key="x3")
        vy3 = st.number_input("Vy (m/s)", value=vy2 + 1022.0, key="vy3")

# --- PHYSICS ENGINE ---
@st.cache_data
def run_physics(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Init
    bodies = [
        {'name': 'Sun', 'P': p1, 'p': np.array([0.0, 0.0, 0.0]), 'v': np.array([0.0, 0.0, 0.0]), 'hist': []},
        {'name': 'Earth', 'P': p2, 'p': np.array([x2, 0.0, 0.0]), 'v': np.array([0.0, vy2, 0.0]), 'hist': []},
        {'name': 'Moon', 'P': p3, 'p': np.array([x3, 0.0, 0.0]), 'v': np.array([0.0, vy3, 0.0]), 'hist': []}
    ]
    
    K_VAL = 3.98e14
    EPSILON = 1.0
    DT = 3600
    
    # Loop
    for s in range(steps):
        # 1. Handover
        eff_P = [b['P'] for b in bodies]
        parents = [-1] * 3
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if bodies[j]['P'] > bodies[i]['P']:
                    if np.linalg.norm(bodies[i]['p'] - bodies[j]['p']) < 1.0e10:
                        eff_P[i] = bodies[j]['P']
                        parents[i] = j
        
        # 2. Forces
        accs = []
        for j in range(3):
            a = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r = bodies[k]['p'] - bodies[j]['p']
                d = np.linalg.norm(r)
                if d == 0: continue
                fric = bodies[j]['P'] if parents[j] == k else eff_P[j]
                mag = (K_VAL / fric) * (bodies[k]['P'] / (d**2 + (EPSILON/(bodies[j]['P']+bodies[k]['P']))))
                a += mag * (r/d)
            accs.append(a)
            
        # 3. Update
        for idx, b in enumerate(bodies):
            b['v'] += accs[idx] * DT
            b['p'] += b['v'] * DT
            # Log Data (Save memory: log every 10th step)
            if s % 10 == 0:
                b['hist'].append(b['p'].copy())
                
    return bodies

# --- EXECUTION ---
if st.button("🚀 INITIALIZE & RUN"):
    with st.spinner("Calculating Physics..."):
        try:
            results = run_physics(steps, p1, p2, x2, vy2, p3, x3, vy3)
            
            # Prepare Arrays
            sun_h = np.array(results[0]['hist'])
            earth_h = np.array(results[1]['hist'])
            moon_h = np.array(results[2]['hist'])
            
            # Animation Settings
            skip = 10 if speed == "Slow" else 20 if speed == "Normal" else 50
            duration = 50
            
            tab1, tab2 = st.tabs(["🌌 3D System View", "🌑 2D Relative View"])
            
            # --- TAB 1: 3D MACRO ---
            with tab1:
                st.write("**Solar System Scale** (Auto-Zoomed)")
                fig3d = go.Figure()
                
                # Trails
                fig3d.add_trace(go.Scatter3d(x=sun_h[:,0], y=sun_h[:,1], z=sun_h[:,2], mode='lines', name='Sun', line=dict(color='#ffcc00', width=5)))
                fig3d.add_trace(go.Scatter3d(x=earth_h[:,0], y=earth_h[:,1], z=earth_h[:,2], mode='lines', name='Earth', line=dict(color='#0099ff', width=3)))
                
                # Markers
                fig3d.add_trace(go.Scatter3d(x=[sun_h[0,0]], y=[sun_h[0,1]], z=[sun_h[0,2]], mode='markers', marker=dict(size=20, color='#ffcc00')))
                fig3d.add_trace(go.Scatter3d(x=[earth_h[0,0]], y=[earth_h[0,1]], z=[earth_h[0,2]], mode='markers', marker=dict(size=10, color='#0099ff')))
                
                # Animation Frames
                frames = []
                for k in range(0, len(sun_h), skip):
                    frames.append(go.Frame(data=[
                        go.Scatter3d(x=[sun_h[k,0]], y=[sun_h[k,1]], z=[sun_h[k,2]]),
                        go.Scatter3d(x=[earth_h[k,0]], y=[earth_h[k,1]], z=[earth_h[k,2]])
                    ], traces=[2, 3])) # Update markers only
                
                fig3d.frames = frames
                fig3d.update_layout(
                    scene=dict(bgcolor="black", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                    paper_bgcolor="black", height=450, margin=dict(l=0,r=0,b=0,t=0),
                    updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
                )
                st.plotly_chart(fig3d, use_container_width=True)

            # --- TAB 2: 2D MICRO & GIF ---
            with tab2:
                col_a, col_b = st.columns([2, 1])
                
                rel = moon_h - earth_h
                
                with col_a:
                    st.write("**Top View (Moon Orbit)**")
                    fig2d = go.Figure()
                    # Earth Center
                    fig2d.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='#0099ff'), name='Earth'))
                    # Moon Path
                    fig2d.add_trace(go.Scatter(x=rel[:,0], y=rel[:,1], mode='lines', line=dict(color='white', width=1), opacity=0.5))
                    # Moon Marker
                    fig2d.add_trace(go.Scatter(x=[rel[0,0]], y=[rel[0,1]], mode='markers', marker=dict(size=8, color='white'), name='Moon'))
                    
                    frames2d = [go.Frame(data=[go.Scatter(x=[rel[k,0]], y=[rel[k,1]])], traces=[2]) for k in range(0, len(rel), skip)]
                    fig2d.frames = frames2d
                    
                    fig2d.update_layout(
                        plot_bgcolor='black', paper_bgcolor='black', height=400,
                        xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,b=0,t=0),
                        updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])]
                    )
                    st.plotly_chart(fig2d, use_container_width=True)
                
                with col_b:
                    st.write("📥 **Data & GIF**")
                    
                    # CSV
                    df = pd.DataFrame({'Step': range(len(sun_h))})
                    df['Sun_X'] = sun_h[:,0]
                    df['Earth_X'] = earth_h[:,0]
                    df['Moon_X'] = moon_h[:,0]
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "anadihilo_data.csv", "text/csv")
                    
                    # GIF Generator (Crash Proof)
                    if st.button("Create GIF"):
                        with st.spinner("Generating..."):
                            frames_gif = []
                            step_gif = max(1, len(rel)//60) # Limit frames
                            
                            for i in range(0, len(rel), step_gif):
                                f, ax = plt.subplots(figsize=(3,3), facecolor='black')
                                ax.set_facecolor('black')
                                ax.plot(rel[:i,0], rel[:i,1], color='white', lw=1)
                                ax.plot(0,0, 'o', color='#0099ff')
                                if i>0: ax.plot(rel[i-1,0], rel[i-1,1], 'o', color='white', ms=4)
                                ax.axis('off'); ax.axis('equal')
                                
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                                plt.close(f)
                                buf.seek(0)
                                frames_gif.append(imageio.imread(buf))
                                
                            imageio.mimsave("orbit.gif", frames_gif, fps=15, loop=0)
                            st.image("orbit.gif")
                            with open("orbit.gif", "rb") as f:
                                st.download_button("Download GIF", f, "orbit.gif", "image/gif")

        except Exception as e:
            st.error(f"Error: {e}. Please reduce steps or reload.")
