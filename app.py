import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo 3D Animation", layout="centered", page_icon="🌌")

# Dark Mode CSS (Blogger Friendly)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    .stButton>button { 
        border: 1px solid #00e5ff; 
        color: #00e5ff; 
        width: 100%; 
        border-radius: 10px;
    }
    .stButton>button:hover { 
        background-color: #00e5ff; 
        color: #000; 
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🌌 Anadihilo Universe Tracer")
st.caption("Deterministic Handover Resolution (Sun-Earth-Moon)")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ Configure Coordinates & Physics (Click to Open)", expanded=True):
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        steps = st.slider("Simulation Duration (Steps)", 1000, 20000, 10000)
    with col_set2:
        K_val = st.number_input("Universal Constant (K)", value=3.98e14, format="%.2e")
        dt = 3600 # 1 Hour fixed
    
    st.markdown("---")
    
    # Body 1: Sun
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Sun**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
        x1 = st.number_input("X", value=0.0, key="x1")
        vy1 = st.number_input("Vy", value=0.0, key="vy1")

    # Body 2: Earth
    with c2:
        st.markdown("🔵 **Earth**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vy", value=29780.0, key="vy2")

    # Body 3: Moon
    with c3:
        st.markdown("⚪ **Moon**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        def_x3 = x2 + 3.844e8 
        def_vy3 = vy2 + 1022.0
        x3 = st.number_input("X", value=def_x3, format="%.2e", key="x3")
        vy3 = st.number_input("Vy", value=def_vy3, key="vy3")

# --- PHYSICS ENGINE ---
def calculate_physics():
    bodies = [
        {'name': 'Sun', 'P': p1, 'p': np.array([x1, 0.0, 0.0]), 'v': np.array([0.0, vy1, 0.0]), 'hist': []},
        {'name': 'Earth', 'P': p2, 'p': np.array([x2, 0.0, 0.0]), 'v': np.array([0.0, vy2, 0.0]), 'hist': []},
        {'name': 'Moon', 'P': p3, 'p': np.array([x3, 0.0, 0.0]), 'v': np.array([0.0, vy3, 0.0]), 'hist': []}
    ]
    
    EPSILON = 1.0
    progress = st.progress(0)
    
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
                force = (K_val / fric) * (bodies[k]['P'] / (d**2 + (EPSILON/(bodies[j]['P']+bodies[k]['P']))))
                a += force * (r/d)
            accs.append(a)
            
        # Update
        for idx, b in enumerate(bodies):
            b['v'] += accs[idx] * dt
            b['p'] += b['v'] * dt
            # Log Data
            if s % 20 == 0: # Optimize memory
                b['hist'].append(b['p'].copy())
        
        if s % (steps//10) == 0:
            progress.progress(s/steps)
            
    progress.progress(100)
    return bodies

# --- EXECUTION ---
if st.button("🚀 EXECUTE & ANIMATE"):
    data = calculate_physics()
    
    tab1, tab2, tab3 = st.tabs(["🌌 Macro Animation", "🌑 Micro Animation", "📊 Data/GIF"])
    
    # 1. MACRO ANIMATION (Sun-Earth)
    with tab1:
        st.write("**Solar System Scale (Click Play ▶️)**")
        
        # Downsample for smooth animation
        anim_step = max(1, len(data[0]['hist']) // 100)
        
        # Create Frames
        frames = []
        for k in range(0, len(data[0]['hist']), anim_step):
            frame_data = []
            for i, b in enumerate(data):
                h = b['hist'][k]
                frame_data.append(go.Scatter3d(x=[h[0]], y=[h[1]], z=[h[2]], mode='markers', marker=dict(color=['#ffcc00', '#0099ff', '#aaaaaa'][i], size=8)))
            frames.append(go.Frame(data=frame_data, name=str(k)))

        # Base Figure
        fig1 = go.Figure(
            data=[
                # Full Paths (Static Lines)
                go.Scatter3d(x=[p[0] for p in data[0]['hist']], y=[p[1] for p in data[0]['hist']], z=[p[2] for p in data[0]['hist']], mode='lines', line=dict(color='#ffcc00', width=2), name='Sun Path'),
                go.Scatter3d(x=[p[0] for p in data[1]['hist']], y=[p[1] for p in data[1]['hist']], z=[p[2] for p in data[1]['hist']], mode='lines', line=dict(color='#0099ff', width=2), name='Earth Path'),
                go.Scatter3d(x=[p[0] for p in data[2]['hist']], y=[p[1] for p in data[2]['hist']], z=[p[2] for p in data[2]['hist']], mode='lines', line=dict(color='#aaaaaa', width=2), name='Moon Path'),
                # Initial Markers (will be animated)
                go.Scatter3d(x=[data[0]['hist'][0][0]], y=[data[0]['hist'][0][1]], z=[data[0]['hist'][0][2]], mode='markers', marker=dict(color='#ffcc00', size=8), name='Sun'),
                go.Scatter3d(x=[data[1]['hist'][0][0]], y=[data[1]['hist'][0][1]], z=[data[1]['hist'][0][2]], mode='markers', marker=dict(color='#0099ff', size=8), name='Earth'),
                go.Scatter3d(x=[data[2]['hist'][0][0]], y=[data[2]['hist'][0][1]], z=[data[2]['hist'][0][2]], mode='markers', marker=dict(color='#aaaaaa', size=8), name='Moon'),
            ],
            layout=go.Layout(
                scene=dict(bgcolor="black"),
                updatemenus=[dict(type="buttons", buttons=[dict(label="▶️ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])],
                height=500, paper_bgcolor="black", font=dict(color="white")
            ),
            frames=frames
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2. MICRO ANIMATION (Moon Relative)
    with tab2:
        st.write("**Moon Orbit Relative to Earth (Click Play ▶️)**")
        
        # Calculate Relative Paths
        h_e = np.array(data[1]['hist'])
        h_m = np.array(data[2]['hist'])
        rel = h_m - h_e
        
        # Frames
        frames_micro = []
        for k in range(0, len(rel), anim_step):
            frames_micro.append(go.Frame(data=[
                go.Scatter3d(x=[rel[k,0]], y=[rel[k,1]], z=[rel[k,2]], mode='markers', marker=dict(color='white', size=8))
            ]))

        fig2 = go.Figure(
            data=[
                # Earth Center
                go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=15, color='blue'), name='Earth'),
                # Moon Path
                go.Scatter3d(x=rel[:,0], y=rel[:,1], z=rel[:,2], mode='lines', line=dict(color='white', width=4), name='Orbit'),
                # Moon Marker (Animated)
                go.Scatter3d(x=[rel[0,0]], y=[rel[0,1]], z=[rel[0,2]], mode='markers', marker=dict(color='white', size=8), name='Moon')
            ],
            layout=go.Layout(
                scene=dict(bgcolor="black"),
                updatemenus=[dict(type="buttons", buttons=[dict(label="▶️ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])],
                height=500, paper_bgcolor="black", font=dict(color="white")
            ),
            frames=frames_micro
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 3. DATA & GIF
    with tab3:
        # CSV
        df = pd.DataFrame({'Step': range(len(data[0]['hist']))})
        for b in data:
            h = np.array(b['hist'])
            df[f"{b['name']}_X"] = h[:,0]
            df[f"{b['name']}_Y"] = h[:,1]
            df[f"{b['name']}_Z"] = h[:,2]
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "anadihilo_data.csv", "text/csv")
        
        # GIF Creator (FIXED)
        st.divider()
        st.write("🎬 **Generate GIF**")
        if st.button("Create GIF (Takes ~15s)"):
            with st.spinner("Generating Animation Frames..."):
                # Use Agg backend for server stability
                plt.switch_backend('Agg') 
                
                frames = []
                # Use fewer frames for GIF to prevent timeout
                gif_step = max(1, len(rel)//60) 
                
                for i in range(0, len(rel), gif_step):
                    f, ax = plt.subplots(figsize=(4,4), facecolor='black')
                    ax.set_facecolor('black')
                    # Plot Path
                    ax.plot(rel[:i,0], rel[:i,1], color='white', lw=1)
                    # Plot Earth
                    ax.plot(0,0, 'o', color='blue', ms=8)
                    # Plot Moon Current Pos
                    if i > 0:
                        ax.plot(rel[i-1,0], rel[i-1,1], 'o', color='gray', ms=5)
                    
                    ax.axis('equal')
                    ax.axis('off')
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                    plt.close(f)
                    buf.seek(0)
                    frames.append(imageio.v3.imread(buf))
                
                # Save GIF
                imageio.mimsave("orbit.gif", frames, fps=10, loop=0)
                
                # Show & Download
                st.success("GIF Created!")
                st.image("orbit.gif")
                with open("orbit.gif", "rb") as file:
                    st.download_button("⬇️ Download GIF", file, "orbit.gif", "image/gif")
