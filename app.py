import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo 3D", layout="centered", page_icon="🌌")

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
    /* Hide Streamlit Header for Clean Look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🌌 Anadihilo Universe Tracer")
st.caption("Deterministic Handover Resolution (Sun-Earth-Moon)")

# --- MAIN PAGE INPUTS (EXPANDER) ---
# Sidebar hatakar 'Expander' lagaya hai taaki Iframe mein saaf dikhe
with st.expander("⚙️ Configure Coordinates & Physics (Click to Open)", expanded=True):
    
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        steps = st.slider("Simulation Duration (Steps)", 1000, 30000, 10000)
    with col_set2:
        K_val = st.number_input("Universal Constant (K)", value=3.98e14, format="%.2e")
        dt = 3600 # 1 Hour fixed
    
    st.markdown("---")
    
    # Body 1: Sun
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Sun (Anchor)**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
        x1 = st.number_input("X", value=0.0, key="x1")
        vy1 = st.number_input("Vy", value=0.0, key="vy1")

    # Body 2: Earth
    with c2:
        st.markdown("🔵 **Earth (Parent)**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vy", value=29780.0, key="vy2")

    # Body 3: Moon
    with c3:
        st.markdown("⚪ **Moon (Child)**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        # Default Moon relative to Earth
        def_x3 = x2 + 3.844e8 
        def_vy3 = vy2 + 1022.0
        x3 = st.number_input("X", value=def_x3, format="%.2e", key="x3")
        vy3 = st.number_input("Vy", value=def_vy3, key="vy3")

# --- PHYSICS ENGINE ---
def calculate_physics():
    # Init Bodies
    bodies = [
        {'name': 'Sun', 'P': p1, 'p': np.array([x1, 0.0, 0.0]), 'v': np.array([0.0, vy1, 0.0]), 'hist': []},
        {'name': 'Earth', 'P': p2, 'p': np.array([x2, 0.0, 0.0]), 'v': np.array([0.0, vy2, 0.0]), 'hist': []},
        {'name': 'Moon', 'P': p3, 'p': np.array([x3, 0.0, 0.0]), 'v': np.array([0.0, vy3, 0.0]), 'hist': []}
    ]
    
    EPSILON = 1.0
    progress = st.progress(0)
    
    for s in range(steps):
        # 1. Handover Logic
        eff_P = [b['P'] for b in bodies]
        parents = [-1] * 3
        
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if bodies[j]['P'] > bodies[i]['P']:
                    # 10 Million km assimilation zone
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
                force = (K_val / fric) * (bodies[k]['P'] / (d**2 + (EPSILON/(bodies[j]['P']+bodies[k]['P']))))
                a += force * (r/d)
            accs.append(a)
            
        # 3. Update
        for idx, b in enumerate(bodies):
            b['v'] += accs[idx] * dt
            b['p'] += b['v'] * dt
            if s % 10 == 0: 
                b['hist'].append(b['p'].copy())
        
        if s % (steps//10) == 0:
            progress.progress(s/steps)
            
    progress.progress(100)
    return bodies

# --- EXECUTION ---
if st.button("🚀 EXECUTE SIMULATION"):
    data = calculate_physics()
    
    # Dual View Tabs
    tab1, tab2, tab3 = st.tabs(["🌌 Macro View", "🌑 Micro View", "📊 Data/GIF"])
    
    with tab1:
        st.write("**System Overview (Sun-Earth Scale)**")
        fig1 = go.Figure()
        cols = ['#ffcc00', '#0099ff', '#aaaaaa']
        for i, b in enumerate(data):
            h = np.array(b['hist'])
            fig1.add_trace(go.Scatter3d(x=h[:,0], y=h[:,1], z=h[:,2], mode='lines', name=b['name'], line=dict(color=cols[i], width=3)))
            fig1.add_trace(go.Scatter3d(x=[h[-1,0]], y=[h[-1,1]], z=[h[-1,2]], mode='markers', marker=dict(size=5, color=cols[i]), showlegend=False))
        
        fig1.update_layout(scene=dict(bgcolor="black"), margin=dict(l=0, r=0, b=0, t=0), height=500, paper_bgcolor="black", font=dict(color="white"))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.write("**Moon Orbit (Relative to Earth)**")
        fig2 = go.Figure()
        # Earth Center
        fig2.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='Earth', marker=dict(size=15, color='blue')))
        # Moon Relative
        h_e = np.array(data[1]['hist'])
        h_m = np.array(data[2]['hist'])
        rel = h_m - h_e
        fig2.add_trace(go.Scatter3d(x=rel[:,0], y=rel[:,1], z=rel[:,2], mode='lines', name='Moon Path', line=dict(color='white', width=4)))
        
        fig2.update_layout(scene=dict(bgcolor="black"), margin=dict(l=0, r=0, b=0, t=0), height=500, paper_bgcolor="black", font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)

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
        
        # GIF
        if st.button("🎬 Generate GIF"):
            st.info("Generating Animation...")
            frames = []
            step_sz = max(1, len(rel)//50)
            for i in range(0, len(rel), step_sz):
                f, ax = plt.subplots(figsize=(4,4))
                f.patch.set_facecolor('black')
                ax.set_facecolor('black')
                ax.plot(rel[:i,0], rel[:i,1], color='white')
                ax.plot(0,0, 'o', color='blue')
                ax.axis('off')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(f)
                buf.seek(0)
                frames.append(imageio.v3.imread(buf))
            imageio.mimsave("orbit.gif", frames, fps=12, loop=0)
            st.image("orbit.gif")
