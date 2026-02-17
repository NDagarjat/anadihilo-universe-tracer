import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="wide", page_icon="🔭")

# --- CUSTOM CSS (UI Preservation) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE SYSTEM (Click to Expand)", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        scale_mode = st.radio("Coordinate Scale:", ["AU Scale", "Custom Scale"])
    with col_b:
        input_type = st.radio("Input Unit:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"])
    with col_c:
        steps = st.slider("Simulation Steps", 5000, 50000, 15000)

    st.markdown("---")
    
    # Scale Constants
    is_au = (scale_mode == "AU Scale")
    K_CONST = 3.98e14 if is_au else 1.0
    DT_VAL = 3600 if is_au else 0.002

    def get_val(label, key, def_dg, def_x, def_vy):
        st.markdown(f"**{label}**")
        # P or n selection logic
        if input_type == "Systemic Boundary (n)":
            n_in = st.number_input(f"n (m)", value=float(def_dg*0.8), format="%.2e", key=f"n{key}")
            p_out = n_in / 0.8
        else:
            p_out = st.number_input(f"P (Dg)", value=float(def_dg), key=f"p{key}")
        
        # Scale handling for Custom values
        x_out = st.number_input(f"X Pos", value=float(def_x if is_au else 1.0), format="%.2e", key=f"x{key}")
        v_out = st.number_input(f"Y Vel", value=float(def_vy if is_au else 0.0), format="%.2f", key=f"v{key}")
        return p_out, x_out, v_out

    c1, c2, c3 = st.columns(3)
    with c1: p1, x1, v1 = get_val("Body 1 (Sun)", "1", 1480000.0, 0.0, 0.0)
    with c2: p2, x2, v2 = get_val("Body 2 (Earth)", "2", 4.44, 1.496e11, 29780.0)
    with c3: p3, x3, v3 = get_val("Body 3 (Moon)", "3", 0.054, 1.49984e11, 30802.0)

# --- PHYSICS ENGINE (STRICT ANADIHILO) ---
@st.cache_data
def run_sim(steps, p1, p2, p3, x1, x2, x3, v1, v2, v3, K, DT):
    P = np.array([p1, p2, p3])
    pos = np.array([[x1,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    vel = np.array([[0,v1,0], [0,v2,0], [0,v3,0]], dtype=float)
    
    h_pos = np.zeros((steps, 3, 3))
    h_vel = np.zeros((steps, 3))
    h_acc = np.zeros((steps, 3))

    for s in range(steps):
        h_pos[s] = pos
        acc = np.zeros_like(pos)
        
        # Handover/Assimilation
        eff_P = P.copy()
        parents = [-1]*3
        for i in range(3):
            for j in range(3):
                if i != j and P[j] > P[i]:
                    dist = np.linalg.norm(pos[i]-pos[j])
                    if dist < (1e10 if K > 1 else 0.5):
                        eff_P[i], parents[i] = P[j], j

        # Forces (Strict PDF Formula: aj = K/fric * net_pull)
        for j in range(3):
            net = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                # Singularity Fix
                eps = 1.0 / (P[j] + P[k])
                net += (P[k]/(r_mag**2 + eps)) * (r_vec/(r_mag + 1e-18))
            
            fric = P[j] if parents[j] == k else eff_P[j]
            acc[j] = (K/fric) * net
        
        vel += acc * DT
        pos += vel * DT
        for b in range(3):
            h_vel[s,b], h_acc[s,b] = np.linalg.norm(vel[b]), np.linalg.norm(acc[b])
            
    return h_pos, h_vel, h_acc

# --- EXECUTION ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    with st.spinner("Processing Dynamics..."):
        h_pos, h_vel, h_acc = run_sim(steps, p1, p2, p3, x1, x2, x3, v1, v2, v3, K_CONST, DT_VAL)
        
        t1, t2, t3 = st.tabs(["🌌 3D Animation", "📊 2D Dashboard", "💾 Logs"])
        
        with t1:
            st.write("### N-Body Interactive System (3:4 Ratio)")
            skip = max(1, steps // 400)
            fig = go.Figure()
            cols = ['#ffcc00', '#0099ff', '#aaaaaa']
            names = ["Sun", "Earth", "Moon"]
            
            # Static Trails
            for i in range(3):
                fig.add_trace(go.Scatter3d(x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2], 
                                           mode='lines', line=dict(color=cols[i], width=3), name=names[i]))
                fig.add_trace(go.Scatter3d(x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]], 
                                           mode='markers', marker=dict(color=cols[i], size=[20,10,6][i])))

            # Frames
            frames = [go.Frame(data=[go.Scatter3d(x=[h_pos[k,i,0]], y=[h_pos[k,i,1]], z=[h_pos[k,i,2]]) for i in range(3)], traces=[3,4,5]) 
                      for k in range(0, steps, skip)]
            
            fig.frames = frames
            fig.update_layout(
                scene=dict(bgcolor="black", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, 
                           aspectmode='manual', aspectratio=dict(x=1, y=0.75, z=0.75)), # 3:4 Proportion
                paper_bgcolor="black", margin=dict(l=0,r=0,b=0,t=0), height=700, 
                uirevision='constant', # Persistent Camera State
                updatemenus=[dict(
                    type="buttons", x=0.5, y=-0.05, xanchor="center", # Play Button position fixed
                    buttons=[dict(label="▶ PLAY / PAUSE", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.write("### Scientific Dashboard (Matplotlib)")
            fig_mpl, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e1117')
            for i in range(3):
                axes[0].plot(h_pos[:,i,0], h_pos[:,i,1], color=['red','blue','teal'][i], label=names[i])
                axes[1].plot(h_vel[:,i], color=['red','blue','teal'][i])
            axes[0].set_title("Trajectory Map", color='white'); axes[1].set_title("Velocity Profile", color='white')
            for ax in axes: ax.set_facecolor('#0e1117'); ax.tick_params(colors='white'); ax.grid(alpha=0.2); ax.legend(facecolor='#111', labelcolor='white')
            st.pyplot(fig_mpl)

        with t3:
            st.write("### Data Logs")
            df = pd.DataFrame({'Step': range(steps)})
            for i, name in enumerate(names):
                df[f'{name}_X'] = h_pos[:, i, 0]
                df[f'{name}_Y'] = h_pos[:, i, 1]
                df[f'{name}_Vel'] = h_vel[:, i]
            st.dataframe(df.head(100))
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "anadihilo_log.csv")

else:
    st.info("👋 Setup your parameters and click 'EXECUTE'. Zoom/Rotate works during Play.")

