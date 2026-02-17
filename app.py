import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo Universal Tracer", layout="wide", page_icon="🛰️")

# --- CUSTOM CSS (Frame & Button Visibility) ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    /* Play Button High Contrast - Placed Below Graph */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
    }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    /* Container to help with 3:4 or centered look */
    .plot-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 ANADIHILO UNIVERSAL TRACER")
st.caption("Deterministic Handover Engine | Persistent Camera 3D")

# --- CONFIGURATION (EXPANDER) ---
with st.expander("⚙️ SYSTEM CONFIGURATION", expanded=True):
    c_scale, c_steps = st.columns(2)
    with c_scale:
        scale_mode = st.radio("Coordinate Scale:", ["AU Scale (Solar System)", "Custom/Default Grid"])
    with c_steps:
        steps = st.slider("Simulation Steps", 10000, 100000, 50000, step=5000)
    
    st.markdown("---")
    input_type = st.radio("Input Logic:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    
    def get_input(label, key, def_dg, def_x, def_vy):
        st.markdown(f"**{label}**")
        if input_type == "Systemic Boundary (n)":
            n_val = st.number_input(f"n (m)", value=float(def_dg * 0.8), format="%.2e", key=f"n_{key}")
            dg_val = n_val / 0.8
        else:
            dg_val = st.number_input(f"P (Dg)", value=float(def_dg), format="%.2f", key=f"dg_{key}")
        
        x_init = def_x if scale_mode == "AU Scale (Solar System)" else (def_x / 1e11)
        v_init = def_vy if scale_mode == "AU Scale (Solar System)" else (def_vy / 30000)
        
        x = st.number_input(f"X Pos", value=float(x_init), format="%.2e", key=f"x_{key}")
        vy = st.number_input(f"Y Vel", value=float(v_init), format="%.2f", key=f"v_{key}")
        return dg_val, x, vy

    p1, x1, vy1 = get_input("Body 1 (Sun)", "b1", 1480000.0, 0.0, 0.0)
    p2, x2, vy2 = get_input("Body 2 (Earth)", "b2", 4.44, 1.496e11, 29780.0)
    p3, x3, vy3 = get_input("Body 3 (Moon)", "b3", 0.054, 1.49984e11, 30802.0)

# --- PHYSICS ENGINE (STRICT ANADIHILO LOGIC) ---
@st.cache_data
def run_anadihilo(steps, scale_mode, p1, p2, p3, x2, vy2, x3, vy3):
    K = 3.98e14 if scale_mode == "AU Scale (Solar System)" else 1.0
    DT = 3600 if scale_mode == "AU Scale (Solar System)" else 0.002
    
    P = np.array([p1, p2, p3])
    pos_hist = np.zeros((steps, 3, 3))
    curr_p = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    curr_v = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=float)
    
    for s in range(steps):
        pos_hist[s] = curr_p
        
        # Handover Logic
        eff_P = P.copy()
        parents = [-1, -1, -1]
        dist_em = np.linalg.norm(curr_p[1] - curr_p[2])
        if dist_em < (1.0e10 if scale_mode == "AU Scale (Solar System)" else 0.5):
            eff_P[2] = P[1] 
            parents[2] = 1

        acc = np.zeros_like(curr_p)
        for j in range(3):
            net = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec = curr_p[k] - curr_p[j]
                r_sq = np.sum(r_vec**2)
                eps = 1.0 / (P[j] + P[k]) # Singularity Resolution
                term = P[k] / (r_sq + eps)
                net += term * (r_vec / (np.sqrt(r_sq) + 1e-18))
            
            fric = P[j] if parents[j] == k else eff_P[j]
            acc[j] = (K / fric) * net
            
        curr_v += acc * DT
        curr_p += curr_v * DT
    return pos_hist

# --- EXECUTION ---
if st.button("🚀 EXECUTE DYNAMICS"):
    with st.spinner("Processing Anadihilo Grid..."):
        history = run_anadihilo(steps, scale_mode, p1, p2, p3, x2, vy2, x3, vy3)
        
        # Downsampling for performance
        skip = max(1, steps // 400)
        anim_data = history[::skip]
        
        fig = go.Figure()
        colors = ['#ffcc00', '#0099ff', '#aaaaaa']
        names = ['Sun', 'Earth', 'Moon']
        
        # 1-3: Path Traces (Static)
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=history[:,i,0], y=history[:,i,1], z=history[:,i,2],
                mode='lines', name=f'{names[i]} Path',
                line=dict(color=colors[i], width=2), opacity=0.3
            ))
        
        # 4-6: Markers (Animated)
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=[anim_data[0,i,0]], y=[anim_data[0,i,1]], z=[anim_data[0,i,2]],
                mode='markers', name=names[i],
                marker=dict(color=colors[i], size=[15, 8, 5][i])
            ))
            
        # Frames Fix: Explicitly defining data for traces 3, 4, 5
        frames = []
        for k in range(len(anim_data)):
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=[anim_data[k,0,0]], y=[anim_data[k,0,1]], z=[anim_data[k,0,2]]),
                    go.Scatter3d(x=[anim_data[k,1,0]], y=[anim_data[k,1,1]], z=[anim_data[k,1,2]]),
                    go.Scatter3d(x=[anim_data[k,2,0]], y=[anim_data[k,2,1]], z=[anim_data[k,2,2]])
                ],
                traces=[3, 4, 5],
                name=f"f{k}"
            ))
        
        fig.frames = frames
        
        fig.update_layout(
            scene=dict(
                bgcolor="black", 
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data'
            ),
            paper_bgcolor="black",
            height=700, # Frame size control
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision='constant', # Keeps zoom/rotate persistent
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.5, y=-0.05, xanchor="center",
                buttons=[dict(
                    label="▶ PLAY / PAUSE", 
                    method="animate", 
                    args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, mode="immediate")]
                )]
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CSV Log
        df_out = pd.DataFrame({'Step': np.arange(0, steps, skip)})
        for i in range(3):
            df_out[f'{names[i]}_X'] = anim_data[:,i,0]
            df_out[f'{names[i]}_Y'] = anim_data[:,i,1]
        st.download_button("📥 Download Trajectory CSV", df_out.to_csv(index=False).encode('utf-8'), "anadihilo_physics.csv")

else:
    st.info("Execute dynamics to view the 3D orbital tracer. Rotate and zoom to explore from any angle.")
