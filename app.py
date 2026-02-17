import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Anadihilo Universal Tracer", layout="wide", page_icon="🛰️")

# --- CUSTOM CSS (Visibility & Frame Control) ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    /* Play Button High Contrast & Position */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
        margin-top: 30px !important;
    }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    /* 3:4 Aspect Ratio Container */
    .plot-container {
        width: 100%;
        max-width: 800px;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 ANADIHILO UNIVERSAL TRACER")
st.caption("Deterministic Handover Engine | NASA-Grade Visualization")

# --- CONFIGURATION (EXPANDER) ---
with st.expander("⚙️ SYSTEM CONFIGURATION", expanded=True):
    c_scale, c_steps = st.columns(2)
    with c_scale:
        scale_mode = st.radio("Coordinate Scale:", ["AU Scale (Solar System)", "Custom/Default Grid"])
    with c_steps:
        steps = st.slider("Simulation Steps", 10000, 100000, 50000, step=5000)
    
    st.markdown("---")
    
    # Unit Type Selection
    input_type = st.radio("Input Logic:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Logic to handle n vs Dg conversion (Dg = n / 0.8)
    def get_input(label, key, def_dg, def_x, def_vy):
        st.markdown(f"**{label}**")
        if input_type == "Systemic Boundary (n)":
            n_val = st.number_input(f"n (m)", value=float(def_dg * 0.8), format="%.2e", key=f"n_{key}")
            dg_val = n_val / 0.8
        else:
            dg_val = st.number_input(f"P (Dg)", value=float(def_dg), format="%.2f", key=f"dg_{key}")
        
        # Scale handling for Coordinates
        x_init = def_x if scale_mode == "AU Scale (Solar System)" else 1.0
        v_init = def_vy if scale_mode == "AU Scale (Solar System)" else 1.0
        
        x = st.number_input(f"X Pos", value=float(x_init), format="%.2e", key=f"x_{key}")
        vy = st.number_input(f"Y Vel", value=float(v_init), format="%.2f", key=f"v_{key}")
        return dg_val, x, vy

    p1, x1, vy1 = get_input("Body 1 (Sun)", "b1", 1480000.0, 0.0, 0.0)
    p2, x2, vy2 = get_input("Body 2 (Earth)", "b2", 4.44, 1.496e11, 29780.0)
    p3, x3, vy3 = get_input("Body 3 (Moon)", "b3", 0.054, 1.49984e11, 30802.0)

# --- PHYSICS ENGINE (STRICT PDF LOGIC) ---
@st.cache_data
def run_anadihilo(steps, p1, p2, p3, x2, vy2, x3, vy3):
    K = 3.98e14 if scale_mode == "AU Scale (Solar System)" else 1.0
    DT = 3600 if scale_mode == "AU Scale (Solar System)" else 0.002
    
    P = np.array([p1, p2, p3])
    pos = np.zeros((steps, 3, 3))
    curr_p = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    curr_v = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=float)
    
    for s in range(steps):
        pos[s] = curr_p
        
        # [span_5](start_span)[span_6](start_span)Handover Logic[span_5](end_span)[span_6](end_span)
        eff_P = P.copy()
        parents = [-1, -1, -1]
        dist_em = np.linalg.norm(curr_p[1] - curr_p[2])
        if dist_em < (1.0e10 if K > 1 else 0.5):
            [span_7](start_span)eff_P[2] = P[1] # Moon assimilates to Earth[span_7](end_span)
            parents[2] = 1

        # [span_8](start_span)Acceleration[span_8](end_span)
        acc = np.zeros_like(curr_p)
        for j in range(3):
            net = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec = curr_p[k] - curr_p[j]
                r_sq = np.sum(r_vec**2)
                [span_9](start_span)eps = 1.0 / (P[j] + P[k]) # Singularity Fix[span_9](end_span)
                term = P[k] / (r_sq + eps)
                net += term * (r_vec / (np.sqrt(r_sq) + 1e-18))
            
            [span_10](start_span)fric = P[j] if parents[j] == k else eff_P[j] # Priority vs Friction[span_10](end_span)
            acc[j] = (K / fric) * net
            
        curr_v += acc * DT
        curr_p += curr_v * DT
    return pos

# --- EXECUTION & UI ---
if st.button("🚀 EXECUTE DYNAMICS"):
    with st.spinner("Processing Anadihilo Grid..."):
        history = run_anadihilo(steps, p1, p2, p3, x2, vy2, x3, vy3)
        
        # Downsample for smooth animation (500 frames max)
        skip = max(1, steps // 500)
        anim_data = history[::skip]
        
        # Plotly 3D Figure
        fig = go.Figure()
        colors = ['#ffcc00', '#0099ff', '#aaaaaa']
        names = ['Sun', 'Earth', 'Moon']
        
        # Static Trails
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=history[:,i,0], y=history[:,i,1], z=history[:,i,2],
                mode='lines', name=f'{names[i]} Path',
                line=dict(color=colors[i], width=2), opacity=0.4
            ))
        
        # Moving Markers
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=[anim_data[0,i,0]], y=[anim_data[0,i,1]], z=[anim_data[0,i,2]],
                mode='markers', name=names[i],
                marker=dict(color=colors[i], size=[15, 8, 4][i])
            ))
            
        # Frames (Persistent Camera Logic)
        frames = [go.Frame(
            data=[go.Scatter3d(x=[anim_data[k,i,0]], y=[anim_data[k,i,1]], z=[anim_data[k,i,2]]) for i in range(3)],
            traces=[3, 4, 5], name=str(k)
        ) for k in range(len(anim_data))]
        
        fig.frames = frames
        
        fig.update_layout(
            scene=dict(
                bgcolor="black", 
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data' # Ensures 1:1:1 coordinate scale
            ),
            paper_bgcolor="black",
            height=600, # This with layout centered creates roughly 3:4 or 16:9 depending on screen
            margin=dict(l=0, r=0, b=0, t=0),
            # PERSISTENT CAMERA: uirevision ensures zoom/rotation stays after play/pause
            uirevision='constant',
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.5, y=-0.1, xanchor="center",
                buttons=[dict(
                    label="▶ PLAY / PAUSE", 
                    method="animate", 
                    args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, mode="immediate")]
                )]
            )]
        )
        
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CSV Download
        df = pd.DataFrame({'Step': range(len(history))})
        for i in range(3):
            df[f'B{i+1}_X'], df[f'B{i+1}_Y'] = history[:,i,0], history[:,i,1]
        st.download_button("📥 Download Trajectory CSV", df.to_csv(index=False).encode('utf-8'), "anadihilo_log.csv")

else:
    st.info("Setup parameters and click Execute. Note: Zoom/Rotate works even during Play.")
