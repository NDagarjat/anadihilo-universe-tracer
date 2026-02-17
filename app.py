import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Server stability fix
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION ---
# 3:4 aspect ratio feel ke liye layout ko "centered" rakha hai
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="centered", page_icon="🔭")

# --- CUSTOM CSS (Visibility & Frame Control) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Play Button: Bright White, High Contrast, Separated from graph */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
        border-radius: 5px;
    }
    
    /* 3:4 Frame Simulation */
    .plot-container {
        border: 1px solid #333;
        border-radius: 10px;
        overflow: hidden;
    }

    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem; max-width: 900px;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE SYSTEM PARAMETERS", expanded=True):
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        # AU Scale Default as requested
        scale_mode = st.radio("Coordinate Scale:", ["AU Scale (Default)", "Custom Grid"])
    with col_b:
        input_type = st.radio("Input Unit:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"])
    with col_c:
        steps = st.slider("Simulation Steps", 5000, 50000, 20000)
    
    st.markdown("---")
    
    # Scale Constants Logic
    is_au = scale_mode == "AU Scale (Default)"
    K_VAL = 3.98e14 if is_au else 1.0
    DT_VAL = 3600 if is_au else 0.002

    def get_p_val(label, key, default_dg):
        if input_type == "Systemic Boundary (n)":
            def_n = default_dg * 0.8
            val = st.number_input(f"{label} (n)", value=float(def_n), format="%.4e", key=key)
            return val / 0.8
        else:
            return st.number_input(f"{label} (Dg)", value=float(default_dg), format="%.4f", key=key)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Body 1: Sun**")
        p1 = get_p_val("P1", "p1", 1480000.0)
    with c2:
        st.markdown("🔵 **Body 2: Earth**")
        p2 = get_p_val("P2", "p2", 4.44)
        x2 = st.number_input("Pos X", value=1.496e11 if is_au else 1.0, format="%.2e", key="x2")
        vy2 = st.number_input("Vel Y", value=29780.0 if is_au else 0.0, format="%.2f", key="vy2")
    with c3:
        st.markdown("⚪ **Body 3: Moon**")
        p3 = get_p_val("P3", "p3", 0.054)
        x3 = st.number_input("Pos X", value=1.49984e11 if is_au else 1.1, format="%.2e", key="x3")
        vy3 = st.number_input("Vel Y", value=30802.0 if is_au else 0.1, format="%.2f", key="vy3")

# --- PHYSICS ENGINE (STRICT PDF LOGIC) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, p3, x2, vy2, x3, vy3, K, DT):
    P_vals = np.array([p1, p2, p3])
    pos = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=np.float64)
    vel = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=np.float64)
    
    h_pos = np.zeros((steps, 3, 3))
    h_vel = np.zeros((steps, 3))
    h_acc = np.zeros((steps, 3))
    
    for s in range(steps):
        h_pos[s] = pos
        acc = np.zeros_like(pos)
        
        # Handover / Assimilation Logic
        eff_P = P_vals.copy()
        parents = [-1, -1, -1]
        
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if P_vals[j] > P_vals[i]:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < (1.0e10 if K > 1 else 0.5):
                        eff_P[i] = P_vals[j]
                        parents[i] = j

        # Force Calculation
        for j in range(3):
            net_pull = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                epsilon = 1.0 / (P_vals[j] + P_vals[k])
                term = P_vals[k] / (r_mag**2 + epsilon)
                net_pull += term * (r_vec / (r_mag + 1e-18))
            
            friction = P_vals[j] if parents[j] == k else eff_P[j]
            acc[j] = (K / friction) * net_pull
        
        vel += acc * DT
        pos += vel * DT
        for b in range(3):
            h_vel[s, b] = np.linalg.norm(vel[b])
            h_acc[s, b] = np.linalg.norm(acc[b])
            
    return h_pos, h_vel, h_acc

# --- EXECUTION ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    
    with st.spinner("Processing Anadihilo Dynamics..."):
        h_pos, h_vel, h_acc = run_simulation(steps, p1, p2, p3, x2, vy2, x3, vy3, K_VAL, DT_VAL)
        P_vals = [p1, p2, p3]
        
        tab1, tab2, tab3 = st.tabs(["🌌 Interactive Animation", "📊 Scientific Dashboard", "💾 Data Logs"])
        
        # --- TAB 1: 3D ANIMATION (3:4 Aspect Ratio + Persistence) ---
        with tab1:
            st.write("### N-Body Trajectory Visualization")
            skip = max(1, steps // 500)
            
            fig = go.Figure()
            colors = ['#ffcc00', '#0099ff', '#aaaaaa']
            names = ["Sun", "Earth", "Moon"]
            
            # Trails
            for i in range(3):
                fig.add_trace(go.Scatter3d(x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2], 
                                           mode='lines', name=names[i], line=dict(color=colors[i], width=4)))
            
            # Animated Markers
            for i in range(3):
                fig.add_trace(go.Scatter3d(x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]], 
                                           mode='markers', name=names[i], marker=dict(color=colors[i], size=[18, 9, 5][i])))
            
            # Frames
            frames = [go.Frame(data=[go.Scatter3d(x=[h_pos[k,i,0]], y=[h_pos[k,i,1]], z=[h_pos[k,i,2]]) for i in range(3)], 
                               traces=[3, 4, 5], name=str(k)) for k in range(0, steps, skip)]
            
            fig.frames = frames
            fig.update_layout(
                scene=dict(bgcolor="black", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
                paper_bgcolor="black", 
                # 3:4 Aspect Ratio Fix (Height 800 for Wide layout makes it look vertical/square)
                height=700, 
                margin=dict(l=0, r=0, b=0, t=0),
                # CAMERA PERSISTENCE: uirevision constant rakhta hai zoom aur rotation
                uirevision='constant',
                updatemenus=[dict(
                    type="buttons", showactive=False, x=0.5, y=-0.05, xanchor="center",
                    buttons=[dict(label="▶ PLAY ANIMATION", method="animate", 
                                  args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, mode="immediate")])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Tip: Zoom/Rotate freely. The camera position is locked even after pausing.")

        # --- TAB 2: SCIENTIFIC DASHBOARD ---
        with tab2:
            st.write("### Static Analysis")
            fig_mpl, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e1117')
            for i in range(3):
                axes[0].plot(h_pos[:, i, 0], h_pos[:, i, 1], color=['yellow','blue','grey'][i], label=names[i])
                axes[1].plot(h_vel[:, i], color=['yellow','blue','grey'][i], label=names[i])
            
            axes[0].set_title("2D Projection", color='white')
            axes[1].set_title("Velocity Curve", color='white')
            for ax in axes:
                ax.set_facecolor('#0e1117'); ax.tick_params(colors='white'); ax.grid(alpha=0.1)
                ax.legend(facecolor='#111', labelcolor='white')
            st.pyplot(fig_mpl)

        # --- TAB 3: DATA ---
        with tab3:
            df = pd.DataFrame({'Step': range(steps), 'Sun_X': h_pos[:,0,0], 'Earth_X': h_pos[:,1,0], 'Moon_X': h_pos[:,2,0]})
            st.download_button("📥 Download Full Log (CSV)", df.to_csv(index=False).encode('utf-8'), "anadihilo_scientific_log.csv")

else:
    st.info("👋 Set Parameters and click EXECUTE. Default is set to AU Solar Scale.")
