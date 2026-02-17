import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIG (3:4 Ratio Layout) ---
st.set_page_config(page_title="Anadihilo Tracer", layout="centered", page_icon="🔭")

# --- CUSTOM CSS (Visibility & Aspect Ratio) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Play Button: Bright White & Clear */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
        border-radius: 5px;
    }
    
    /* Center the container for 3:4 feel */
    .block-container { max-width: 800px; padding-top: 2rem; }
    
    header {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")
st.caption("AU Default Scale | 3:4 Frame | NASA-Grade Visualization")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE PARAMETERS", expanded=True):
    col_x, col_y = st.columns(2)
    with col_x:
        input_type = st.radio("Input Unit:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"], horizontal=True)
    with col_y:
        steps = st.slider("Simulation Steps", 5000, 50000, 20000)

    st.markdown("---")
    
    # AU Scale Constants (Fixed as Default)
    K_CONST = 3.98e14
    DT_VAL = 3600 # 1 Hour steps

    def get_p_input(label, key, def_dg):
        if input_type == "Systemic Boundary (n)":
            val_n = st.number_input(f"{label} (n)", value=float(def_dg*0.8), format="%.2e", key=f"n{key}")
            return val_n / 0.8
        else:
            return st.number_input(f"{label} (Dg)", value=float(def_dg), format="%.4f", key=f"p{key}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Body 1 (Sun)**")
        p1 = get_p_input("Intensity", "1", 1480000.0)
    with c2:
        st.markdown("🔵 **Body 2 (Earth)**")
        p2 = get_p_input("Intensity", "2", 4.44)
        x2 = st.number_input("Pos X", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vel Y", value=29780.0, key="vy2")
    with c3:
        st.markdown("⚪ **Body 3 (Moon)**")
        p3 = get_p_input("Intensity", "3", 0.054)
        x3 = st.number_input("Pos X ", value=1.49984e11, format="%.2e", key="x3")
        vy3 = st.number_input("Vel Y ", value=30802.0, key="vy3")

# --- PHYSICS ENGINE (STRICT ANADIHILO LOGIC) ---
@st.cache_data(show_spinner=False)
def run_anadihilo(steps, p1, p2, p3, x2, vy2, x3, vy3):
    P = np.array([p1, p2, p3])
    pos = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    vel = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=float)
    
    h_pos = np.zeros((steps, 3, 3))
    h_vel = np.zeros((steps, 3))
    h_acc = np.zeros((steps, 3))

    for s in range(steps):
        h_pos[s] = pos
        acc = np.zeros_like(pos)
        
        # Handover Logic
        eff_P = P.copy()
        parents = [-1, -1, -1]
        dist_em = np.linalg.norm(pos[1] - pos[2])
        if dist_em < 1.0e10: # Earth Influence Zone
            eff_P[2] = P[1]
            parents[2] = 1

        # Force Calculation
        for j in range(3):
            net = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                # Singularity Fix
                epsilon = 1.0 / (P[j] + P[k])
                term = P[k] / (np.sum(r_vec**2) + epsilon)
                net += term * (r_vec / (r_mag + 1e-18))
            
            fric = P[j] if parents[j] == k else eff_P[j]
            acc[j] = (K_CONST / fric) * net
        
        vel += acc * DT_VAL
        pos += vel * DT_VAL
        for b in range(3):
            h_vel[s, b] = np.linalg.norm(vel[b])
            h_acc[s, b] = np.linalg.norm(acc[b])
            
    return h_pos, h_vel, h_acc

# --- DASHBOARD ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    h_pos, h_vel, h_acc = run_anadihilo(steps, p1, p2, p3, x2, vy2, x3, vy3)
    
    tab1, tab2, tab3 = st.tabs(["🌌 3D Animation (3:4)", "📈 Scientific Dashboard", "💾 Logs"])
    
    with tab1:
        # Optimization for smooth play
        skip = max(1, steps // 400)
        
        fig = go.Figure()
        colors = ['#ffcc00', '#0099ff', '#aaaaaa']
        names = ["Sun", "Earth", "Moon"]
        
        # 1. Trails
        for i in range(3):
            fig.add_trace(go.Scatter3d(x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2], 
                                       mode='lines', line=dict(color=colors[i], width=3), name=f"{names[i]} Path"))
        
        # 2. Markers (Visible Sizes)
        for i in range(3):
            fig.add_trace(go.Scatter3d(x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]], 
                                       mode='markers', marker=dict(color=colors[i], size=[18, 9, 5][i]), name=names[i]))

        # 3. Frames
        frames = [go.Frame(data=[go.Scatter3d(x=[h_pos[k,i,0]], y=[h_pos[k,i,1]], z=[h_pos[k,i,2]]) for i in range(3)], 
                           traces=[3, 4, 5]) for k in range(0, steps, skip)]
        
        fig.frames = frames
        fig.update_layout(
            scene=dict(bgcolor="black", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
            paper_bgcolor="black", 
            # 3:4 Ratio approx (Height 800, Width 600)
            height=800, 
            margin=dict(l=0, r=0, b=0, t=0),
            # PERSISTENT CAMERA: Camera zoom/rotate nahi hilega play ke baad
            uirevision='constant', 
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.5, y=-0.05, xanchor="center",
                buttons=[dict(label="▶ PLAY ANIMATION", method="animate", 
                              args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)])]
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Tip: Zoom/Rotate freely. Pause karke bhi har angle se dekh sakte ho.")

    with tab2:
        # Matplotlib Dashboard (Based on your provided code)
        fig_mpl, axes = plt.subplots(2, 1, figsize=(10, 10), facecolor='#0e1117')
        sci_colors = ['#E63946', '#457B9D', '#2A9D8F']
        
        # Trajectory
        for i in range(3):
            axes[0].plot(h_pos[:,i,0], h_pos[:,i,1], color=sci_colors[i], label=names[i], lw=2)
            # End Circle
            final_x, final_y = h_pos[-1, i, 0], h_pos[-1, i, 1]
            circ = Circle((final_x, final_y), radius=np.max(np.abs(h_pos))*0.02, color=sci_colors[i], alpha=0.6)
            axes[0].add_patch(circ)
            
        axes[0].set_title("Scientific 2D Map", color='white')
        
        # Velocity
        for i in range(3):
            axes[1].plot(h_vel[:,i], color=sci_colors[i], label=names[i])
        axes[1].set_title("Velocity Curve", color='white')
        
        for ax in axes:
            ax.set_facecolor('#0e1117'); ax.tick_params(colors='white'); ax.grid(alpha=0.1)
            ax.legend(facecolor='#111', labelcolor='white')
        st.pyplot(fig_mpl)

    with tab3:
        df = pd.DataFrame({'Step': range(steps), 'Sun_X': h_pos[:,0,0], 'Earth_X': h_pos[:,1,0], 'Moon_X': h_pos[:,2,0]})
        st.download_button("📥 Download Log (CSV)", df.to_csv(index=False).encode('utf-8'), "anadihilo_scientific_log.csv")

else:
    st.info("Ready to Calculate. AU Scale is active by default.")
