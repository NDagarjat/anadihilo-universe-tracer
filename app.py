Import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Server stability fix
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="wide", page_icon="🔭")

# --- CUSTOM CSS (High Visibility) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Play Button High Contrast (White on Black) */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
        border-radius: 5px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1c1f26; border-radius: 5px; color: #fff; }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
    
    /* Input Labels */
    .stRadio label { font-weight: bold; color: #00e5ff !important; }
    
    /* Hide Default Elements */
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")
st.markdown("**Status:** Ready | **Mode:** Multi-Body System | **Logic:** PDF Strict Implementation")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE PARAMETERS (Click to Expand)", expanded=True):
    
    # Global Settings
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        # Input Mode Selection (As requested)
        input_type = st.radio("Select Input Unit:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"])
    with col_b:
        steps = st.slider("Simulation Steps", 5000, 50000, 15000)
    with col_c:
        speed = st.select_slider("Animation Speed", options=["Precision", "Normal", "Hyper"], value="Normal")
    
    st.markdown("---")
    
    # Helper to convert n to Dg if needed
    def get_p_val(label, key, default_dg):
        if input_type == "Systemic Boundary (n)":
            # Default n = Dg * 0.8
            def_n = default_dg * 0.8
            val = st.number_input(f"{label} - n (meters)", value=float(def_n), format="%.4e", key=key)
            return val / 0.8 # Return Dg for calculation
        else:
            return st.number_input(f"{label} - P (Dg)", value=float(default_dg), format="%.4f", key=key)

    # Body Inputs
    c1, c2, c3 = st.columns(3)
    
    # Body 1
    with c1:
        st.markdown("🟡 **Body 1: Sun (Core 1)**")
        p1 = get_p_val("Intensity", "p1", 1480000.0)
        # Sun fixed at 0,0
        
    # Body 2
    with c2:
        st.markdown("🔵 **Body 2: Earth (Core 2)**")
        p2 = get_p_val("Intensity", "p2", 4.44)
        x2 = st.number_input("Pos X (m)", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vel Y (m/s)", value=29780.0, key="vy2")
        
    # Body 3
    with c3:
        st.markdown("⚪ **Body 3: Moon (Core 3)**")
        p3 = get_p_val("Intensity", "p3", 0.054)
        # Auto-calc default
        def_x3 = x2 + 3.844e8
        def_vy3 = vy2 + 1022.0
        x3 = st.number_input("Pos X (m)", value=def_x3, format="%.2e", key="x3")
        vy3 = st.number_input("Vel Y (m/s)", value=def_vy3, key="vy3")

# --- PHYSICS ENGINE (STRICT PDF LOGIC) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Constants
    K_constant = 3.98e14 
    dt = 3600 # 1 Hour
    
    # Arrays
    P_vals = np.array([p1, p2, p3])
    pos = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=np.float64)
    vel = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=np.float64)
    
    # History
    hist_pos = np.zeros((steps, 3, 3))
    hist_vel = np.zeros((steps, 3))
    hist_acc = np.zeros((steps, 3))
    
    for s in range(steps):
        hist_pos[s] = pos
        acc = np.zeros_like(pos)
        
        # --- HANDOVER / ASSIMILATION LOGIC ---
        # "Moon within Earth's P-field... priority signal of Earth dominates"
        eff_P = P_vals.copy()
        parents = [-1, -1, -1]
        
        for i in range(3):
            for j in range(3):
                if i == j: continue
                # Assimilation Condition: Neighbor is heavier & close
                if P_vals[j] > P_vals[i]:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    # 10 Million km Threshold (Influence Zone)
                    if dist < 1.0e10:
                        eff_P[i] = P_vals[j] # Adopt Parent's Friction
                        parents[i] = j       # Mark Parent

        # --- FORCE CALCULATION (Eq 6) ---
        for j in range(3):
            net_pull = np.zeros(3)
            for k in range(3):
                if j == k: continue
                
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                
                # Singularity Fix: epsilon = 1 / (Sum of P)
                epsilon = 1.0 / (P_vals[j] + P_vals[k])
                
                # Anadihilo Force Term
                term = P_vals[k] / (r_mag**2 + epsilon)
                net_pull += term * (r_vec / (r_mag + 1e-18))
            
            # Friction Logic (The PDF Key)
            # Internal Interaction (with Parent) -> Intrinsic P
            # External Interaction (with Sun/Others) -> Effective P
            if parents[j] == k:
                fric = P_vals[j] 
            else:
                fric = eff_P[j]
                
            acc[j] = (K_constant / fric) * net_pull
        
        vel += acc * dt
        pos += vel * dt
        
        # Logging
        for b in range(3):
            hist_vel[s, b] = np.linalg.norm(vel[b])
            hist_acc[s, b] = np.linalg.norm(acc[b])
            
    return hist_pos, hist_vel, hist_acc

# --- EXECUTION ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    
    with st.spinner("Calculating Anadihilo Dynamics..."):
        h_pos, h_vel, h_acc = run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3)
        P_vals = [p1, p2, p3]
        
        # Colors & Names
        sci_colors = ['#E63946', '#457B9D', '#2A9D8F'] 
        names = ["Body 1 (Sun)", "Body 2 (Earth)", "Body 3 (Moon)"]
        
        tab1, tab2, tab3 = st.tabs(["🌌 Interactive Animation", "📊 Scientific Dashboard", "💾 Data Logs"])
        
        # --- TAB 1: 3D ANIMATION ---
        with tab1:
            st.write("### N-Body Trajectory Visualization")
            
            # Optimization
            skip = 20 if speed == "Normal" else 50 if speed == "Hyper" else 10
            duration = 20
            
            fig = go.Figure()
            
            # Static Trails
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2],
                    mode='lines', name=names[i],
                    line=dict(color=sci_colors[i], width=4)
                ))
            
            # Dynamic Markers
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]],
                    mode='markers', name=names[i],
                    marker=dict(color=sci_colors[i], size=[20, 10, 6][i])
                ))
            
            # Frames
            frames = []
            for k in range(0, steps, skip):
                frames.append(go.Frame(data=[
                    go.Scatter3d(x=[h_pos[k,0,0]], y=[h_pos[k,0,1]], z=[h_pos[k,0,2]]),
                    go.Scatter3d(x=[h_pos[k,1,0]], y=[h_pos[k,1,1]], z=[h_pos[k,1,2]]),
                    go.Scatter3d(x=[h_pos[k,2,0]], y=[h_pos[k,2,1]], z=[h_pos[k,2,2]])
                ], traces=[3, 4, 5]))
            
            fig.frames = frames
            fig.update_layout(
                scene=dict(bgcolor="black", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                paper_bgcolor="black", height=600, margin=dict(l=0,r=0,b=0,t=0),
                # FIXED VISIBLE PLAY BUTTON
                updatemenus=[dict(
                    type="buttons", showactive=False, x=0.5, y=0.05, xanchor="center",
                    bgcolor="white", bordercolor="#00e5ff", borderwidth=2,
                    buttons=[dict(label="▶ PLAY ANIMATION", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: SCIENTIFIC DASHBOARD (Matplotlib) ---
        with tab2:
            st.write("### Static Scientific Analysis")
            
            # 1. Trajectory Map
            st.subheader("1. 2D Projection Map")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            
            for i in range(3):
                # Path
                ax1.plot(h_pos[:, i, 0], h_pos[:, i, 1], color=sci_colors[i], label=f"{names[i]} (P={P_vals[i]:.3f})", linewidth=2)
                # Final Circle
                final_x, final_y = h_pos[-1, i, 0], h_pos[-1, i, 1]
                circ = Circle((final_x, final_y), radius=np.max(h_pos)*0.015, color=sci_colors[i], alpha=0.8)
                ax1.add_patch(circ)
            
            ax1.set_xlabel("X (Meters)", color='white'); ax1.set_ylabel("Y (Meters)", color='white')
            ax1.tick_params(colors='white'); ax1.grid(True, ls='--', alpha=0.2)
            ax1.legend(facecolor='#111', labelcolor='white')
            ax1.set_aspect('equal')
            for spine in ax1.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig1)

            # 2. Dynamics
            st.subheader("2. Velocity & Acceleration")
            fig2, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig2.patch.set_facecolor('#0e1117')
            t_axis = np.arange(steps)
            
            for i in range(3):
                axes[0].plot(t_axis, h_vel[:, i], color=sci_colors[i], label=names[i])
                axes[1].plot(t_axis, h_acc[:, i], color=sci_colors[i], label=names[i])
                
            axes[0].set_title("Velocity Profile", color='white')
            axes[1].set_title("Acceleration (Finite Peaks)", color='white')
            
            for ax in axes:
                ax.set_facecolor('#0e1117'); ax.tick_params(colors='white'); ax.grid(alpha=0.2)
                ax.legend(facecolor='#111', labelcolor='white')
                for spine in ax.spines.values(): spine.set_edgecolor('white')
            
            st.pyplot(fig2)

        # --- TAB 3: DATA ---
        with tab3:
            df = pd.DataFrame({'Step': range(steps)})
            for i, name in enumerate(names):
                df[f'{name}_X'] = h_pos[:, i, 0]
                df[f'{name}_Y'] = h_pos[:, i, 1]
                df[f'{name}_Vel'] = h_vel[:, i]
                df[f'{name}_Acc'] = h_acc[:, i]
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Log (CSV)", csv, "anadihilo_log.csv", "text/csv")

else:
    st.info("👋 Select Input Mode & Parameters above, then click 'EXECUTE'.")
