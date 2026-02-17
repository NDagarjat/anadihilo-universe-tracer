import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Server stability fix
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="wide", page_icon="🔭")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Play Button High Contrast */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1c1f26; border-radius: 5px; color: #fff; }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
    
    /* Hide Default Elements */
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")
st.markdown("**Status:** Ready | **Engine:** Strict Anadihilo Logic | **Visuals:** 3D + Scientific 2D")

# --- INPUTS ---
with st.expander("⚙️ CONFIGURE PARAMETERS (Click to Expand)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Simulation Steps", 5000, 50000, 15000)
    with col2:
        speed = st.select_slider("Animation Speed", options=["Precision", "Normal", "Hyper"], value="Normal")
    
    st.markdown("---")
    
    # Body Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Core 1 (Sun)**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
    
    with c2:
        st.markdown("🔵 **Core 2 (Earth)**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X Pos", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Y Vel", value=29780.0, key="vy2")
    
    with c3:
        st.markdown("⚪ **Core 3 (Moon)**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        # Auto-calc default for ease
        def_x3 = x2 + 3.844e8
        def_vy3 = vy2 + 1022.0
        x3 = st.number_input("X Pos", value=def_x3, format="%.2e", key="x3")
        vy3 = st.number_input("Y Vel", value=def_vy3, key="vy3")

# --- PHYSICS ENGINE (User's Exact Logic) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Constants
    K_constant = 3.98e14 
    dt = 3600 # 1 Hour
    
    # Init Arrays
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
        
        # Handover Logic (Assimilation)
        eff_P = P_vals.copy()
        parents = [-1, -1, -1]
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if P_vals[j] > P_vals[i]:
                    if np.linalg.norm(pos[i] - pos[j]) < 1.0e10:
                        eff_P[i] = P_vals[j]
                        parents[i] = j

        # Force Loop (User's Formula)
        for j in range(3):
            net_pull = np.zeros(3)
            for k in range(3):
                if j == k: continue
                
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                
                # --- SINGULARITY RESOLUTION ---
                epsilon = 1.0 / (P_vals[j] + P_vals[k])
                
                # Priority Magnitude
                priority_mag = P_vals[k] / (r_mag**2 + epsilon)
                net_pull += priority_mag * (r_vec / (r_mag + 1e-18))
            
            # Final Accel = Priority / Friction
            fric = P_vals[j] if parents[j] == k else eff_P[j]
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
    
    with st.spinner("Processing Dynamics..."):
        h_pos, h_vel, h_acc = run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3)
        P_vals = [p1, p2, p3] # For labels
        
        # Colors from User's Code
        sci_colors = ['#E63946', '#457B9D', '#2A9D8F'] # Red, Blue, Teal
        names = ["Sun", "Earth", "Moon"]
        
        tab1, tab2, tab3 = st.tabs(["🌌 3D Animation", "📊 2D Scientific Dashboard", "💾 Data Logs"])
        
        # --- TAB 1: 3D ANIMATION (Plotly) ---
        with tab1:
            st.write("### Interactive 3D System")
            
            # Optimization: Skip frames for smooth play
            skip = 20 if speed == "Normal" else 50 if speed == "Hyper" else 10
            duration = 20
            
            fig = go.Figure()
            
            # Static Paths
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
            
            # Animation Frames
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
                updatemenus=[dict(
                    type="buttons", showactive=False, x=0.5, y=0.1, xanchor="center",
                    bgcolor="white", bordercolor="#00e5ff", borderwidth=2,
                    buttons=[dict(label="▶ PLAY ANIMATION", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: SCIENTIFIC DASHBOARD (Your Matplotlib Code) ---
        with tab2:
            st.write("### Static Scientific Analysis")
            
            # 1. TRAJECTORY MAP (As per your code)
            st.subheader("1. Trajectory Map (2D Projection)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            
            for i in range(3):
                # Path
                ax1.plot(h_pos[:, i, 0], h_pos[:, i, 1], color=sci_colors[i], label=f"{names[i]} (P={P_vals[i]})", linewidth=2)
                # Final Position Circle (Your feature)
                final_x, final_y = h_pos[-1, i, 0], h_pos[-1, i, 1]
                circ = Circle((final_x, final_y), radius=np.max(h_pos)*0.02, color=sci_colors[i], alpha=0.8)
                ax1.add_patch(circ)
                # Text Label
                ax1.text(final_x, final_y, f" {names[i]}", color=sci_colors[i], fontsize=8, fontweight='bold')

            ax1.set_title("Scientific Trajectory Map", color='white')
            ax1.set_xlabel("X (Meters)", color='white'); ax1.set_ylabel("Y (Meters)", color='white')
            ax1.tick_params(colors='white'); ax1.grid(True, ls='--', alpha=0.2)
            ax1.legend(facecolor='#111', labelcolor='white')
            ax1.set_aspect('equal')
            for spine in ax1.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig1)

            # 2. DYNAMICS (Velocity & Accel)
            st.subheader("2. Dynamics Dashboard")
            fig2, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig2.patch.set_facecolor('#0e1117')
            t_axis = np.arange(steps)
            
            # Velocity
            for i in range(3):
                axes[0].plot(t_axis, h_vel[:, i], color=sci_colors[i], label=names[i])
            axes[0].set_title("Velocity Profile", color='white')
            axes[0].set_facecolor('#0e1117'); axes[0].tick_params(colors='white'); axes[0].grid(alpha=0.2)
            axes[0].legend(facecolor='#111', labelcolor='white')
            for spine in axes[0].spines.values(): spine.set_edgecolor('white')
            
            # Acceleration
            for i in range(3):
                axes[1].plot(t_axis, h_acc[:, i], color=sci_colors[i], label=names[i])
            axes[1].set_title("Acceleration (Finite Peaks)", color='white')
            axes[1].set_facecolor('#0e1117'); axes[1].tick_params(colors='white'); axes[1].grid(alpha=0.2)
            for spine in axes[1].spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig2)

            # 3. STATS BAR CHART (Peak Velocity)
            st.subheader("3. Peak Velocity Comparison")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            fig3.patch.set_facecolor('#0e1117'); ax3.set_facecolor('#0e1117')
            
            max_vels = [np.max(h_vel[:, i]) for i in range(3)]
            ax3.bar(names, max_vels, color=sci_colors, alpha=0.8)
            
            ax3.set_title("Peak Systemic Velocity", color='white')
            ax3.tick_params(colors='white'); ax3.grid(axis='y', alpha=0.2)
            for spine in ax3.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig3)

        # --- TAB 3: DATA LOGS ---
        with tab3:
            df = pd.DataFrame({'Step': range(steps)})
            for i, name in enumerate(names):
                df[f'{name}_X'] = h_pos[:, i, 0]
                df[f'{name}_Y'] = h_pos[:, i, 1]
                df[f'{name}_Vel'] = h_vel[:, i]
                df[f'{name}_Acc'] = h_acc[:, i]
            
            st.dataframe(df.head(100))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Log (CSV)", csv, "anadihilo_log.csv", "text/csv")

else:
    st.info("👋 Set Parameters above and click 'EXECUTE SCIENTIFIC TRACER'")
