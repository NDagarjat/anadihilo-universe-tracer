import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="wide", page_icon="🔭")

# --- CUSTOM CSS (Visibility Fix) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Play Button & Widget High Contrast */
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #00e5ff !important;
        font-weight: bold;
    }
    .updatemenu-item-text { color: #000 !important; }
    
    /* Streamlit Button Styling */
    div.stButton > button { 
        background-color: #00e5ff; color: #000; font-weight: bold; border: none;
    }
    div.stButton > button:hover { background-color: #ffffff; }
    
    /* Headers */
    h1, h2, h3 { color: #00e5ff !important; font-family: 'Courier New', monospace; }
    
    /* Hide Streamlit Default Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")
st.markdown("**Status:** Ready | **Visualization:** Combined N-Body System | **Logic:** User Defined")

# --- INPUTS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE SYSTEM PARAMETERS (Click to Expand)", expanded=True):
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
        # Sun is anchor at 0,0
    
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

# --- PHYSICS ENGINE (Based on User's Python Code) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Constants from user code
    K_constant = 3.98e14 # Adjusted for scale
    dt = 3600 # 1 Hour
    
    # Arrays for 3 bodies
    P_vals = np.array([p1, p2, p3])
    pos = np.array([
        [0.0, 0.0, 0.0],      # Sun
        [x2, 0.0, 0.0],       # Earth
        [x3, 0.0, 0.0]        # Moon
    ], dtype=np.float64)
    
    vel = np.array([
        [0.0, 0.0, 0.0],      # Sun
        [0.0, vy2, 0.0],      # Earth
        [0.0, vy3, 0.0]       # Moon
    ], dtype=np.float64)
    
    # History Storage
    hist_pos = np.zeros((steps, 3, 3)) # Steps, Bodies, Coords
    hist_vel = np.zeros((steps, 3))    # Steps, Bodies (Magnitude)
    hist_acc = np.zeros((steps, 3))    # Steps, Bodies (Magnitude)
    
    for s in range(steps):
        hist_pos[s] = pos
        
        # Calculate Accelerations
        acc = np.zeros_like(pos)
        
        # Determine Handover/Parents (Logic from previous requests integrated)
        eff_P = P_vals.copy()
        parents = [-1, -1, -1]
        
        # Handover Check
        for i in range(3):
            for j in range(3):
                if i == j: continue
                if P_vals[j] > P_vals[i]:
                    if np.linalg.norm(pos[i] - pos[j]) < 1.0e10:
                        eff_P[i] = P_vals[j]
                        parents[i] = j

        # Force Loop
        for j in range(3):
            net_pull = np.zeros(3)
            for k in range(3):
                if j == k: continue
                
                r_vec = pos[k] - pos[j]
                r_mag = np.linalg.norm(r_vec)
                
                # --- ANADIHILO SINGULARITY RESOLUTION ---
                # epsilon = 1.0 / (P_vals[j] + P_vals[k])
                epsilon = 1.0 / (P_vals[j] + P_vals[k])
                
                # Friction Logic
                fric = P_vals[j] if parents[j] == k else eff_P[j]
                
                # Formula
                priority_mag = P_vals[k] / (r_mag**2 + epsilon)
                net_pull += priority_mag * (r_vec / (r_mag + 1e-18))
            
            acc[j] = (K_constant / fric) * net_pull
        
        # Update
        vel += acc * dt
        pos += vel * dt
        
        # Store Metrics
        for b in range(3):
            hist_vel[s, b] = np.linalg.norm(vel[b])
            hist_acc[s, b] = np.linalg.norm(acc[b])
            
    return hist_pos, hist_vel, hist_acc

# --- EXECUTION ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    
    with st.spinner("Processing N-Body Dynamics..."):
        h_pos, h_vel, h_acc = run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3)
        
        # Define P_vals explicitly for plotting scope (This fixes the NameError)
        P_vals = [p1, p2, p3] 
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["🌌 Interactive Animation", "📈 Scientific Analysis", "💾 Data Logs"])
        
        # --- TAB 1: COMBINED ANIMATION ---
        with tab1:
            st.write("### Combined System Trajectory (Interactive)")
            
            # Downsampling for smooth animation
            skip = 10 if speed == "Normal" else 50 if speed == "Hyper" else 5
            duration = 20
            
            fig = go.Figure()
            
            colors = ['#ffcc00', '#0099ff', '#aaaaaa']
            names = ['Sun', 'Earth', 'Moon']
            sizes = [25, 12, 6]
            
            # 1. Static Trails (Full Path)
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=h_pos[:, i, 0], y=h_pos[:, i, 1], z=h_pos[:, i, 2],
                    mode='lines', name=f'{names[i]} Path',
                    line=dict(color=colors[i], width=3)
                ))
            
            # 2. Dynamic Bodies (Start Position)
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[h_pos[0, i, 0]], y=[h_pos[0, i, 1]], z=[h_pos[0, i, 2]],
                    mode='markers', name=names[i],
                    marker=dict(color=colors[i], size=sizes[i])
                ))
            
            # 3. Frames
            frames = []
            for k in range(0, steps, skip):
                frame_data = []
                for i in range(3):
                    # We only update the markers (indices 3, 4, 5 in data list)
                    frame_data.append(go.Scatter3d(x=[h_pos[k, i, 0]], y=[h_pos[k, i, 1]], z=[h_pos[k, i, 2]]))
                frames.append(go.Frame(data=frame_data, traces=[3, 4, 5]))
            
            fig.frames = frames
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="X", backgroundcolor="black", gridcolor="#333"),
                    yaxis=dict(title="Y", backgroundcolor="black", gridcolor="#333"),
                    zaxis=dict(title="Z", backgroundcolor="black", gridcolor="#333"),
                    bgcolor="black"
                ),
                paper_bgcolor="black",
                font=dict(color="white"),
                height=650,
                margin=dict(l=0, r=0, b=0, t=0),
                # HIGH CONTRAST PLAY BUTTON
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.5, y=0.1, xanchor="center",
                    bgcolor="white", bordercolor="#00e5ff", borderwidth=2,
                    font=dict(color="black", size=14, family="Arial Black"),
                    buttons=[dict(label="▶ PLAY ANIMATION", method="animate", args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Tip: Use mouse to Rotate/Zoom. Click 'PLAY ANIMATION' (White Button) to start.")

        # --- TAB 2: SCIENTIFIC ANALYSIS (Matplotlib Recreated) ---
        with tab2:
            st.write("### Static Scientific Dashboard")
            
            # Colors for Matplotlib
            mpl_colors = ['#E63946', '#457B9D', '#2A9D8F']
            mpl_names = ["Sun", "Earth", "Moon"]
            
            # 1. TRAJECTORY MAP (2D Projection)
            st.subheader("1. Trajectory Map (2D Projection)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            
            for i in range(3):
                # P_vals is now correctly defined above
                ax1.plot(h_pos[:, i, 0], h_pos[:, i, 1], color=mpl_colors[i], label=f"{mpl_names[i]} (P={P_vals[i]})", linewidth=2)
                # Final Position Circle
                circ = Circle((h_pos[-1, i, 0], h_pos[-1, i, 1]), radius=np.max(h_pos)*0.02, color=mpl_colors[i], alpha=0.8)
                ax1.add_patch(circ)
            
            ax1.set_xlabel("X Position", color='white')
            ax1.set_ylabel("Y Position", color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, ls='--', alpha=0.2)
            ax1.legend(facecolor='#111', labelcolor='white')
            ax1.set_aspect('equal')
            for spine in ax1.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig1)
            
            # 2. DYNAMICS (Velocity & Accel)
            st.subheader("2. Systemic Dynamics")
            fig2, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig2.patch.set_facecolor('#0e1117')
            
            t_axis = np.arange(steps)
            
            # Velocity Plot
            for i in range(3):
                axes[0].plot(t_axis, h_vel[:, i], color=mpl_colors[i], label=mpl_names[i])
            axes[0].set_title("Velocity Profile", color='white')
            axes[0].set_facecolor('#0e1117')
            axes[0].tick_params(colors='white')
            axes[0].grid(alpha=0.2)
            axes[0].legend(facecolor='#111', labelcolor='white')
            for spine in axes[0].spines.values(): spine.set_edgecolor('white')
            
            # Acceleration Plot
            for i in range(3):
                axes[1].plot(t_axis, h_acc[:, i], color=mpl_colors[i], label=mpl_names[i])
            axes[1].set_title("Acceleration (Finite Peaks)", color='white')
            axes[1].set_facecolor('#0e1117')
            axes[1].tick_params(colors='white')
            axes[1].grid(alpha=0.2)
            for spine in axes[1].spines.values(): spine.set_edgecolor('white')
            
            st.pyplot(fig2)

        # --- TAB 3: DATA LOGS ---
        with tab3:
            st.write("### Detailed Data Logs")
            
            df = pd.DataFrame({'Step': range(steps)})
            for i, name in enumerate(names):
                df[f'{name}_X'] = h_pos[:, i, 0]
                df[f'{name}_Y'] = h_pos[:, i, 1]
                df[f'{name}_Vel'] = h_vel[:, i]
                df[f'{name}_Acc'] = h_acc[:, i]
            
            st.dataframe(df.head(100))
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Scientific Log (CSV)",
                data=csv,
                file_name="anadihilo_full_scientific_log.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👋 Set Parameters above and click 'EXECUTE SCIENTIFIC TRACER'")
