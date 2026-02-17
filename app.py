import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Anadihilo Core", layout="wide", page_icon="🛰️")

# --- CSS FOR UI & PERFORMANCE ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    
    /* Play Button Container - Moved Down */
    .updatemenu-button {
        background-color: #00e5ff !important;
        color: #000 !important;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Streamlit widgets */
    div.stButton > button { 
        width: 100%; border: 1px solid #00e5ff; color: #00e5ff; background: transparent;
    }
    div.stButton > button:hover { background-color: #00e5ff; color: #000; }
    
    /* Hide extra elements */
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ ANADIHILO DYNAMICS: SCIENTIFIC TRACER")
st.markdown("**Logic:** Strict PDF Implementation ($a_j = K/P_{eff} \dots$) | **Performance:** Optimized for Mobile")

# --- SETTINGS (EXPANDER) ---
with st.expander("⚙️ CONFIGURE PHYSICS (Click to Expand)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Physics Accuracy (Steps)", 5000, 30000, 15000)
    with col2:
        speed = st.select_slider("Animation Speed", options=["Slow", "Normal", "Fast"], value="Normal")

    st.markdown("---")
    
    # Body Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🟡 **Sun**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
    with c2:
        st.markdown("🔵 **Earth**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("X (m)", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vy (m/s)", value=29780.0, key="vy2")
    with c3:
        st.markdown("⚪ **Moon**")
        p3 = st.number_input("P (Dg)", value=0.054, format="%.3f", key="p3")
        def_x3 = x2 + 3.844e8
        def_vy3 = vy2 + 1022.0
        x3 = st.number_input("X (m)", value=def_x3, format="%.2e", key="x3")
        vy3 = st.number_input("Vy (m/s)", value=def_vy3, key="vy3")

# --- PHYSICS ENGINE (STRICT LOGIC) ---
@st.cache_data(show_spinner=False)
def calculate_dynamics(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # [span_1](start_span)PDF Constants[span_1](end_span)
    K = 3.98e14
    DT = 3600
    
    # Init Arrays
    P = np.array([p1, p2, p3])
    pos = np.zeros((steps, 3, 3)) # History: [Step, Body, Coord]
    
    # Initial State
    current_pos = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    current_vel = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=float)
    
    # MAIN LOOP
    for s in range(steps):
        pos[s] = current_pos
        
        # 1. [span_2](start_span)Handover / Assimilation[span_2](end_span)
        # Moon(2) adopts Earth(1) friction if close
        eff_P = P.copy()
        parents = [-1, -1, -1]
        
        # Check Earth-Moon Proximity (10 Million km limit)
        dist_em = np.linalg.norm(current_pos[1] - current_pos[2])
        if dist_em < 1.0e10: 
            if P[1] > P[2]: 
                eff_P[2] = P[1] # Moon assimilates to Earth's P
                parents[2] = 1

        # 2. [span_3](start_span)Force Calculation (Eq 6)[span_3](end_span)
        acc = np.zeros_like(current_pos)
        
        for j in range(3):
            net_force = np.zeros(3)
            for k in range(3):
                if j == k: continue
                
                r_vec = current_pos[k] - current_pos[j]
                r_sq = np.sum(r_vec**2)
                d = np.sqrt(r_sq)
                
                # [span_4](start_span)Epsilon (Singularity Fix)[span_4](end_span)
                epsilon = 1.0 / (P[j] + P[k])
                
                # Anadihilo Force Term
                term = P[k] / (r_sq + epsilon)
                net_force += term * (r_vec / d)
            
            # [span_5](start_span)Friction Application (Handover Logic)[span_5](end_span)
            # If j is child of k, use internal P. Else use Effective P.
            friction = P[j] if parents[j] == k else eff_P[j]
            acc[j] = (K / friction) * net_force
            
        # 3. Update
        current_vel += acc * DT
        current_pos += current_vel * DT
        
    return pos, P

# --- VISUALIZATION ---
if st.button("🚀 EXECUTE TRACER", use_container_width=True):
    
    with st.spinner("Processing Physics..."):
        h_pos, P_vals = calculate_dynamics(steps, p1, p2, x2, vy2, p3, x3, vy3)
        
        # --- OPTIMIZATION (THE FIX) ---
        # We only animate 500 frames max, regardless of steps.
        # This prevents the browser from crashing.
        skip_rate = max(1, steps // 400)
        
        # Animation Duration
        frame_dur = 10 if speed == "Fast" else 30 if speed == "Normal" else 60
        
        tab1, tab2, tab3 = st.tabs(["🌌 Interactive Map", "📈 Static Dashboard", "💾 Data"])
        
        # --- TAB 1: INTERACTIVE PLOT ---
        with tab1:
            st.write("### N-Body System Trajectory")
            
            fig = go.Figure()
            colors = ['#ffcc00', '#0099ff', '#aaaaaa']
            names = ['Sun', 'Earth', 'Moon']
            
            # 1. Static Trails (Always visible, cheap to render)
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2],
                    mode='lines', name=f'{names[i]} Path',
                    line=dict(color=colors[i], width=3)
                ))
            
            # 2. Dynamic Markers (The moving dots)
            for i in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]],
                    mode='markers', name=names[i],
                    marker=dict(color=colors[i], size=[20, 10, 5][i])
                ))
            
            # 3. Frames (Optimized)
            frames = []
            for k in range(0, steps, skip_rate):
                frame_data = []
                for i in range(3):
                    frame_data.append(go.Scatter3d(x=[h_pos[k,i,0]], y=[h_pos[k,i,1]], z=[h_pos[k,i,2]]))
                frames.append(go.Frame(data=frame_data, traces=[3, 4, 5])) # Only update markers
                
            fig.frames = frames
            
            # Layout with Play Button separated
            fig.update_layout(
                scene=dict(bgcolor="black", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                paper_bgcolor="black", height=600, margin=dict(l=0,r=0,b=0,t=0),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=-0.1, x=0.5, xanchor="center", # MOVED DOWN
                    pad={"t": 20}, # Added Padding
                    buttons=[dict(label="▶ PLAY ANIMATION", method="animate", args=[None, dict(frame=dict(duration=frame_dur, redraw=True), fromcurrent=True)])]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: STATIC DASHBOARD (Matplotlib) ---
        with tab2:
            st.write("### Scientific Dashboard (Static)")
            
            fig_mpl, ax = plt.subplots(figsize=(10, 6))
            fig_mpl.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            cols = ['#ffcc00', '#0099ff', '#aaaaaa']
            for i in range(3):
                ax.plot(h_pos[:,i,0], h_pos[:,i,1], color=cols[i], label=names[i], lw=1.5)
                # Final pos
                ax.plot(h_pos[-1,i,0], h_pos[-1,i,1], 'o', color=cols[i])
                
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(facecolor='black', labelcolor='white')
            st.pyplot(fig_mpl)
            st.info("Top-down 2D Projection of the 3D data.")

        # --- TAB 3: DATA ---
        with tab3:
            df = pd.DataFrame({'Step': range(steps)})
            for i, name in enumerate(names):
                df[f'{name}_X'] = h_pos[:,i,0]
                df[f'{name}_Y'] = h_pos[:,i,1]
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Log (CSV)", csv, "anadihilo_data.csv", "text/csv")

else:
    st.info("Ready. Click 'EXECUTE TRACER' to calculate.")
