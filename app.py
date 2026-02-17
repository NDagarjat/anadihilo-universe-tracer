import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- PAGE CONFIGURATION (3:4 Frame Feel) ---
st.set_page_config(page_title="Anadihilo Scientific Tracer", layout="centered", page_icon="🔭")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .updatemenu-button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00e5ff !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container {padding-top: 1rem; max-width: 850px;}
</style>
""", unsafe_allow_html=True)

st.title("🔭 ANADIHILO SCIENTIFIC TRACER")

# --- INPUTS ---
with st.expander("⚙️ CONFIGURE PARAMETERS", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        input_type = st.radio("Select Input Unit:", ["Mass Intensity (Dg)", "Systemic Boundary (n)"])
    with col_b:
        steps = st.slider("Steps", 5000, 50000, 20000)
    with col_c:
        speed = st.select_slider("Speed", options=["Precision", "Normal", "Hyper"], value="Normal")

    st.markdown("---")
    
    def get_p_val(label, key, def_dg):
        if input_type == "Systemic Boundary (n)":
            val = st.number_input(f"{label} - n", value=float(def_dg*0.8), format="%.2e", key=key)
            return val / 0.8
        else:
            return st.number_input(f"{label} - P (Dg)", value=float(default_dg), format="%.4f", key=key)

    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown("🟡 **Body 1 (Sun)**")
        p1 = st.number_input("P (Dg)", value=1480000.0, key="p1")
    with c2: 
        st.markdown("🔵 **Body 2 (Earth)**")
        p2 = st.number_input("P (Dg)", value=4.44, key="p2")
        x2 = st.number_input("Pos X", value=1.496e11, format="%.2e", key="x2")
        vy2 = st.number_input("Vel Y", value=29780.0, format="%.1f", key="vy2")
    with c3: 
        st.markdown("⚪ **Body 3 (Moon)**")
        p3 = st.number_input("P (Dg)", value=0.054, key="p3")
        x3 = st.number_input("Pos X ", value=1.49984e11, format="%.2e", key="x3")
        vy3 = st.number_input("Vel Y ", value=30802.0, format="%.1f", key="vy3")

# --- PHYSICS ENGINE (STRICT LOGIC) ---
@st.cache_data(show_spinner=False)
def run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3):
    # Responsible Scaling: Detects if it's Solar or Custom scale
    K = 3.98e14 if x2 > 1e6 else 1.0
    dt = 3600 if x2 > 1e6 else 0.002
    
    P = np.array([p1, p2, p3])
    pos = np.array([[0.0,0,0], [x2,0,0], [x3,0,0]], dtype=float)
    vel = np.array([[0.0,0,0], [0,vy2,0], [0,vy3,0]], dtype=float)
    
    h_pos, h_vel, h_acc = np.zeros((steps, 3, 3)), np.zeros((steps, 3)), np.zeros((steps, 3))
    
    for s in range(steps):
        h_pos[s] = pos
        acc = np.zeros_like(pos)
        
        # Handover Logic
        eff_P = P.copy()
        parents = [-1]*3
        for i in range(3):
            for j in range(3):
                if i != j and P[j] > P[i]:
                    if np.linalg.norm(pos[i]-pos[j]) < (x2 * 0.1 if x2 > 0 else 1.0):
                        eff_P[i], parents[i] = P[j], j

        # Force Calculation
        for j in range(3):
            net = np.zeros(3)
            for k in range(3):
                if j == k: continue
                r_vec, r_mag = pos[k]-pos[j], np.linalg.norm(pos[k]-pos[j])
                eps = 1.0 / (P[j] + P[k])
                net += (P[k]/(np.sum(r_vec**2) + eps)) * (r_vec/(r_mag + 1e-18))
            
            fric = P[j] if parents[j] == k else eff_P[j]
            acc[j] = (K/fric) * net
        
        vel += acc * dt
        pos += vel * dt
        for b in range(3): h_vel[s,b], h_acc[s,b] = np.linalg.norm(vel[b]), np.linalg.norm(acc[b])
            
    return h_pos, h_vel, h_acc

# --- EXECUTION ---
if st.button("🚀 EXECUTE SCIENTIFIC TRACER", use_container_width=True):
    h_pos, h_vel, h_acc = run_simulation(steps, p1, p2, x2, vy2, p3, x3, vy3)
    
    tab1, tab2, tab3 = st.tabs(["🌌 3D Animation", "📈 Dashboard", "💾 Logs"])
    
    with tab1:
        skip = max(1, steps // 400)
        fig = go.Figure()
        cols = ['#ffcc00', '#0099ff', '#aaaaaa']
        names = ["Sun", "Earth", "Moon"]
        
        for i in range(3):
            fig.add_trace(go.Scatter3d(x=h_pos[:,i,0], y=h_pos[:,i,1], z=h_pos[:,i,2], mode='lines', line=dict(color=cols[i], width=4), name=names[i]))
            fig.add_trace(go.Scatter3d(x=[h_pos[0,i,0]], y=[h_pos[0,i,1]], z=[h_pos[0,i,2]], mode='markers', marker=dict(color=cols[i], size=[18, 9, 5][i]), showlegend=False))

        frames = [go.Frame(data=[go.Scatter3d(x=[h_pos[k,i,0]], y=[h_pos[k,i,1]], z=[h_pos[k,i,2]]) for i in range(3)], traces=[3, 4, 5]) for k in range(0, steps, skip)]
        fig.frames = frames
        fig.update_layout(scene=dict(bgcolor="black", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
                          paper_bgcolor="black", height=750, margin=dict(l=0,r=0,b=0,t=0), uirevision='constant',
                          updatemenus=[dict(type="buttons", showactive=False, x=0.5, y=-0.05, buttons=[
                              dict(label="▶ PLAY", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]),
                              dict(label="⏸ PAUSE", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                          ])])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_mpl, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e1117')
        for i in range(3):
            axes[0].plot(h_pos[:, i, 0], h_pos[:, i, 1], color=['yellow','blue','grey'][i], label=names[i])
            axes[1].plot(h_vel[:, i], color=['yellow','blue','grey'][i])
            final_x, final_y = h_pos[-1, i, 0], h_pos[-1, i, 1]
            axes[0].add_patch(Circle((final_x, final_y), radius=np.max(np.abs(h_pos))*0.015, color=['yellow','blue','grey'][i], alpha=0.6))
        for ax in axes: ax.set_facecolor('#0e1117'); ax.tick_params(colors='white'); ax.grid(alpha=0.1); ax.legend()
        st.pyplot(fig_mpl)

    with tab3:
        df = pd.DataFrame({'Step': range(steps), 'B1_X': h_pos[:,0,0], 'B2_X': h_pos[:,1,0], 'B3_X': h_pos[:,2,0]})
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "anadihilo_log.csv")
