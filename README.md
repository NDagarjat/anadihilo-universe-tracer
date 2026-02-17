# 🌌 Anadihilo Universe Tracer
### Deterministic Resolution of the N-Body Problem via Anadihilo Dynamics

![License](https://img.shields.io/badge/License-GPLv3-blue.svg) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📖 Overview
The **Anadihilo Universe Tracer** is a high-precision 3D simulation engine designed to demonstrate the **Anadihilo Dynamics Framework**. Unlike standard Newtonian mechanics which break down at singularities ($r \to 0$), this tool applies the **Dagar Mass Intensity ($Dg$)** scale and **Systemic Boundary ($n$)** logic to resolve orbital chaos into deterministic informational updates.

This tool provides a "crash-proof" simulation of celestial mechanics, implementing the **Axiom of Normalization** and **Priority Handover Logic** to maintain stability in complex 3-body systems (e.g., Sun-Earth-Moon).

---

## 🔬 Theoretical Foundation
This simulation is not based on traditional mass weights (kg) but on **Grid Friction** and **Informational Density**.

### Key Formulas Implemented:
1.  **Dagar Intensity ($Dg$):** Mass is defined as the friction a core exerts on the universal grid ($i=0.0001$).
    $$\Omega(Dg) = \frac{n}{0.8}$$
    *(Where $n$ is the Systemic Boundary in meters)*

2.  **The Singularity Killer ($\epsilon$):**
    To prevent infinite force at collision coordinates:
    $$\epsilon = \frac{1}{P_j + P_k}$$

3.  **Anadihilo Acceleration:**
    $$\vec{a}_j = \frac{K}{P_j} \sum \left( \frac{P_k}{r^2 + \epsilon} \right) \cdot \hat{r}$$

---

## 📚 Read the Full Paper
For a complete understanding of the mathematical proofs, the derivation of the Grid Constant ($i$), and the ontological foundation of the Absolute Void ($\infty$), please refer to the published research paper.

> **📄 Paper Title:** Anadihilo Dynamics: The Final Resolution of the 3-Body Problem and the Dagar (Dg) Scale of Mass Intensity
>
> **🔗 DOI (Access Full Text):** [https://doi.org/10.5281/zenodo.18604635](https://doi.org/10.5281/zenodo.18604635)

---

## 🚀 Features
* **3D Interactive Visualization:** Rotate, zoom, and explore orbits in a 3D space (NASA-style dark mode).
* **Strict Physics Engine:** Runs purely on Anadihilo formulas ($Dg$, $n$, $\epsilon$). No curve fitting.
* **Handover/Assimilation Logic:** Demonstrates how lighter bodies (like the Moon) adopt the friction of their parent body (Earth) to remain in stable orbit.
* **Universal Inputs:** Accepts inputs in either **Dagar Units (Dg)** or **Boundary Meters (n)**.
* **Data Export:** Download full simulation logs (X, Y, Z coordinates) in CSV format.
* **GIF Generation:** Create and download animations of your simulation.

---

## 💻 Installation & Usage

You can run this tool locally or deploy it via Streamlit Cloud.

### Prerequisites
* Python 3.8+
* pip

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/anadihilo-universe-tracer.git](https://github.com/YOUR_USERNAME/anadihilo-universe-tracer.git)
    cd anadihilo-universe-tracer
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 📜 Citation Policy
This software is the intellectual property of **Nitin Dagar**. If you use this code, the **Dagar (Dg)** unit system, or the **Anadihilo** framework in your research, videos, or projects, **you must cite the original paper**:

**BibTeX:**
```bibtex
@article{dagar2026anadihilo,
  title={Anadihilo Dynamics: The Final Resolution of the 3-Body Problem},
  author={Dagar, Nitin},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18604635},
  url={[https://doi.org/10.5281/zenodo.18604635](https://doi.org/10.5281/zenodo.18604635)}
}
