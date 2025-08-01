import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Picard Iterative Method (General) ----------------------

def picard_iterative_method(f, y0, t0, T, h, max_iter=5, tol=1e-6):
    """
    General Picard Iterative Method for y' = f(t, y)
    """
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]

    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        # Picard iteration at each time step
        y_n = y.copy()
        for _ in range(max_iter):
            y_new = y + h * np.array(f(t, y_n))
            if np.linalg.norm(y_new - y_n) < tol:
                break
            y_n = y_new
        t += h
        y = y_n
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.array(y_values)

# --------------------- Streamlit UI ------------------------------------------

st.set_page_config(page_title="Picard Iterative Method Solver", layout="wide")

st.title("Picard Iterative Method Solver (General ODE System)")
st.write("""
Solve any ODE or system of ODEs using the Picard Iterative Method.  
Select a sample equation/system or add your own code in `f(t, y)` (for advanced users).
""")

example = st.selectbox(
    "Choose an ODE/system example:",
    [
        "Single ODE: y' = -2y, y(0)=1",
        "System: y1' = y2, y2' = -y1, y1(0)=1, y2(0)=0 (Simple Harmonic)",
        "System: y1' = -y1 + y2, y2' = -y2, y1(0)=2, y2(0)=1"
    ]
)

if example == "Single ODE: y' = -2y, y(0)=1":
    def f(t, y):
        return [-2 * y[0]]
    y0 = [1.0]
    labels = ["y"]
elif example == "System: y1' = y2, y2' = -y1, y1(0)=1, y2(0)=0 (Simple Harmonic)":
    def f(t, y):
        y1, y2 = y
        return [y2, -y1]
    y0 = [1.0, 0.0]
    labels = ["y1", "y2"]
elif example == "System: y1' = -y1 + y2, y2' = -y2, y1(0)=2, y2(0)=1":
    def f(t, y):
        y1, y2 = y
        return [-y1 + y2, -y2]
    y0 = [2.0, 1.0]
    labels = ["y1", "y2"]

st.subheader("Simulation Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    t0 = st.number_input("Initial time (t0)", value=0.0)
with col2:
    T = st.number_input("End time (T)", value=5.0)
with col3:
    h = st.number_input("Step size (h)", value=0.1, min_value=0.01, step=0.01, format="%.2f")

col4, col5 = st.columns(2)
with col4:
    max_iter = st.number_input("Picard max iterations", value=5, min_value=1, step=1)
with col5:
    tol = st.number_input("Convergence tolerance", value=1e-6, format="%.1e")

if st.button("Run Picard Iterative Method"):
    t_vals, y_vals = picard_iterative_method(f, y0, t0, T, h, int(max_iter), tol)

    st.subheader("Solution Table")
    import pandas as pd
    df = pd.DataFrame(y_vals, columns=labels)
    df.insert(0, "t", t_vals)
    st.dataframe(df)

    st.subheader("Solution Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        ax.plot(t_vals, y_vals[:, i], label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("Solution")
    ax.set_title("Picard Iterative Method Solution")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.write("Developed for educational and demonstration purposes. © 2025")

