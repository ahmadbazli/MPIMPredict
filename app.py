import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Picard Iterative Method Solver (Custom ODE)", layout="wide")

# ------------------- Picard Iterative Method ----------------------

def picard_iterative_method(f, y0, t0, T, h, tol=1e-6, max_iter=1000):
    """
    Picard Iterative Method for y' = f(t, y), iterating until convergence at each time step.
    Returns (t_values, y_values, iter_counts)
    """
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    iter_counts = []

    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        y_n = y.copy()
        for it in range(1, max_iter + 1):
            y_new = y + h * np.array(f(t, y_n))
            if np.linalg.norm(y_new - y_n) < tol:
                break
            y_n = y_new
        else:
            st.warning(f"Did not converge in {max_iter} Picard iterations at t={t:.4f}")
        t += h
        y = y_n
        t_values.append(t)
        y_values.append(y.copy())
        iter_counts.append(it)
    return np.array(t_values), np.array(y_values), np.array(iter_counts)

# -------------------- Streamlit UI ----------------------

st.title("Picard Iterative Method Solver (Custom ODE)")

st.markdown("""
Write your ODE system below as a Python lambda function in variables `t` and `y`:

- **For a single ODE:**<br>
  Example: `lambda t, y: [-2*y[0]]` &nbsp;&nbsp; (for \( y' = -2y \))
- **For a system:**<br>
  Example: `lambda t, y: [y[1], -y[0]]` &nbsp;&nbsp; (for \( y_1' = y_2,\ y_2' = -y_1 \))
- **Initial conditions:** as a list, e.g. `[1.0]` or `[1.0, 0.0]`
""", unsafe_allow_html=True)

ode_str = st.text_input("Right-hand side of ODE system (lambda function):", value="lambda t, y: [-2*y[0]]")
y0_str = st.text_input("Initial conditions (as list):", value="[1.0]")

col1, col2, col3 = st.columns(3)
with col1:
    t0 = st.number_input("Initial time (t0)", value=0.0)
with col2:
    T = st.number_input("End time (T)", value=5.0)
with col3:
    h = st.number_input("Step size (h)", value=0.1, min_value=0.01, step=0.01, format="%.2f")

tolerance = st.number_input("Convergence tolerance", value=1e-6, format="%.1e")

if st.button("Run Picard Method"):
    try:
        # Security warning: use eval ONLY in trusted environments!
        f = eval(ode_str)
        y0 = eval(y0_str)
        t_vals, y_vals, iter_counts = picard_iterative_method(f, y0, t0, T, h, tol=tolerance)
        labels = [f"y{i+1}" for i in range(len(y0))]

        st.subheader("Solution Table")
        df = pd.DataFrame(y_vals, columns=labels)
        df.insert(0, "t", t_vals)
        df["Iterations"] = np.append([np.nan], iter_counts)  # First time step has no iteration
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

        st.subheader("Picard Iterations per Step")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(t_vals[1:], iter_counts, marker='o')
        ax2.set_xlabel("t")
        ax2.set_ylabel("Iterations")
        ax2.set_title("Number of Picard Iterations until Convergence")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.write("Developed for demonstration. Always check your ODE format! © 2025")

