import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Modified Picard Iterative Method ---------
def modified_picard(f, y0, t, num_iter=3):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y_approx = y[i-1].copy()
        for _ in range(num_iter):
            # Simple Euler-style integration (replace with quadrature for higher accuracy if desired)
            y_new = y[i-1] + h * f(y_approx, t[i-1])
            y_approx = y_new
        y[i] = y_approx
    return y

# --------- Fourth Order Runge-Kutta Method (RK4) ---------
def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(y[i-1], t[i-1])
        k2 = f(y[i-1] + 0.5*h*k1, t[i-1] + 0.5*h)
        k3 = f(y[i-1] + 0.5*h*k2, t[i-1] + 0.5*h)
        k4 = f(y[i-1] + h*k3, t[i-1] + h)
        y[i] = y[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# --------- Example ODE system: Exponential Growth ---------
def example_ode(y, t):
    # dy/dt = r * y
    r = 0.1
    return np.array([r * y[0]])

# --------- Streamlit Interface ---------
st.set_page_config(page_title="Prediction using Modified Picard Iterative Method", layout="centered")
st.title("Prediction using Modified Picard Iterative Method (with RK4 Benchmark)")

st.markdown("""
**Upload your initial value and select your simulation settings. This demo uses a simple exponential growth ODE.**
""")

col1, col2 = st.columns(2)
with col1:
    y0 = st.number_input("Initial value (y₀)", min_value=0.0, value=100.0)
with col2:
    r = st.number_input("Growth rate (r)", value=0.1)
    
t_end = st.number_input("Prediction time (t₁)", min_value=1.0, value=10.0)
dt = st.number_input("Step size (Δt)", min_value=0.01, value=0.1)
num_iter = st.slider("Number of Picard Iterations", min_value=1, max_value=10, value=3)

t = np.arange(0, t_end+dt, dt)
# Use user input for ODE
def user_ode(y, t):
    return np.array([r * y[0]])

if st.button("Predict"):
    # Run both solvers
    mpim_y = modified_picard(user_ode, [y0], t, num_iter=num_iter).flatten()
    rk4_y = rk4(user_ode, [y0], t).flatten()

    df = pd.DataFrame({
        'Time': t,
        'MPIM': mpim_y,
        'RK4': rk4_y
    })
    
    st.markdown("### Results Comparison")
    st.line_chart(df.set_index('Time'))

    st.write("**Prediction Table:**")
    st.dataframe(df)

    st.markdown("**Download Results:**")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download as CSV", data=csv, file_name='prediction_results.csv', mime='text/csv')

    st.markdown("""
    **Method Explanation:**
    - *Modified Picard Iterative Method* (MPIM) is an iterative solver for ODEs.
    - *Fourth Order Runge-Kutta* (RK4) is the classical numerical benchmark.
    """)

else:
    st.info("Enter values and click Predict to see the result.")

st.markdown("""
---
*This is a demo system. You can adapt the code to solve more complex ODE systems by editing the ODE function section.*
""")
