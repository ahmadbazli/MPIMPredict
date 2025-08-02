import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="PicardPredict: Editable Compartment Model System", layout="wide")

# ----------------- Picard Iterative Method (General) ----------------------
def picard_iterative_method(f, y0, t0, T, h, tol=1e-6, max_iter=1000):
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]

    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        y_n = y.copy()
        for _ in range(1, max_iter + 1):
            y_new = y + h * np.array(f(t, y_n))
            if np.linalg.norm(y_new - y_n) < tol:
                break
            y_n = y_new
        t += h
        y = y_n
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.array(y_values)

# ------------- Model Library (for starting templates) -----------------
def get_model_info(choice):
    if choice == "SIR Epidemic (Infectious Disease)":
        description = "A classic SIR model: S (Susceptible), I (Infected), R (Recovered)"
        compartments = ["S", "I", "R"]
        parameters = ["beta", "gamma"]
        param_defaults = {"beta": 0.3, "gamma": 0.1}
        y0_default = [0.99, 0.01, 0.0]
        code_example = (
            "lambda t, y: [\n"
            "    -beta*y[0]*y[1],\n"
            "    beta*y[0]*y[1] - gamma*y[1],\n"
            "    gamma*y[1]\n"
            "]"
        )
    elif choice == "Logistic Growth (Population)":
        description = "Logistic growth: dP/dt = r*P*(1-P/K)"
        compartments = ["P"]
        parameters = ["r", "K"]
        param_defaults = {"r": 0.5, "K": 10.0}
        y0_default = [0.5]
        code_example = (
            "lambda t, y: [\n"
            "    r*y[0]*(1 - y[0]/K)\n"
            "]"
        )
    elif choice == "Custom (write your own)":
        description = "Write any ODE system with variables t (time), y (vector), and your parameters."
        compartments = ["y1"]
        parameters = []
        param_defaults = {}
        y0_default = [1.0]
        code_example = "lambda t, y: [\n    -2*y[0]\n]"
    else:
        description = "Simple ODE: dy/dt = -2y"
        compartments = ["y"]
        parameters = []
        param_defaults = {}
        y0_default = [1.0]
        code_example = "lambda t, y: [\n    -2*y[0]\n]"
    return description, compartments, parameters, param_defaults, y0_default, code_example

# ---------------------- Streamlit UI -------------------------------------
st.title("🚀 PicardPredict: Editable Compartment Model System")
st.caption("For flexible epidemic, population, and engineering modeling with Picard's Method")

tab1, tab2 = st.tabs(["🔬 Simulation", "📚 Learn Picard"])

with tab1:
    st.header("1️⃣ Choose a Template or Start from Scratch")
    model_list = [
        "SIR Epidemic (Infectious Disease)",
        "Logistic Growth (Population)",
        "Custom (write your own)"
    ]
    model_choice = st.selectbox("Select a model template", model_list, index=0)
    description, base_compartments, base_params, param_defaults, y0_default, code_example = get_model_info(model_choice)
    st.markdown(f"**Description:** {description}")

    st.markdown("#### Compartments (e.g., S, E, I, R):")
    compartments = st.text_input("Compartments (comma separated):", value=", ".join(base_compartments))
    compartment_list = [c.strip() for c in compartments.split(",") if c.strip()]
    n_compartments = len(compartment_list)

    st.markdown("#### Parameters (e.g., beta, gamma, alpha):")
    param_string = st.text_input("Parameters (comma separated):", value=", ".join(base_params))
    param_list = [p.strip() for p in param_string.split(",") if p.strip()]

    st.markdown("#### Set Parameters:")
    param_dict = {}
    for param in param_list:
        default = param_defaults[param] if param in param_defaults else 1.0
        param_dict[param] = st.number_input(f"{param}:", value=float(default), key=param)

    st.markdown("#### Set Initial Values:")
    y0_inputs = []
    for i, c in enumerate(compartment_list):
        default_val = y0_default[i] if i < len(y0_default) else 0.0
        val = st.number_input(f"Initial value for {c}:", value=float(default_val), key=f"init_{c}")
        y0_inputs.append(val)

    st.markdown(
        "**Variables in y:**  \n" +
        ", ".join([f"y[{i}] = {name}" for i, name in enumerate(compartment_list)])
    )

    st.markdown("#### Write your ODE system (as a Python lambda):")
    ode_str = st.text_area(
        "ODE system (use t, y, and your parameter names):",
        value=code_example, height=100
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        t0 = st.number_input("Initial time (t0)", value=0.0)
    with col2:
        T = st.number_input("End time (T)", value=10.0)
    with col3:
        h = st.number_input("Step size (h)", value=0.1, min_value=0.01, step=0.01, format="%.2f")

    tolerance = st.number_input("Convergence tolerance (Picard)", value=1e-6, format="%.1e")

    st.markdown("---")
    if st.button("🚦 Run Picard Simulation"):
        try:
            # Replace param names with values in code string
            func_str = ode_str
            for param, val in param_dict.items():
                func_str = func_str.replace(param, str(val))
            f = eval(func_str)
            y0 = y0_inputs
            t_vals, y_vals = picard_iterative_method(f, y0, t0, T, h, tol=tolerance)

            labels = [f"{name}" for name in compartment_list]

            st.success("Simulation completed!")

            st.subheader("Results Table")
            df = pd.DataFrame(y_vals, columns=labels)
            df.insert(0, "t", t_vals)
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="picard_predict_results.csv")

            st.subheader("Solution Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, label in enumerate(labels):
                ax.plot(t_vals, y_vals[:, i], label=label)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Solution Trajectories (Picard Method)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in ODE function or initial values: {e}")

with tab2:
    st.header("📚 What is the Picard Iterative Method?")
    st.markdown("""
    - The Picard Iterative Method is a numerical method for solving differential equations.
    - At each time step, it successively refines the solution until it converges to a stable value within your set tolerance.
    - This system lets you use the method on any ODE or system of ODEs you choose.
    - It's used in many fields: science, engineering, epidemiology, and finance.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Iterative_methods_ODE.png", width=400)
    st.info("""
    **Try different models, initial values, and tolerances.  
    Explore how the Picard Iterative Method adapts to your system!**
    """)
    st.markdown("---")
    st.markdown("""
    **Made for innovation. If you like it, use and share!**
    """)

st.caption("© 2025 PicardPredict | Empowering Differential Solutions for All")
