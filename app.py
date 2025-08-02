import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="PicardPredict: Editable Compartment Model System", layout="wide")

# ----- Picard Iterative Method -----
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

# ----- Euler Method -----
def euler_method(f, y0, t0, T, h):
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        y = y + h * np.array(f(t, y))
        t += h
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.array(y_values)

# ----- RK4 Method -----
def rk4_method(f, y0, t0, T, h):
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + h/2, y + h/2 * k1))
        k3 = np.array(f(t + h/2, y + h/2 * k2))
        k4 = np.array(f(t + h, y + h * k3))
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
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
st.caption("Flexible, trustworthy ODE modeling: Picard, Euler, and RK4, side by side.")

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
    if st.button("🚦 Run Picard & Benchmark"):
        try:
            # Replace param names with values in code string
            func_str = ode_str
            for param, val in param_dict.items():
                func_str = func_str.replace(param, str(val))
            f = eval(func_str)
            y0 = y0_inputs

            t_vals, y_picard = picard_iterative_method(f, y0, t0, T, h, tol=tolerance)
            t_euler, y_euler = euler_method(f, y0, t0, T, h)
            t_rk4, y_rk4 = rk4_method(f, y0, t0, T, h)

            labels = [f"{name}" for name in compartment_list]

            st.success("Simulation completed!")

            st.subheader("Results Table (Picard)")
            df = pd.DataFrame(y_picard, columns=labels)
            df.insert(0, "t", t_vals)
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Picard CSV", data=csv, file_name="picard_predict_results.csv")

            # Results Table (Euler)
            with st.expander("Show Euler Results Table"):
                df_euler = pd.DataFrame(y_euler, columns=labels)
                df_euler.insert(0, "t", t_euler)
                st.dataframe(df_euler)
                st.download_button("📥 Download Euler CSV", data=df_euler.to_csv(index=False), file_name="euler_results.csv")

            # Results Table (RK4)
            with st.expander("Show RK4 Results Table"):
                df_rk4 = pd.DataFrame(y_rk4, columns=labels)
                df_rk4.insert(0, "t", t_rk4)
                st.dataframe(df_rk4)
                st.download_button("📥 Download RK4 CSV", data=df_rk4.to_csv(index=False), file_name="rk4_results.csv")

            # Solution Plot
            st.subheader("Solution Plot: Picard, Euler, RK4")
            fig, ax = plt.subplots(figsize=(12, 6))
            linestyles = ["-", "--", ":"]
            methods = [("Picard", t_vals, y_picard, "-"),
                       ("Euler", t_euler, y_euler, "--"),
                       ("RK4", t_rk4, y_rk4, ":")]
            for i, label in enumerate(labels):
                for method_name, t_method, y_method, ls in methods:
                    ax.plot(t_method, y_method[:, i], ls, label=f"{label} ({method_name})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Solution Trajectories: Picard, Euler, RK4")
            ax.legend()
            st.pyplot(fig)

            # Total Population Plot
            st.subheader("Total (Sum of Compartments)")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(t_vals, np.sum(y_picard, axis=1), '-', label="Picard")
            ax2.plot(t_euler, np.sum(y_euler, axis=1), '--', label="Euler")
            ax2.plot(t_rk4, np.sum(y_rk4, axis=1), ':', label="RK4")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Total")
            ax2.set_title("Population Conservation Check")
            ax2.legend()
            st.pyplot(fig2)

            # Warnings
            min_picard = np.min(y_picard)
            max_picard = np.max(y_picard)
            if np.isnan(min_picard) or np.isnan(max_picard):
                st.warning("Warning: Solution contains NaN values (may be unstable or diverged).")
            elif min_picard < -1e-6:
                st.warning("Warning: Some compartment values became negative. Check your model/parameters.")

        except Exception as e:
            st.error(f"Error in ODE function or initial values: {e}")

with tab2:
    st.header("📚 What is the Picard Iterative Method?")
    st.markdown("""
    - The Picard Iterative Method is a numerical method for solving differential equations.
    - At each time step, it successively refines the solution until it converges to a stable value within your set tolerance.
    - This system lets you use the method on any ODE or system of ODEs you choose.
    - For trust and benchmarking, you can compare with both Euler and Runge-Kutta 4th order (RK4) solutions!
    - It's used in many fields: science, engineering, epidemiology, and finance.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Iterative_methods_ODE.png", width=400)
    st.info("""
    **Try different models, initial values, and tolerances.  
    Explore how the Picard, Euler, and RK4 methods compare!**
    """)
    st.markdown("---")
    st.markdown("""
    **Made for innovation. If you like it, use and share!**
    """)

st.caption("© 2025 PicardPredict | Empowering Differential Solutions for All")
