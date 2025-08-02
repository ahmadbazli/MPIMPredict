import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="PicardPredict: Innovative Differential Equation Prediction", layout="wide")

# ----------------- Picard Iterative Method (General) ----------------------
def picard_iterative_method(f, y0, t0, T, h, tol=1e-6, max_iter=1000):
    """
    Picard Iterative Method for y' = f(t, y), iterating until convergence at each time step.
    Returns (t_values, y_values, iter_counts, error_traces)
    """
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    iter_counts = []
    error_traces = []

    t = t0
    y = np.array(y0, dtype=float)
    while t < T:
        y_n = y.copy()
        errors = []
        for it in range(1, max_iter + 1):
            y_new = y + h * np.array(f(t, y_n))
            error = np.linalg.norm(y_new - y_n)
            errors.append(error)
            if error < tol:
                break
            y_n = y_new
        else:
            st.warning(f"Did not converge in {max_iter} Picard iterations at t={t:.4f}")
        t += h
        y = y_n
        t_values.append(t)
        y_values.append(y.copy())
        iter_counts.append(it)
        error_traces.append(errors)
    return np.array(t_values), np.array(y_values), np.array(iter_counts), error_traces

# ------------- Model Library (for innovation wow factor) -----------------
def get_model_info(choice):
    if choice == "SIR Epidemic (Infectious Disease)":
        description = "A classic epidemiology model: S (Susceptible), I (Infected), R (Recovered)"
        code = "lambda t, y: [-beta*y[0]*y[1], beta*y[0]*y[1] - gamma*y[1], gamma*y[1]]"
        variables = ["S", "I", "R"]
        params = {"beta": 0.3, "gamma": 0.1}
        y0 = [0.99, 0.01, 0.0]
    elif choice == "Logistic Growth (Population)":
        description = "Logistic population growth: dP/dt = r*P*(1-P/K)"
        code = "lambda t, y: [r*y[0]*(1 - y[0]/K)]"
        variables = ["P"]
        params = {"r": 0.5, "K": 10.0}
        y0 = [0.5]
    elif choice == "Custom (write your own)":
        description = "Write any ODE system with variables t (time) and y (array of current values)"
        code = "lambda t, y: [your_equation_here]"
        variables = ["y1"]
        params = {}
        y0 = [1.0]
    else:
        description = "Simple ODE: dy/dt = -2y"
        code = "lambda t, y: [-2*y[0]]"
        variables = ["y"]
        params = {}
        y0 = [1.0]
    return description, code, variables, params, y0

# ---------------------- Streamlit UI -------------------------------------
st.title("🚀 PicardPredict: Data-Driven Differential Equation Prediction System")
st.caption("Innovation for Science, Engineering, and Education - Solve, Visualize, and Learn with the Picard Iterative Method!")

tab1, tab2, tab3 = st.tabs(["🔬 Simulation", "📈 Live Convergence", "📚 Learn Picard"])

with tab1:
    st.header("1️⃣ Choose or Define a Model")
    model_list = [
        "SIR Epidemic (Infectious Disease)",
        "Logistic Growth (Population)",
        "Simple ODE: dy/dt = -2y",
        "Custom (write your own)"
    ]
    model_choice = st.selectbox("Select a model", model_list, index=0)
    description, code_example, var_names, params, y0_default = get_model_info(model_choice)
    st.markdown(f"**Description:** {description}")
    st.markdown("**Right-hand side of ODE system** (as lambda function):")
    ode_str = st.text_input("ODE system (Python lambda):", value=code_example)
    st.markdown("**Initial conditions (as Python list, order matches variables):**")
    y0_str = st.text_input("Initial values:", value=str(y0_default))

    # Dynamic parameter input
    st.markdown("**Model parameters (if any):**")
    param_dict = {}
    for param, default in params.items():
        param_dict[param] = st.number_input(f"{param}:", value=float(default), key=param)

    # Show which variable is which
    st.markdown("**Variables:** " + ", ".join([f"y{i+1} ({name})" for i, name in enumerate(var_names)]))

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
            # Prepare function string with parameter replacement (for library models)
            func_str = ode_str
            for param, val in param_dict.items():
                func_str = func_str.replace(param, str(val))
            f = eval(func_str)
            y0 = eval(y0_str)
            t_vals, y_vals, iter_counts, error_traces = picard_iterative_method(
                f, y0, t0, T, h, tol=tolerance)

            labels = [f"y{i+1}" + (f" ({name})" if name != f"y{i+1}" else "") for i, name in enumerate(var_names)]

            st.success("Simulation completed!")

            st.subheader("Results Table")
            df = pd.DataFrame(y_vals, columns=labels)
            df.insert(0, "t", t_vals)
            df["Picard Iter"] = np.append([np.nan], iter_counts)
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

            st.subheader("Picard Iterations per Time Step")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(t_vals[1:], iter_counts, marker='o', color='tab:orange')
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Iterations")
            ax2.set_title("Number of Picard Iterations until Convergence")
            st.pyplot(fig2)

            with tab2:
                st.header("📈 Live Convergence Per Step")
                step = st.slider("Pick a time step to inspect:", 1, len(error_traces), 1)
                st.write(f"Convergence at t = {t_vals[step]:.3f}")
                err_arr = error_traces[step-1]
                fig3, ax3 = plt.subplots()
                ax3.plot(range(1, len(err_arr)+1), err_arr, marker='o')
                ax3.set_yscale('log')
                ax3.set_xlabel("Picard Iteration")
                ax3.set_ylabel("Error (log scale)")
                ax3.set_title(f"Convergence to Tolerance (t={t_vals[step]:.3f})")
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"Error in ODE or initial values: {e}")

with tab3:
    st.header("📚 What is the Picard Iterative Method?")
    st.markdown("""
    - The Picard Iterative Method is a numerical method for solving differential equations.
    - At each time step, it successively refines the solution until it converges to a stable value within your set tolerance.
    - This system lets you see how the method works **live**, step-by-step!
    - It's used in many fields: science, engineering, epidemiology, and finance.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Iterative_methods_ODE.png", width=400)
    st.info("""
    **Try different models, initial values, and tolerances.  
    Watch how the number of iterations and convergence change with your settings!**
    """)
    st.markdown("---")
    st.markdown("""
    **Made for innovation. If you like it, use and share!**
    """)

st.caption("© 2025 PicardPredict | Empowering Differential Solutions for All")
