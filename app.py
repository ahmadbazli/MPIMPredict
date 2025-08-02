import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- Custom CSS for an amazing look --
st.markdown("""
    <style>
        .main {background-color: #f5f9fc;}
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #e5ebf8 !important;
            color: #0a3967 !important;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            margin-right: 5px;
            min-width: 150px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #c9e7fa 40%, #a2cef4 100%) !important;
            color: #124076 !important;
        }
        .stDownloadButton {color: #003c7a;}
        .stButton button {background-color: #124076; color: white; font-weight: bold; border-radius: 12px;}
        .stAlert {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="PicardPredict: Modern ODE App", layout="wide")

# -------- Sidebar: Branding, Navigation, and Controls ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Iterative_methods_ODE.png", width=120)
st.sidebar.title("🚀 PicardPredict")
st.sidebar.caption("Elegant ODE Modeling\nPicard • Euler • RK4")

model_list = [
    "SIR Epidemic (Infectious Disease)",
    "Logistic Growth (Population)",
    "Custom (write your own)"
]
model_choice = st.sidebar.selectbox("🧬 Model Template", model_list, index=0)
def get_model_info(choice):
    if choice == "SIR Epidemic (Infectious Disease)":
        desc = "A classic SIR model: S (Susceptible), I (Infected), R (Recovered)"
        comps = ["S", "I", "R"]
        params = ["beta", "gamma"]
        param_defs = {"beta": 0.3, "gamma": 0.1}
        y0_def = [0.99, 0.01, 0.0]
        code_ex = (
            "lambda t, y: [\n"
            "    -beta*y[0]*y[1],\n"
            "    beta*y[0]*y[1] - gamma*y[1],\n"
            "    gamma*y[1]\n"
            "]"
        )
    elif choice == "Logistic Growth (Population)":
        desc = "Logistic growth: dP/dt = r*P*(1-P/K)"
        comps = ["P"]
        params = ["r", "K"]
        param_defs = {"r": 0.5, "K": 10.0}
        y0_def = [0.5]
        code_ex = (
            "lambda t, y: [\n"
            "    r*y[0]*(1 - y[0]/K)\n"
            "]"
        )
    else:
        desc = "Write any ODE system: use t, y (vector), and your parameter names."
        comps = ["y1"]
        params = []
        param_defs = {}
        y0_def = [1.0]
        code_ex = "lambda t, y: [\n    -2*y[0]\n]"
    return desc, comps, params, param_defs, y0_def, code_ex
desc, base_comps, base_params, param_defs, y0_def, code_ex = get_model_info(model_choice)
st.sidebar.info(desc)

st.sidebar.markdown("**Compartments:**")
comps = st.sidebar.text_input("e.g. S, E, I, R", value=", ".join(base_comps))
comp_list = [c.strip() for c in comps.split(",") if c.strip()]
params = st.sidebar.text_input("Parameters", value=", ".join(base_params))
param_list = [p.strip() for p in params.split(",") if p.strip()]
st.sidebar.markdown("**Set parameter values:**")
param_dict = {param: st.sidebar.number_input(f"{param}:", value=float(param_defs[param]) if param in param_defs else 1.0, key=param) for param in param_list}

st.sidebar.markdown("**Initial values:**")
y0_inputs = []
for i, c in enumerate(comp_list):
    default_val = y0_def[i] if i < len(y0_def) else 0.0
    val = st.sidebar.number_input(f"{c}₀:", value=float(default_val), key=f"init_{c}")
    y0_inputs.append(val)

st.sidebar.markdown("---")
t0 = st.sidebar.number_input("Start time", value=0.0)
T = st.sidebar.number_input("End time", value=10.0)
h = st.sidebar.number_input("Step size h", value=0.1, min_value=0.01, step=0.01, format="%.2f")
tol = st.sidebar.number_input("Tolerance", value=1e-6, format="%.1e")
run = st.sidebar.button("✨ Run Simulation")

# ----------- Main Layout: Tabs and Results -------------------
tab1, tab2, tab3 = st.tabs([
    "🧮 Simulation Results", 
    "🛡️ Population Check", 
    "📘 About & Learn"
])

with tab1:
    st.markdown("<h2 style='color:#124076;'>🧮 Simulation Results</h2>", unsafe_allow_html=True)
    st.markdown(f"<b>Variables:</b> " + ", ".join([f"y[{i}] = {c}" for i, c in enumerate(comp_list)]), unsafe_allow_html=True)
    st.markdown("#### Write your ODE system (as a Python lambda):")
    ode_str = st.text_area("ODE system (use t, y, and your parameters):", value=code_ex, height=100,
        help="Example for SIR: lambda t, y: [-beta*y[0]*y[1], beta*y[0]*y[1] - gamma*y[1], gamma*y[1]]")

    # -- Numerical Methods --
    def picard_iterative_method(f, y0, t0, T, h, tol=1e-6, max_iter=1000):
        t_values, y_values, t, y = [t0], [np.array(y0, dtype=float)], t0, np.array(y0, dtype=float)
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
    def euler_method(f, y0, t0, T, h):
        t_values, y_values, t, y = [t0], [np.array(y0, dtype=float)], t0, np.array(y0, dtype=float)
        while t < T:
            y = y + h * np.array(f(t, y))
            t += h
            t_values.append(t)
            y_values.append(y.copy())
        return np.array(t_values), np.array(y_values)
    def rk4_method(f, y0, t0, T, h):
        t_values, y_values, t, y = [t0], [np.array(y0, dtype=float)], t0, np.array(y0, dtype=float)
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

    if run:
        try:
            func_str = ode_str
            for param, val in param_dict.items():
                func_str = func_str.replace(param, str(val))
            f = eval(func_str)
            y0 = y0_inputs
            t_vals, y_picard = picard_iterative_method(f, y0, t0, T, h, tol=tol)
            t_euler, y_euler = euler_method(f, y0, t0, T, h)
            t_rk4, y_rk4 = rk4_method(f, y0, t0, T, h)
            labels = [f"{name}" for name in comp_list]
            st.success("Simulation complete!")

            # ---- Plotting Results
            st.markdown("<h4 style='color:#124076;'>Solution Plot</h4>", unsafe_allow_html=True)
            color_cycle = plt.get_cmap('tab10').colors
            fig, ax = plt.subplots(figsize=(12, 6))
            methods = [("Picard", t_vals, y_picard, "-"),
                       ("Euler", t_euler, y_euler, "--"),
                       ("RK4", t_rk4, y_rk4, ":")]
            for i, label in enumerate(labels):
                for m, t_m, y_m, ls in methods:
                    ax.plot(t_m, y_m[:, i], ls, label=f"{label} ({m})", color=color_cycle[i%10])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Solution Trajectories: Picard, Euler, RK4")
            ax.grid(True, ls=":")
            ax.legend(frameon=True, ncol=3)
            st.pyplot(fig)

            # --- Results Table
            with st.expander("Show Results Table (Picard)", expanded=True):
                df = pd.DataFrame(y_picard, columns=labels)
                df.insert(0, "t", t_vals)
                st.dataframe(df, height=400)
                st.download_button("📥 Download Picard CSV", data=df.to_csv(index=False), file_name="picard_predict_results.csv")

            # --- Euler & RK4 Tables
            with st.expander("Show Results Table (Euler)"):
                df_euler = pd.DataFrame(y_euler, columns=labels)
                df_euler.insert(0, "t", t_euler)
                st.dataframe(df_euler)
                st.download_button("📥 Download Euler CSV", data=df_euler.to_csv(index=False), file_name="euler_results.csv")
            with st.expander("Show Results Table (RK4)"):
                df_rk4 = pd.DataFrame(y_rk4, columns=labels)
                df_rk4.insert(0, "t", t_rk4)
                st.dataframe(df_rk4)
                st.download_button("📥 Download RK4 CSV", data=df_rk4.to_csv(index=False), file_name="rk4_results.csv")

            # --- Warnings/Info
            if np.isnan(np.min(y_picard)) or np.isnan(np.max(y_picard)):
                st.warning("⚠️ Solution contains NaN values (may be unstable or diverged).")
            elif np.min(y_picard) < -1e-6:
                st.warning("⚠️ Some compartment values became negative. Check your model/parameters.")

        except Exception as e:
            st.error(f"❌ Error in ODE function or initial values: {e}")

with tab2:
    st.markdown("<h2 style='color:#124076;'>🛡️ Population/Total Check</h2>", unsafe_allow_html=True)
    if run:
        try:
            st.markdown("**Total Population (Sum of Compartments) for Each Method**")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(t_vals, np.sum(y_picard, axis=1), '-', label="Picard", color='#1877c0')
            ax2.plot(t_euler, np.sum(y_euler, axis=1), '--', label="Euler", color='#e67d1d')
            ax2.plot(t_rk4, np.sum(y_rk4, axis=1), ':', label="RK4", color='#2bbf7a')
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Total")
            ax2.set_title("Population Conservation Check")
            ax2.legend()
            st.pyplot(fig2)
        except:
            st.info("Please run the simulation first.")

with tab3:
    st.markdown("<h2 style='color:#124076;'>📘 About & Learn</h2>", unsafe_allow_html=True)
    st.markdown("""
    - The Picard Iterative Method is a robust, intuitive method for solving differential equations.
    - Benchmarking with Euler and RK4 lets you see and trust your result.
    - Change your model, step size, or tolerance and instantly see the effect!
    - Export your results for further analysis, teaching, or publication.
    ---
    **Made for innovation, research, and education!**
    """)
    st.caption("© 2025 PicardPredict | Empowering Differential Solutions for All")
