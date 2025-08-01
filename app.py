import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MPIM and RK4 for exponential model
def modified_picard(f, y0, t, num_iter=3):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y_approx = y[i-1].copy()
        for _ in range(num_iter):
            y_new = y[i-1] + h * f(y_approx, t[i-1])
            y_approx = y_new
        y[i] = y_approx
    return y

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

st.title("Prediction using Curve Fit, MPIM, and RK4")

st.write("Enter your data below (Year and Deaths). Add rows as needed.")

# Example starter data (user can edit)
data = {
    "Year": [2000, 2001, 2002, 2003, 2004],
    "Deaths": [100707, 104531, 110367, 112744, 113192]
}

df = st.data_editor(
    pd.DataFrame(data),
    num_rows="dynamic",
    use_container_width=True,
    key="input_table"
)

if len(df) > 1 and not df['Deaths'].isnull().any() and not df['Year'].isnull().any():
    st.write("### Data Preview")
    st.dataframe(df)

    years = df['Year'].astype(int).values
    deaths = df['Deaths'].astype(float).values

    # --- Polynomial Curve Fit ---
    degree = st.selectbox("Select polynomial degree for curve fitting:", [1, 2, 3], index=1)
    coefs = np.polyfit(years, deaths, degree)
    poly = np.poly1d(coefs)

    n_future = st.number_input("Number of future years to predict", min_value=1, value=5)
    last_year = years[-1]
    pred_years = np.arange(years[0], last_year + n_future + 1)
    pred_deaths_poly = poly(pred_years)

    # --- Exponential Fit for MPIM & RK4 ---
    time = np.arange(len(df))
    log_deaths = np.log(deaths)
    r, log_y0 = np.polyfit(time, log_deaths, 1)
    y0 = float(df['Deaths'].iloc[0])

    total_time = np.arange(len(df) + n_future)

    def exp_ode(y, t):
        return np.array([r * y[0]])

    if st.button("Predict"):
        # --- MPIM and RK4 Predictions ---
        mpim_y = modified_picard(exp_ode, [y0], total_time, num_iter=3).flatten()
        rk4_y = rk4(exp_ode, [y0], total_time).flatten()

        # Prepare table
        pred_labels = list(df['Year']) + [f"Future {i+1}" for i in range(n_future)]
        result_df = pd.DataFrame({
            'Year': pred_labels,
            'Actual': list(deaths) + [np.nan]*n_future,
            f'Poly Fit (deg {degree})': pred_deaths_poly,
            'MPIM (Exp)': mpim_y,
            'RK4 (Exp)': rk4_y
        })

        st.write("### Prediction Results (All Methods)")
        st.dataframe(result_df)

        # Plot all predictions
        fig, ax = plt.subplots()
        ax.scatter(df['Year'], deaths, color='black', label='Actual Data')
        ax.plot(pred_years, pred_deaths_poly, color='blue', linestyle='-', marker='o', label=f'Polynomial Fit (deg {degree})')
        ax.plot(pred_labels, mpim_y, color='green', linestyle='--', marker='s', label='MPIM (Exp)')
        ax.plot(pred_labels, rk4_y, color='orange', linestyle='-.', marker='^', label='RK4 (Exp)')
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("Deaths")
        ax.set_xticks(range(0, len(pred_labels), max(1, len(pred_labels)//10)))
        ax.set_xticklabels([str(y) for y in pred_labels], rotation=45)
        st.pyplot(fig)

else:
    st.info("Please enter at least two rows of data for prediction.")

st.markdown("""
---
- **Actual Data:** Black dots
- **Polynomial Fit:** Blue line (most accurate to your data)
- **MPIM and RK4:** Exponential prediction for benchmark comparison
""")
