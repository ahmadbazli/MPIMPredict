import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MPIM and RK4 functions
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

st.title("Prediction using Modified Picard Iterative Method (Manual Data Entry)")

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

    # Use the first Deaths value as initial value
    y0 = float(df['Deaths'].iloc[0])
    st.write(f"**Initial Value (from first row, Year {df['Year'].iloc[0]}):** {y0:.2f}")

    # Estimate growth rate (but do NOT display it)
    time = np.arange(len(df))
    deaths = df['Deaths'].astype(float).values
    log_deaths = np.log(deaths)
    r, log_y0 = np.polyfit(time, log_deaths, 1)

    # User chooses how many years to predict
    n_future = st.number_input("Number of future years to predict", min_value=1, value=5)
    total_time = np.arange(len(df) + n_future)
    dt = 1  # yearly step

    def user_ode(y, t):
        return np.array([r * y[0]])

    if st.button("Predict"):
        mpim_y = modified_picard(user_ode, [y0], total_time, num_iter=3).flatten()
        rk4_y = rk4(user_ode, [y0], total_time).flatten()

        pred_years = list(df['Year']) + [f"Future {i+1}" for i in range(n_future)]
        result_df = pd.DataFrame({
            'Year': pred_years,
            'MPIM': mpim_y,
            'RK4': rk4_y
        })

        st.write("### Prediction Results")
        st.dataframe(result_df)

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(pred_years, mpim_y, marker='o', label='MPIM')
        ax.plot(pred_years, rk4_y, marker='s', label='RK4')
        ax.scatter(df['Year'], df['Deaths'], color='black', label='Actual Data')
        ax.legend()
        ax.set_xticklabels(pred_years, rotation=45)
        st.pyplot(fig)

else:
    st.info("Please enter at least two rows of data for prediction.")

st.markdown("---\n*Manually enter your data to generate predictions. The system uses the first Deaths value as the initial value and automatically forecasts the future values.*")
