import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Prediction using Polynomial Fit (Manual Data Entry)")

st.write("Enter your data below (Year and Deaths). Add rows as needed.")

# Example starter data
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

    # Years as numbers for fitting
    years = df['Year'].astype(int).values
    deaths = df['Deaths'].astype(float).values

    # Fit a polynomial (degree 2 or user-selected)
    degree = st.selectbox("Select polynomial degree for fitting:", [1, 2, 3], index=1)
    coefs = np.polyfit(years, deaths, degree)
    poly = np.poly1d(coefs)

    # Predict future years
    n_future = st.number_input("Number of future years to predict", min_value=1, value=5)
    last_year = years[-1]
    pred_years = np.arange(years[0], last_year + n_future + 1)
    pred_deaths = poly(pred_years)

    # Combine results
    result_df = pd.DataFrame({
        'Year': pred_years,
        'Predicted Deaths': pred_deaths
    })

    st.write("### Prediction Results (Polynomial Fit)")
    st.dataframe(result_df)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(years, deaths, color='black', label='Actual Data')
    ax.plot(pred_years, pred_deaths, color='blue', label=f'Polynomial (deg {degree})')
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Deaths")
    st.pyplot(fig)
else:
    st.info("Please enter at least two rows of data for prediction.")

st.markdown("---\n*The system fits a polynomial to your data and predicts future values accordingly.*")
