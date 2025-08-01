import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- ODE System (Monkeypox Model, Picard Iterative Method) ---

def monkeypox_picard(params, initials, h=0.1, T=20, max_iter=5, tol=1e-6):
    # Unpack parameters
    alpha1, alpha2, alpha3 = params['alpha1'], params['alpha2'], params['alpha3']
    beta1, beta2, beta3 = params['beta1'], params['beta2'], params['beta3']
    Lambdah, Lambdar = params['Lambdah'], params['Lambdar']
    phi, rho = params['phi'], params['rho']
    muh, mur = params['muh'], params['mur']
    deltah, deltar = params['deltah'], params['deltar']
    tau = params['tau']

    # Initial values
    SH, EH, UH, QH, RH, SR, ER, VR = (
        initials['SH'], initials['EH'], initials['UH'], initials['QH'],
        initials['RH'], initials['SR'], initials['ER'], initials['VR']
    )

    NH = SH + EH + UH + QH + RH
    NR = SR + ER + VR

    t_points = np.arange(0, T + h, h)
    results = []
    for t in t_points:
        # Store previous values for Picard iterations
        SH_n, EH_n, UH_n, QH_n, RH_n, SR_n, ER_n, VR_n = SH, EH, UH, QH, RH, SR, ER, VR
        for _ in range(max_iter):
            NH = SH_n + EH_n + UH_n + QH_n + RH_n
            NR = SR_n + ER_n + VR_n

            dSH = Lambdah - (beta1 * VR_n + beta2 * UH_n) * SH_n / NH - muh * SH_n + phi * QH_n
            dEH = (beta1 * VR_n + beta2 * UH_n) * SH_n / NH - (alpha1 + alpha2 + muh) * EH_n
            dUH = alpha1 * EH_n - (muh + deltah + rho) * UH_n
            dQH = alpha2 * EH_n - (phi + tau + muh + deltah) * QH_n
            dRH = rho * UH_n + tau * QH_n - muh * RH_n
            dSR = Lambdar - beta3 * SR_n * VR_n / NR - mur * SR_n
            dER = beta3 * SR_n * VR_n / NR - (mur + alpha3) * ER_n
            dVR = alpha3 * ER_n - (mur + deltar) * VR_n

            # Euler's method for the integral part
            SH_new = SH + h * dSH
            EH_new = EH + h * dEH
            UH_new = UH + h * dUH
            QH_new = QH + h * dQH
            RH_new = RH + h * dRH
            SR_new = SR + h * dSR
            ER_new = ER + h * dER
            VR_new = VR + h * dVR

            # Check for convergence
            if (all([
                abs(SH_new - SH_n) < tol, abs(EH_new - EH_n) < tol, abs(UH_new - UH_n) < tol,
                abs(QH_new - QH_n) < tol, abs(RH_new - RH_n) < tol, abs(SR_new - SR_n) < tol,
                abs(ER_new - ER_n) < tol, abs(VR_new - VR_n) < tol,
            ])):
                break

            SH_n, EH_n, UH_n, QH_n, RH_n, SR_n, ER_n, VR_n = (
                SH_new, EH_new, UH_new, QH_new, RH_new, SR_new, ER_new, VR_new
            )

        # Update for next step
        SH, EH, UH, QH, RH, SR, ER, VR = SH_n, EH_n, UH_n, QH_n, RH_n, SR_n, ER_n, VR_n
        results.append([t, SH, EH, UH, QH, RH, SR, ER, VR])

    df = pd.DataFrame(results, columns=['Time', 'SH', 'EH', 'UH', 'QH', 'RH', 'SR', 'ER', 'VR'])
    return df

# --- Streamlit UI ---

st.set_page_config(page_title="Monkeypox Model - Picard Iterative Method", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3e/Monkeypox_Virus_Particle.png", width=100)
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Home", "Picard Method Solver", "Comparison (Coming Soon)", "About"])

if tab == "Home":
    st.title("Welcome to the Monkeypox Model Solver")
    st.write("This application demonstrates the Modified Picard Iterative Method for solving compartment models like Monkeypox dynamics. Use the menu to get started.")
    st.image("https://www.cdc.gov/poxvirus/monkeypox/images/monkeypox-virus-768px.jpg", use_column_width=True)

elif tab == "Picard Method Solver":
    st.title("Picard Iterative Method - Monkeypox Model")
    st.write("Enter parameters and initial values for each compartment.")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Initial Values")
        SH = st.number_input("SH (Susceptible Humans)", value=10000.0)
        EH = st.number_input("EH (Exposed Humans)", value=50.0)
        UH = st.number_input("UH (Infected Humans)", value=0.57)
        QH = st.number_input("QH (Isolation Humans)", value=0.43)
        RH = st.number_input("RH (Recovered Humans)", value=0.0)
        SR = st.number_input("SR (Susceptible Rodents)", value=1000.0)
        ER = st.number_input("ER (Exposed Rodents)", value=100.0)
        VR = st.number_input("VR (Infected Rodents)", value=10.0)
        initials = dict(SH=SH, EH=EH, UH=UH, QH=QH, RH=RH, SR=SR, ER=ER, VR=VR)

    with col2:
        st.header("Parameters")
        alpha1 = st.number_input("alpha1 (Human-to-Infectious Rate)", value=0.423890)
        alpha2 = st.number_input("alpha2 (Suspected Case Identification Rate)", value=1.797575)
        alpha3 = st.number_input("alpha3 (Rodent-to-Infectious Rate)", value=0.025289)
        beta1 = st.number_input("beta1 (Rodent-Human Contact Rate)", value=0.011503)
        beta2 = st.number_input("beta2 (Human-Human Contact Rate)", value=0.747322)
        beta3 = st.number_input("beta3 (Rodent-Rodent Contact Rate)", value=0.321102)
        Lambdah = st.number_input("Lambdah (Human Recruitment Rate)", value=0.34857)
        Lambdar = st.number_input("Lambdar (Rodent Recruitment Rate)", value=0.60822)
        phi = st.number_input("phi (Undetected Proportion)", value=1.575454)
        rho = st.number_input("rho (Recovery Rate)", value=0.067689)
        muh = st.number_input("muh (Human Death Rate)", value=1/(79*365))
        mur = st.number_input("mur (Rodent Death Rate)", value=1/(5*365))
        deltah = st.number_input("deltah (Disease-induced Death Rate, Human)", value=0.015016)
        deltar = st.number_input("deltar (Disease-induced Death Rate, Rodent)", value=0.000025)
        tau = st.number_input("tau (Isolation-to-Recovered Rate)", value=0.999763)
        params = dict(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                      beta1=beta1, beta2=beta2, beta3=beta3,
                      Lambdah=Lambdah, Lambdar=Lambdar, phi=phi, rho=rho,
                      muh=muh, mur=mur, deltah=deltah, deltar=deltar, tau=tau)

    st.header("Simulation Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        h = st.number_input("Step size (h)", value=0.1, min_value=0.01)
    with col2:
        T = st.number_input("Total time (T)", value=20.0, min_value=1.0)
    with col3:
        max_iter = st.number_input("Picard max iterations", value=5, min_value=1, step=1)

    if st.button("Run Picard Method"):
        df = monkeypox_picard(params, initials, h=h, T=T, max_iter=int(max_iter))
        st.subheader("Results Table")
        st.dataframe(df)

        st.subheader("Compartment Trends")
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in df.columns[1:]:
            ax.plot(df['Time'], df[col], label=col)
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_title("Monkeypox Model Compartments Over Time")
        ax.legend()
        st.pyplot(fig)

elif tab == "Comparison (Coming Soon)":
    st.title("Comparison (Coming Soon)")
    st.info("This section will allow you to compare Picard with RK4 and other methods in future updates.")

elif tab == "About":
    st.title("About")
    st.write("Developed by [Your Name].")
    st.write("This app numerically solves the Monkeypox model ODE system using the Modified Picard Iterative Method.")
    st.write("Contact: your.email@example.com")

