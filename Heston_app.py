import streamlit as st
import numpy as np
import scipy.integrate as spi

class Heston_Model:
    def __init__(self, K, t, S0, r, v0, theta, kappa, sigma, rho):
        self.K = K
        self.t = t
        self.S0 = S0
        self.r = r
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho

    def characteristic_function(self, phi, type):
        if type == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa

        a = self.kappa * self.theta
        x = np.log(self.S0)
        d = np.sqrt(
            (self.rho * self.sigma * phi * 1j - b) ** 2
            - self.sigma**2 * (2 * u * phi * 1j - phi**2)
        )
        g = (b - self.rho * self.sigma * phi * 1j + d) / (
            b - self.rho * self.sigma * phi * 1j - d
        )
        D = self.r * phi * 1j * self.t + (a / self.sigma**2) * (
            (b - self.rho * self.sigma * phi * 1j + d) * self.t
            - 2 * np.log((1 - g * np.exp(d * self.t)) / (1 - g))
        )
        E = (
            ((b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2)
            * (1 - np.exp(d * self.t))
            / (1 - g * np.exp(d * self.t))
        )

        return np.exp(D + E * self.v0 + 1j * phi * x)

    def integral_function(self, phi, type):
        integral = np.exp(-1 * 1j * phi * np.log(self.K)) * self.characteristic_function(phi, type=type)
        return integral

    def P_Value(self, type):
        ifun = lambda phi: np.real(self.integral_function(phi, type=type) / (1j * phi))
        intervals = [(0, 10), (10, 100), (100, 1000)]
        result = 0
        for interval in intervals:
            res, err = spi.quad(ifun, interval[0], interval[1], limit=100)
            result += res
        return 0.5 + (1 / np.pi) * result

    def Call_Value(self):
        P1 = self.S0 * self.P_Value(type=1)
        P2 = self.K * np.exp(-self.r * self.t) * self.P_Value(type=2)
        if np.isnan(P1 - P2):
            return 1000000
        else:
            return P1 - P2

# Streamlit UI
st.title("Heston Model Option Pricing App")

st.markdown("Enter the parameters below:")

K = st.number_input("Strike Price (K)", value=100.0)
t = st.number_input("Time to Maturity in Years (t)", value=1.0)
S0 = st.number_input("Initial Stock Price (S0)", value=100.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
v0 = st.number_input("Initial Variance (v0)", value=0.04)
theta = st.number_input("Long-Term Variance (theta)", value=0.04)
kappa = st.number_input("Mean Reversion Speed (kappa)", value=1.5)
sigma = st.number_input("Volatility of Variance (sigma)", value=0.3)
rho = st.number_input("Correlation (rho)", value=-0.7)

if st.button("Calculate Call Option Price"):
    model = Heston_Model(K, t, S0, r, v0, theta, kappa, sigma, rho)
    result = model.Call_Value()
    st.success(f"The Call Option Price is: {result:.4f}")
