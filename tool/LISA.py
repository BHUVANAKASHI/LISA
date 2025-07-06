import numpy as np
import matplotlib.pyplot as plt

class LISASensitivity:
    def __init__(self, observation_time="1yr"):
        self.L = 2.5e9  # arm length in meters
        self.f_star = 19.09e-3  # transfer frequency in Hz
        self.observation_time = observation_time
        self.frequencies = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
        self.params = {
            "6mo": {"alpha": 0.133, "beta": 243, "kappa": 482, "gamma": 917, "f_knee": 0.00258},
            "1yr": {"alpha": 0.171, "beta": 292, "kappa": 1020, "gamma": 1680, "f_knee": 0.00215},
            "2yr": {"alpha": 0.165, "beta": 299, "kappa": 611, "gamma": 1340, "f_knee": 0.00173},
            "4yr": {"alpha": 0.138, "beta": -221, "kappa": 521, "gamma": 1680, "f_knee": 0.00113}
        }
        if observation_time not in self.params:
            raise ValueError("Invalid observation time. Choose from '6mo', '1yr', '2yr', '4yr'.")
        self.sensitivity = self.compute_sensitivity()

    def compute_sensitivity(self):
        f = self.frequencies
        p = self.params[self.observation_time]

        # Instrument noise components
        P_oms = (1.5e-11)**2 * (1 + (2e-3 / f)**4)
        P_acc = (3e-15)**2 * (1 + (0.4e-3 / f)**2) * (1 + (f / 8e-3)**4)
        Pn = (P_oms + 2 * (1 + np.cos(f / self.f_star)**2) * P_acc / (2 * np.pi * f)**4) / self.L**2
        T = 3. / 20. / (1. + 6. / 10. * (f / self.f_star)**2) * 2
        S_n_instrumental = Pn / T

        # Galactic confusion noise
        A_conf = 9e-45
        alpha, beta, kappa, gamma, f_knee = p["alpha"], p["beta"], p["kappa"], p["gamma"], p["f_knee"]
        with np.errstate(over='ignore', invalid='ignore'):
            exp_term = np.exp(np.clip(-f * alpha + beta * f * np.sin(kappa * f), -100, 100))
            tanh_term = np.tanh(gamma * (f_knee - f))
            S_conf = A_conf * f**(-7/3) * (1 + tanh_term) * exp_term
            S_conf[np.isnan(S_conf)] = 0

        # Total noise and strain sensitivity
        S_n_total = S_n_instrumental + S_conf
        return np.sqrt(S_n_total)

    def get_data(self):
        return self.frequencies, self.sensitivity

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.loglog(self.frequencies, self.sensitivity, label='LISA Sensitivity', color='skyblue')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Characteristic Strain [1/sqrt(Hz)]')
        plt.title(f'LISA Sensitivity Curve ({self.observation_time})')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(1e-5, 1)
        plt.ylim(3e-22, 1e-14)
        plt.legend()
        plt.show()