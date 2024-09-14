import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ฟังก์ชันสำหรับคำนวณ Likelihood
def likelihood(x, mean, std_dev):
    return (1.0 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

# ฟังก์ชันสำหรับคำนวณ Posterior โดยสามารถปรับค่า Prior ได้
def posterior(x, mean0, mean1, std_dev, prior0, prior1):
    likelihood0 = likelihood(x, mean0, std_dev)
    likelihood1 = likelihood(x, mean1, std_dev)
    
    posterior0 = (likelihood0 * prior0) / (likelihood0 * prior0 + likelihood1 * prior1)
    posterior1 = (likelihood1 * prior1) / (likelihood0 * prior0 + likelihood1 * prior1)
    
    return posterior0, posterior1

# ฟังก์ชันสำหรับคำนวณขอบเขตการตัดสินใจ (Decision Boundary) ที่คำนึงถึง Prior
def decision_boundary(mean0, mean1, std_dev, prior0, prior1):
    return (mean0 + mean1) / 2 - (std_dev**2 / (mean1 - mean0)) * np.log(prior1 / prior0)

# สร้างกราฟสำหรับพารามิเตอร์ที่กำหนดเอง
def fixed_parameters():
    mean0, mean1 = 2, 5  # ค่าเฉลี่ยของ Class 0 และ Class 1
    std_dev = 1.5  # ความแปรปรวนของทั้งสองคลาสเท่ากัน
    prior0, prior1 = 0.9, 0.1  # ค่า Prior สำหรับ Class 0 และ Class 1

    # กำหนดช่วงของ x สำหรับการพล็อต
    x_values = np.linspace(-2, 10, 200)
    
    # คำนวณ likelihood และ posterior
    likelihood0_values = likelihood(x_values, mean0, std_dev)
    likelihood1_values = likelihood(x_values, mean1, std_dev)
    posterior_values0, posterior_values1 = posterior(x_values, mean0, mean1, std_dev, prior0, prior1)

    # กำหนดจุดตัดสินใจ
    decision_point = decision_boundary(mean0, mean1, std_dev, prior0, prior1)

    # พล็อตกราฟ
    plt.figure(figsize=(15, 5))
    
    # Likelihood plot
    plt.subplot(1, 3, 1)
    plt.plot(x_values, likelihood0_values, label='Likelihood Class 0', color='blue')
    plt.plot(x_values, likelihood1_values, label='Likelihood Class 1', color='red')
    plt.title('Likelihood')
    plt.legend()

    # Posterior plot
    plt.subplot(1, 3, 2)
    plt.plot(x_values, posterior_values0, label='Posterior Class 0', color='green')
    plt.plot(x_values, posterior_values1, label='Posterior Class 1', color='orange')
    plt.axvline(x=decision_point, color='black', linestyle='--', label='Decision Boundary')
    plt.title('Posterior with Decision Boundary')
    plt.legend()

    # Decision Boundary plot
    plt.subplot(1, 3, 3)
    plt.plot(x_values, likelihood0_values, label='Likelihood Class 0', color='blue')
    plt.plot(x_values, likelihood1_values, label='Likelihood Class 1', color='red')
    plt.axvline(x=decision_point, color='black', linestyle='--', label='Decision Boundary')
    plt.title('Decision Boundary on Likelihood')
    plt.legend()

    plt.show()

fixed_parameters()
