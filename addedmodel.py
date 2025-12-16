import numpy as np
from scipy.optimize import minimize
#De-neutrosophication part
def deneutrosophicate_gtnn(u, v, w, Psi, Pi, Phi):
    

    u1, u2, u3 = u
    v1, v2, v3 = v
    w1, w2, w3 = w

    truth_part = (u1 + u3) / 2 + Psi * (2*u2 - u1 - u3) / 2
    indet_part = (v1 + v3) / 2 + (2*v2 - v1 - v3) / (2*Pi)
    falsity_part = (w1 + w3) / 2 + (2*w2 - w1 - w3) / (2*Phi)

    return (truth_part + indet_part + falsity_part) / 3

A0 = deneutrosophicate_gtnn((350,500,650),(400,550,700),(500,550,600),0.5,0.7,0.8)
A1 = deneutrosophicate_gtnn((5.5,6,6.5),(5.5,6.5,7.5),(6,7,8),0.2,0.4,0.6)
A2 = deneutrosophicate_gtnn((4,5,6),(5,6,7),(6,7,8),0.7,0.4,0.3)
A3 = deneutrosophicate_gtnn((25,28,31),(28,30,32),(30,32,34),0.6,0.8,1.0)
A4 = deneutrosophicate_gtnn((8,10,12),(10,12,14),(12,14,16),0.7,0.3,0.3)
A5 = deneutrosophicate_gtnn((6,7,8),(7,8,9),(8,9,10),0.7,0.2,0.2)
A6 = deneutrosophicate_gtnn((1,2,3),(2,3,4),(3,4,5),0.7,0.2,0.2)
#only above part was added extra

m = 20
b = 0.5
c = 4
c_prime = 4.5
delta = 0.3
p = 60
xi = 5
theta0 = 0.1
a1 = 0.5
qs = 0.9
A = 30
nu = 0.4
g = 10
alpha = 0.2
gamma = 0.5
T = 1.0

qr = 1 - np.exp(-a1 * qs)

# Assumption 3)
theta = theta0 * np.exp(-a1 * xi)



def exponential_term(t1):
   
    return np.exp((theta + b * A**nu) * t1)

def total_profit(t1):
    
    if t1 <= 0 or t1 >= T:
        return 1e6

    term1 = (m*p - delta + c*qr) * A**nu
    term2 = (m*p - delta + c_prime*qr) * A**nu

    exp_val = exponential_term(t1)

    
    holding_deterioration = (A1 + A2*theta) * term1 / (theta + b*A**nu)**2 * \
        (exp_val - 1 - (theta + b*A**nu)*t1)

    
    shortage_cost = A4 * term2 * (T - t1)**2 / 2

    rework_screening_cost = (A5*alpha + A6) * term1 / ((theta + b*A**nu)*(1 - alpha)) * \
        (exp_val - 1)

    
    purchase_cost = A3 * (
        term1/(theta + b*A**nu)*(exp_val - 1)
        + term2*(T - t1)
        + gamma*alpha*term1/((theta + b*A**nu)*(1 - alpha))*(exp_val - 1)
    )

   
    sales_revenue = p * (
        (c - c_prime)*qr*A**nu*t1
        + b*term1*A**nu/(theta + b*A**nu)**2 *
        (exp_val - 1 - (theta + b*A**nu)*t1)
        + term2*T
    )

    profit = (
        sales_revenue
        - A0
        - holding_deterioration
        - shortage_cost
        - rework_screening_cost
        - purchase_cost
        - g*A
        - xi*T
    )

    return -profit / T

result = minimize(
    total_profit,
    x0=[0.8],
    bounds=[(0.01, T - 0.01)]
)

t1_star = result.x[0]
profit_star = -result.fun
print("Optimal replenishment time t1* =", round(t1_star, 4))
print("Optimal total profit =", round(profit_star, 2))
