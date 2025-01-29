import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

D = 1e-4 #Axial dispersion coefficient 
v = 0.01 #Superficial velocity 
V = 2.43 #Reactor volume
T = 720 #Temperature
P = 1 #Pressure
k1 = np.exp(12.16-(5473/T))
k2 = np.exp(-9.953+(8619/T))
k3 = np.exp(-71.745+(52596/T))
kP = np.exp(-10.68+(11300/T))
rho_cat = 3360
void = 0.4  
D = 1
n_in = {"SO2":35.278,"SO3":1e-4,"oxygen":35.278,"nitrogen":132.712}

def reaction_rates(n):
    n_SO2, n_SO3, n_oxygen, n_nitrogen = n
    n_total = n_SO2+n_SO3+n_oxygen+n_nitrogen
    p_O2 = (n_oxygen/n_total)*P
    p_SO2 = (n_SO2/n_total)*P
    p_SO3 = (n_SO3/n_total)*P
    r1 = ((k1*p_O2*p_SO2)*(1-(p_SO3/(kP*p_SO2*(p_O2**0.5)))))/(22.414*((1+k2*p_SO2+k3*p_SO3)**2))
    #r1 = k1*((p_SO2/p_SO3)**0.5)*(p_O2-(((p_SO3/(kP*p_SO2)))**2))
    return np.array([-r1,r1,-0.5*r1,0])

def pfr_model(z,n):
    dndV = np.zeros_like(n)
    rates = reaction_rates(n)
    dndV = rates*rho_cat*(1-void)
    return dndV

V_span = (0.1,V)
V_eval = np.linspace(0.1,V,1000)
n0 = [n_in["SO2"],n_in["SO3"],n_in["oxygen"],n_in["nitrogen"]]
Cp_SO2_1 = 9210.395 - (21.43049*T) - (0.037175*(T**2)) - ((1.925e-5)*(T**3)) - ((4.0888835e-9)*(T**4)) + (86731/T)
Cp_O2_1 = 11362.45735 -(31.32234*T) + (0.01012*(T**2)) - ((1.929e-5)*(T**3)) + ((9.12656e-9)*(T**4)) - (741599/T)
Cp_SO2_3 = -Cp_SO2_1
Cp_O2_3 = -Cp_SO2_1
Cp_SO3_3 = (24.02503*T) + (0.05973*(T**2)) - ((3.146e-5)*(T**3)) + ((6.741e-9)*(T**4)) + (117517/T) - 12078.69014
DeltaH1 = n0[0]*Cp_SO2_1 + n0[2]*Cp_O2_1

solution = solve_ivp(pfr_model,V_span,n0,t_eval=V_eval,method="RK45")

def reaction_rates_heat(n):
    n_SO2, n_SO3, n_oxygen, n_nitrogen = n
    n_total = n_SO2+n_SO3+n_oxygen+n_nitrogen
    p_O2 = (n_oxygen/n_total)*P
    p_SO2 = (n_SO2/n_total)*P
    p_SO3 = (n_SO3/n_total)*P
    r1 = ((k1*p_O2*p_SO2)*(1-(p_SO3/(kP*p_SO2*(p_O2**0.5)))))/(22.414*((1+k2*p_SO2+k3*p_SO3)**2))
    #r1 = k1*((p_SO2/p_SO3)**0.5)*(p_O2-(((p_SO3/(kP*p_SO2)))**2))
    return r1

DeltaH2 = np.zeros(len(V_eval))

for i in range(len(V_eval)):
    reaction = reaction_rates_heat(solution.y[:,i])
    DeltaH2[i] = -96232*reaction

DeltaH3 = np.zeros(len(V_eval))

for i in range(len(V_eval)):
    DeltaH3[i] = (solution.y[0,i]*Cp_SO2_3) + (solution.y[1,i]*Cp_SO3_3) + (solution.y[2,i]*Cp_O2_3)

DeltaH = DeltaH1 + DeltaH2*V + DeltaH3

plt.plot(solution.t,DeltaH)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Heat Duty (W)")
plt.title("Reactor Duty Progression")
plt.show()

print("The heat duty is: ",DeltaH[-1])
    
conversion = (n_in["SO2"] - solution.y[0])/n_in["SO2"]
plt.plot(solution.t,conversion)

plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Conversion of SO2")
plt.title("Conversion Progression across PFR")
plt.show()

n_SO3 = solution.y[1]
n_SO2 = solution.y[0]
n_O2 = solution.y[2]
n_N2 = solution.y[3]
ntot = n_SO3 + n_SO2 + n_O2 + n_N2

outlet = solution.y[:,-1]
print(outlet)

for i,species in enumerate(["SO2","SO3","oxygen"]):
    plt.plot(solution.t,solution.y[i],label=species)

plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Molar Flowrate (mol/s)")
plt.legend()
plt.title("Flowrate Profiles in PFR")
plt.show()


for T in [650,660,670,680,690,700,710,720,730,740,750]:
    k1 = np.exp(12.16-(5473/T))
    k2 = np.exp(-9.953+(8619/T))
    k3 = np.exp(-71.745+(52596/T))
    kP = np.exp(-10.68+(11300/T))
    solution = solve_ivp(pfr_model,V_span,n0,t_eval=V_eval,method="RK45")
    plt.plot(solution.t,solution.y[1], label=f"T = {T} K")


plt.xlabel("Reactor Volume (m)")
plt.ylabel("Molar Flowrate (mol/s)")
plt.legend()
plt.title("Flowrate Profiles in PFR")
plt.show()

