import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

D = 1e-4 #Axial dispersion coefficient 
v = 0.01 #Superficial velocity 
V = 20 #Reactor length
T = 323 #Temperature
P = 1 #Pressure
k = 0.14 #Rate constant
rho_cat = 3360
void = 0.4
D = 1
alpha = 0.3 # Recycle Ratio 

n_in_fresh = {"Precursor":6.771,"SAF-n":0,"SO3":6.77,"SO2":0}
n_Prec = n_in_fresh["Precursor"]
n_SAF = n_in_fresh["SAF-n"]
n_SO3 = n_in_fresh["SO3"]
n_SO2 = n_in_fresh["SO2"]


specvol_Prec = 0.2855e-9
specvol_Surf = 0.3181e-9


v_Prec = n_Prec*specvol_Prec
v_SAF = n_SAF*specvol_Surf
v_SO3 = n_SO3*0.0224    
v_SO2 = n_SO2*0.0224
v_tot = v_Prec + v_SAF + v_SO3 + v_SO2

Cp_Prec_1 = 641.7*(298-T)
Cp_Prec_2 = -Cp_Prec_1
Cp_SO3_2 = (24.02503*T) + (0.05973*(T**2)) - ((3.146e-5)*(T**3)) + ((6.741e-9)*(T**4)) + (117517/T) - 12078.69014
Cp_SO3_1 = -Cp_SO3_2
Cp_Surf_1 = 1500.62*(298-T)
Cp_Surf_2 = -Cp_Surf_1

n0_SO3 = n_in_fresh["SO3"]
n0_Precursor = n_in_fresh["Precursor"]

DeltaH1 = n0_SO3*Cp_SO3_1 + n0_Precursor*Cp_Prec_1

def reaction_rates(n):
    n_Prec, n_Surf, n_SO3, n_SO2 = n
    C_SO3 = n_SO3/v_tot
    r = k*C_SO3
    return np.array([-r,r,-r,0])

def pfr_model(z,n):
    dndz = np.zeros_like(n)
    rates = reaction_rates(n)
    dndz = ((np.pi*(z**2))/400)*rates
    return dndz

def reaction_rate_heat(n):
    n_Prec, n_Surf, n_SO3, n_SO2 = n
    C_SO3 = n_SO3/v_tot
    r = k*C_SO3
    return np.array(r)

tol = 1e-4 
max_iter = 50
error = 1
iteration = 0

n_out = ([n_in_fresh[species] for species in ["Precursor","SAF-n","SO3","SO2"]])
n_out = np.array(n_out, dtype =float)

while error>tol and iteration<max_iter:
    iteration += 1
    n_in_total = np.array([n_in_fresh[species] for species in ["Precursor", "SAF-n","SO3","SO2"]]) + alpha*n_out


    z_span = (0.1,V)
    z_eval = np.linspace(0.1,V,1000)
    #n0 = [n_in["Precursor"],n_in["SAF-n"],n_in["SO3"],n_in["SO2"]]

    solution = solve_ivp(pfr_model,z_span,n_in_total,t_eval=z_eval,method="RK45")
    new_n_out = solution.y[:,-1]
    
    error = np.linalg.norm(new_n_out-n_out) / np.linalg.norm(new_n_out)
    n_out = new_n_out

    print(f"Iteration {iteration}: Error = {error}")

n_outlet = (1-alpha)*solution.y

DeltaH2 = np.zeros(len(z_eval))

for i in range(len(z_eval)):
    reaction = reaction_rate_heat(solution.y[:,i])
    DeltaH2[i] = reaction*160000

DeltaH3 = np.zeros(len(z_eval))

for i in range(len(z_eval)):
    DeltaH3[i] = (solution.y[0,i]*Cp_Prec_2) + (solution.y[1,i]*Cp_Prec_2) + (solution.y[3,i]*Cp_SO3_2)

DeltaH = DeltaH1 + DeltaH2*V + DeltaH3 
DeltaH = DeltaH* (10**(-3))

plt.rcParams['font.family'] = 'Times New Roman'  # Change to desired font
plt.rcParams['font.size'] = 11
plt.plot(solution.t,DeltaH)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Heat Duty (kW)")
plt.title("Reactor Duty in PFR")
plt.show()

print("The Heat Duty is:", DeltaH[-1])
total_conversion = (n_in_fresh["Precursor"]-n_outlet[0,:])/n_in_fresh["Precursor"]
single_pass_conversion = (n_in_total[0] - solution.y[0,:])/n_in_total[0]
plt.plot(single_pass_conversion,total_conversion)
plt.xlabel("Single Pass Conversion")
plt.ylabel("Total Conversion")
plt.title("Conversion Progression in PFR")
plt.show()

plt.plot(solution.t,single_pass_conversion)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Single Pass Conversion")
plt.title("Conversion Progression")
plt.show()

for i,species in enumerate(["Precursor","SAF-n","SO3","SO2"]):
    plt.plot(solution.t,n_outlet[i],label=species)

plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Molar Flowrates (mol/s)")
plt.legend()
plt.title("Flowrate Projections in PFR")
plt.show()

print(n_outlet[:,-1])

for recycle_ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    alpha = recycle_ratio
    while error>tol and iteration<max_iter:
        iteration += 1
        n_in_total = np.array([n_in_fresh[species] for species in ["Precursor", "SAF-n","SO3","SO2"]]) + alpha*n_out


        z_span = (0.1,V)
        z_eval = np.linspace(0.1,V,1000)
        #n0 = [n_in["Precursor"],n_in["SAF-n"],n_in["SO3"],n_in["SO2"]]

        solution = solve_ivp(pfr_model,z_span,n_in_total,t_eval=z_eval,method="RK45")
        new_n_out = solution.y[:,-1]
    
        error = np.linalg.norm(new_n_out-n_out) / np.linalg.norm(new_n_out)
        n_out = new_n_out
    n_outlet = (1-alpha) * solution.y
    total_conversion = (n_in_fresh["Precursor"] - n_outlet[0,:])/n_in_fresh["Precursor"]
    single_pass_conversion = (n_in_total[0] - solution.y[0,:])/n_in_total[0]
    plt.plot(single_pass_conversion,total_conversion,label= f"Recycle Ratio = {recycle_ratio}")
plt.xlabel("Single Pass Conversion")
plt.ylabel("Total Conversion")
plt.title("Conversion Progression")
plt.legend()
plt.xlim(0.8,1.0)
plt.ylim(0.8,1.0)
plt.show()
