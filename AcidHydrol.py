import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

D = 1e-4 #Axial dispersion coefficient 
v = 0.01 #Superficial velocity 
V = 9.3 #Reactor volume
T = 438 #Temperature
P = 1 #Pressure
b = [0.01778,0.00276,0.00662,0.00288]
a = [1.2453,0.44,0.8135,0.59]
A = np.zeros(4)
CH2SO4 = 6
D = 10
A[0] = 2.007e14
A[1] = 300.8125
A[2] = 0.24924
A[3] = 3.93678
#A = A
#for i in range(len(A)):
    #A[i] = (b[i]*(CH2SO4**a[i])) #Pre-exponential factors
#A = A/60
Ea = [116430,30862,11351,13049] # Activation energies
R = 8.314 
k = [A[i]*np.exp(-Ea[i]/(R*T)) for i in range(4)] #Rate constants
#k = [0.16258/60,0.03006/60,0.00842/60,0.08012/60]

n_in = {"xylan":22.502,"cellulose":29.761,"lignin":18.147,"xylose":0,"glucose":0,"furfural":0,"acetic acid":0,"water":2203.203,"sulphuric acid":25.83}
n0_xylan = n_in["xylan"]
n0_cellulose = n_in["cellulose"]
n0_lignin = n_in["lignin"]
n0_xylose = n_in["xylose"]
n0_glucose = n_in["glucose"]
n0_furfural = n_in["furfural"]
n0_aceticacid = n_in["acetic acid"]
n0_water = n_in["water"]
n0_sulphuricacid = n_in["sulphuric acid"]

Mr_xylan = 25000
Mr_cellulose = 1621600
Mr_lignin = 1000
Mr_xylose = 150.13
Mr_glucose = 180.156
Mr_furfural = 96.0846
Mr_aceticacid = 60.052
Mr_water = 18
Mr_sulphuricacid = 98.079

dens_xylan = 1.52
dens_cellulose = 1.5
dens_lignin = 1.3
dens_xylose = 1.52
dens_glucose = 1.5
dens_furfural = 1.16
dens_aceticacid = 1.05
dens_water = 0.997
dens_sulphuricacid = 1.83

specvol_xylan = 0.0133

v_xylan = (n0_xylan*specvol_xylan)*(10**(-6))
v_cellulose = n0_cellulose*(Mr_cellulose/dens_cellulose)*(10**(-6))
v_lignin = n0_lignin*(Mr_lignin/dens_lignin)*(10**(-6))
v_xylose = n0_xylose*(Mr_xylose/dens_xylose)*(10**(-6))
v_glucose = n0_glucose*(Mr_glucose/dens_glucose)*(10**(-6))
v_furfural = n0_furfural*(Mr_furfural/dens_furfural)*(10**(-6))
v_aceticacid = n0_aceticacid*(Mr_aceticacid/dens_aceticacid)*(10**(-6))
v_water = n0_water*(Mr_water/dens_water)*(10**(-6))
v_sulphuricacid = n0_sulphuricacid*(Mr_sulphuricacid/dens_sulphuricacid)*(10**(-6))

v_tot = v_xylan + v_cellulose + v_lignin + v_xylose + v_glucose + v_furfural + v_aceticacid + v_water + v_sulphuricacid
#v_tot = 90
#Specific Heat Capacities for deltaH1
Cp1_Hemicellulose = 1209*(298-T) #J/kg
Cp1_Cellulose = 1305*(298-T) #J/kg
Cp1_Lignin = 1301*(298-T) #J/kg
Cp_H20_1 = 1329.062959 - 4.217*T + ((8.95e-4)*(T**2)) - ((8.693e-6)*(T**3)) + ((9.9125e-9)*(T**4))
Cp_H2SO4_1 = 17660.61825 - (0.1567*T) - ((2.5933e-4)*(T**2)) + ((4.6978e-5)*(T**3)) - ((2.38825e-6)*(T**4))

#Enthalpy change from T to 298K
DeltaH1 = n0_xylan*Cp1_Hemicellulose + n0_cellulose*Cp1_Cellulose + n0_lignin*Cp1_Lignin + n0_water*Cp_H20_1 + n0_sulphuricacid*Cp_H2SO4_1

#Heats of Reaction j/mol
Heat_r1 = -9470
Heat_r2 = 425730
Heat_r3 = -1152790
Heat_r4 = -163770

def reaction_rates(n,k):
    n_xylan, n_cellulose,n_lingin,n_xylose,n_glucose, n_furfural, n_aceticacid, n_water,n_sulphuricacid = n
    C_xylan = n_xylan/v_tot
    C_water = n_water/v_tot
    C_cellulose = n_cellulose/v_tot
    C_xylose = n_xylose/v_tot
    r1 = k[0]*C_xylan #Xylan + H20 --> Xylose 
    r2 = k[1]*C_cellulose #Cellulose + H20 --> Glucose
    r3 = k[2]*C_xylose #Xylose --> Furfural + 3H20
    r4 = k[3]*C_xylan #Xylan + H20 --> 2.5Acetic Acid
    return np.array([-r1-r2,-r2,0,r1-r3,r2,r3,2.5*r4,-r1-r2+3*r3-r4,0])

def pfr_model(V,n):
    dndV = np.zeros_like(n)
    rates = reaction_rates(n,k)
    #dndz = ((3*np.pi*(z**2))/400)*rates
    dndV = rates
    return dndV

V_span = (0.1,V)
z_eval = np.linspace(0.1,V,1000)
n0 = [n_in["xylan"],n_in["cellulose"],n_in["lignin"],n_in["xylose"],n_in["glucose"],n_in["furfural"],n_in["acetic acid"],n_in["water"],n_in["sulphuric acid"]]
solution = solve_ivp(pfr_model,V_span,n0,t_eval=z_eval,method="RK45")

print(solution.y.shape)

Cp3_Hemicellulose = 1209*(T-298) #J/kg
Cp3_Cellulose = 1305*(T-298) #J/kg
Cp3_Lignin = 1301*(T-298)
Cp_H20_3 = 1329.062959 + 4.217*T - ((8.95e-4)*(T**2)) + ((8.693e-6)*(T**3)) - ((9.9125e-9)*(T**4))
Cp_H2SO4_3 = 17660.61825 + (0.1567*T) + ((2.5933e-4)*(T**2)) - ((4.6978e-5)*(T**3)) + ((2.38825e-6)*(T**4))
Cp_xylose_3 = 178.1*(T-298)
Cp_glucose_3 = (113.6*T) +(0.206*(T**2)) - ((5e-5)*(T**3)) - 50823.2444
Cp_furfural_3 = (1.047619*(T**2)) - (164.738*T) + 43940.696
Cp_acetic_acid_3 = (120*T) + (0.141*(T**2)) - ((5e-5)*(T**3)) - 46958.1844
DeltaH3 = np.zeros(len(z_eval))
for i in range(len(z_eval)):
    DeltaH3[i] = solution.y[0,i]*Cp3_Hemicellulose + solution.y[1,i]*Cp3_Cellulose + solution.y[2,i]*Cp3_Lignin + solution.y[3,i]*Cp_xylose_3 + solution.y[4,i]*Cp_glucose_3 + solution.y[5,i]*Cp_furfural_3 + solution.y[6,i]*Cp_acetic_acid_3 + solution.y[7,i]*Cp_H20_3 + solution.y[8,i]*Cp_H2SO4_3

outlet = solution.y[:,-1]
print(outlet)

def reaction_rates_heat(n,k):
    n_xylan, n_cellulose,n_lingin,n_xylose,n_glucose, n_furfural, n_aceticacid, n_water,n_sulphuricacid = n
    C_xylan = n_xylan/v_tot
    C_water = n_water/v_tot
    C_cellulose = n_cellulose/v_tot
    C_xylose = n_xylose/v_tot
    r1 = k[0]*C_xylan #Xylan + H20 --> Xylose 
    r2 = k[1]*C_cellulose #Cellulose + H20 --> Glucose
    r3 = k[2]*C_xylose #Xylose --> Furfural + 3H20
    r4 = k[3]*C_xylan #Xylan + H20 --> 2.5Acetic Acid
    return np.array([r1,r2,r3,r4])

DeltaH2 = np.zeros(len(z_eval))

for i in range(len(z_eval)):
    reactions = reaction_rates_heat(solution.y[:,i],k)
    DeltaH2[i] = Heat_r1*reactions[0] + Heat_r2*reactions[1] + Heat_r3*reactions[2] + Heat_r3*reactions[3]

DeltaH = (DeltaH1 + DeltaH2*V + DeltaH3)*(10**(-6))
print("Heat Duty:", DeltaH[-1])

plt.plot(solution.t,DeltaH)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Heat Duty (MW)")
plt.show()

xylan_consumed = n_in["xylan"]-solution.y[0]
xylose_prod = solution.y[3]
selectivity_xylose = xylose_prod/xylan_consumed
conversion_xylan = xylan_consumed/n_in["xylan"]


plt.plot(conversion_xylan,selectivity_xylose)
plt.xlabel("conversion_xylan")
plt.ylabel("Selectivity to Xylose")
plt.title("Selectivity Profile in PFR")
plt.show()

plt.plot(solution.t,conversion_xylan)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Conversion of Xylan")
plt.title("Conversion Progression")
plt.show()

plt.plot(solution.t,selectivity_xylose)
plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Selectivity to Xylose")
plt.title("Selectivity Progression")
plt.show()


for i,species in enumerate(["Xylan","Cellulose","Lignin","Xylose","Glucose","Furfural","Acetic Acid"]):
    plt.plot(solution.t,solution.y[i],label=species)

plt.xlabel("Reactor Volume (m3)")
plt.ylabel("Molar Flowrates (mol/s)")
plt.legend()
plt.title("Flowrate Projections in PFR")
plt.show()

for T in [408,418,428,438,448,458]:
    k = [A[i]*np.exp(-Ea[i]/(R*T)) for i in range(4)]
    solution = solve_ivp(pfr_model,V_span,n0,t_eval=z_eval,method="RK45")
    plt.plot(solution.t,solution.y[3], label=f"T = {T} K")

plt.xlabel("Reactor Length (m)")
plt.ylabel("Molar Flowrates (mol/s)")
plt.legend()
plt.title("Flowrate Projections in PFR")
plt.show()

