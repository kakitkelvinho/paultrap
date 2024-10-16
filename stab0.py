import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e
from scipy.integrate import odeint


def main():
    # things to be defined
    mass = 1.8e-18 # mass of particle, in kg
    radius = 150e-9 # radius of particle, in mm
    r0 = 0.25e-3 # center of trap to electrode
    z0 = 2e-3 # center of trap to endcap
    Z = 100 # charge of particle, integer
    trapfreq = 18e3 # Trap drive freq, in Hz
    endcapvolt = 50 # end cap voltage
    rfvolt = 200 # rf voltage
    

    # time
    n = 1000 # resolution
    xi = np.linspace(0, np.pi, n)
    ax, qx = aq_calc(mass, r0, z0, endcapvolt, rfvolt, trapfreq, Z, axis=1)

    # initial condition
    y0 = [0,-0.2]

    sol = odeint(particle_ode, y0, xi, args=(ax, qx))
    print(sol.shape)

    # plot sol
    plt.figure()
    plt.plot(xi, sol[:,0])
    plt.plot(xi, sol[:,1])
    plt.show()

    plt.close()


# Mathieu's equation
# u'(xi) = uu
# uu'(xi) = - f(xi) * u

def particle_ode(y, xi, a, q):
    u, uu = y
    dydt = [uu, - (a - 2*q*np.cos(2*xi))*u ]
    return dydt

def aq_calc(mass, r0, z0, endcapvolt, rfvolt, trapfreq, Z, axis=1):
    '''Calculates the a and q constants given:
    mass: particle mass
    r0: center of trap to rf electrode
    z0: center of trap to endcap
    endcapvolt: End Cap voltages
    rfvolt: rf amplitude voltage
    trapfreq: trap frequency
    Z: number of charge
    axis: x(1), y(2), z(3)'''
    prefac = [1,1,-2]
    a = prefac[axis] * (8*e*Z*endcapvolt) / (mass * (trapfreq**2) * (r0**2 + 2*z0**2))
    q = prefac[axis] * (-4*e*Z*rfvolt) / (mass * (trapfreq**2)* (r0**2 + 2 * z0**2))
    return a, q 

if __name__ == '__main__':
    main()
