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
    
    ax = np.linspace(-0.8, 0.2, 100)
    qx = np.linspace(0.0, 1.5, 100)


    stabilities = np.zeros((len(ax), len(qx)))
    for i in range(len(ax)):
        for j in range(len(qx)):
            stabilities[i,j] = stability_test(ax[i], qx[j])

    # plot sol
    plt.figure()
    plt.imshow(stabilities)
    plt.show()

    plt.close()


# Mathieu's equation
# u'(xi) = uu
# uu'(xi) = - f(xi) * u

def particle_ode(y, xi, a, q):
    u, v = y
    dydt = [v, - (a - 2*q*np.cos(2*xi))*u ]
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

def matrix_method(mass, r0, z0, endcapvolt, rfvolt, trapfreq, Z, axis=1):
    '''Does the same but together?'''
    # 0. Define constants for ode and set up time
    n = 1000 # resolution
    xi = np.linspace(0, np.pi, n)
    ax, qx = aq_calc(mass, r0, z0, endcapvolt, rfvolt, trapfreq, Z, axis)

    # 1. set up initial conditions for the 2 independent solutions
    u1_0 = [1,0]
    u2_0 = [0,1]

    # 2. solve to get u1 and u2, 2 independent sols with given init. cond.
    [u1, v1] = odeint(particle_ode, u1_0, xi, args=(ax, qx))
    [u2, v2] = odeint(particle_ode, u2_0, xi, args=(ax, qx))

    # 3. generate M matrix using u(T), namely the last element of the sols
    matrix_M = np.array([[u1[-1], u2[-1]], 
                         [v1[-1], v2[-1]]])
    return matrix_M

def stability_test(a, q, n=1000):
    xi = np.linspace(0, np.pi, n) 
    # 1. set up initial conditions for the 2 independent solutions
    u1_0 = [1.,0.]
    u2_0 = [0.,1.]

    # 2. solve to get u1 and u2, 2 independent sols with given init. cond.
    sol1 = odeint(particle_ode, u1_0, xi, args=(a, q))
    sol2 = odeint(particle_ode, u2_0, xi, args=(a, q))

    u1, v1 = sol1[:,0], sol1[:,1]
    u2, v2 = sol2[:,0], sol2[:,1]
 

    # 3. generate M matrix using u(T), namely the last element of the sols
    matrix_M = np.array([[u1[-1], u2[-1]], 
                         [v1[-1], v2[-1]]])

    # 4. Evaluate stability, if trace < 2 = stable, if > 2 = unstable
    # True - stable, False - unstable
    result = np.abs(np.trace(matrix_M))
    if result < 2:
        return 1
    elif result > 2:
        return 0
    else:
        return 0.5


if __name__ == '__main__':
    main()
