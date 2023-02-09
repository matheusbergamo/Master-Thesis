#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import scipy.optimize as sci
from scipy import constants
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('font', size=16)
plt.rcParams["figure.figsize"] = (7,4.2023)
#import sigfig as sf # Pacote para expressar incertezas e dígitos significativos
import pandas as pd
from scipy.integrate import quad


# In[2]:


# Basic Espectrum
p1 = 2.09487E-01
p2 = 2.07754E+00
p3 = 6.75167E-01

def p4(d,r): 
    
    g1 = 1.59395E-02
    g2 = -4.84698E-05
    g3 = 8.60724E-03
    g4 = 7.00749E+00
    g5 = 5.62439E-01
    a5 = g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))
    
    a6 = 1.984E-04
    
    a7 = 5.673E-01
    
    a8 = 1.147E-03
    
    return a5 + a6*d/(1 + a7*np.exp(a8*d))

p5 = 1.23027E+02
p6 = 2.01221E+00
p7 = 8.72599E-04
p8 = 2.35930E-14
p9 = 1.33919E+00
p10 = 1.22991E-07
p11 = 1.19145E+00

def p12(d,r): 
    
    g1 = 1.72867E+03
    g2 = -2.84704E+01
    g3 = -1.97361E+03
    g4 = 2.78420E+00
    g5 = 9.01841E+00
    a9 = g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))
    
    h1 = 8.05282E-04
    h2 = 4.97528E-05
    h3 = 1.26287E-03
    h4 = 1.44990E+01
    h5 = 5.98991E+00
    a10 = h1 + h2*r + h3/(1 + np.exp((r - h4)/h5))
    
    i1 = -9.92421E-01
    i2 = 2.24381E-01
    i3 = 2.44255E+00
    i4 = 1.06884E+01
    i5 = 2.17121E+00
    a11 = i1 + i2*r + i3/(1 + np.exp((r - i4)/i5))
    
    a12 = 1.147E-03
    
    return a9*(np.exp(-a10*d) + a11*np.exp(-a12*d))

def EϕB(E,d,r):
    
    evap = p1*((E/p2)**p3)*np.exp(-E/p2)
    
    gaus = p4(d,r)*np.exp(-((np.log10(E) - np.log10(p5))**2)/(2*np.log10(p6)**2))
    
    conti = p7*np.log10(E/p8)*(1 + np.tanh(p9*np.log10(E/p10)))*(1 - np.tanh(p11*np.log10(E/p12(d,r))))
    
    return (conti + evap + gaus)


# In[3]:


# Local Effect
def Local(E,w):
    g1 = -0.023499
    
    g2 = -0.012938
    
    h31 = -25.184
    h32 = 2.7298
    h33 = 0.071526
    g3 = 10**(h31 + h32/(w + h33))
    
    g4 = 0.96889
    
    h51 = 0.3479
    h52 = 3.3493
    h53 = -1.5744
    g5 = h51 + h52*w + h53*w**2
    
    expo = g1 + g2*np.log10(E/g3)*(1 - np.tanh(g4*np.log10(E/g5)))
    
    return 10**expo


# In[4]:


# Thermal Neutrons
def ΦT(E, w):
    h1 = 0.118
    h2 = 0.14438
    h3 = 3.8733
    h4 = 0.65298
    h5 = 42.752
    g6 = (h1 + h2*np.exp(-h3*w))/(1 + h4*np.exp(-h5*w))
    
    ET = 2.5E-8
    
    return g6*((E/ET)**2)*np.exp(-E/ET)


# In[5]:


# Solar modulation
#For solar minimum
def ΦN_min(d,r): 
    
    g1 = 1.12855E+02
    g2 = -3.03495E+00
    g3 = 6.06068E+01
    g4 = 6.75256E+00
    g5 = 1.37460E+00
    c1 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    g1 = 8.04612E-03
    g2 = -2.14822E-05
    g3 = 1.08329E-03 
    g4 = 2.67391E+00
    g5 = 3.49189E+00
    c2 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    g1 = 9.90742E-01
    g2 = 4.32602E-04
    g3 = -1.29238E+00
    g4 = -4.96557E+00
    g5 = 1.86193E+00
    c3 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    g1 = 8.61572E-03
    g2 = -3.57673E-05
    g3 = 1.23255E-02
    g4 = -6.86389E+00
    g5 = 3.60618E+00
    c4 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    return c1 * (np.exp(-c2*d) - c3*np.exp(-c4*d))


# In[6]:


# Solar maximum
def ΦN_max(d,r):
    
    g1 = 1.15802E+02
    g2 = -2.85820E+00
    g3 = 4.75639E+01
    g4 = 5.04684E+00
    g5 = 8.32753E-01
    c1 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))
 
    g1 = 8.16401E-03
    g2 = -2.42661E-05
    g3 = 4.15214E-04
    g4 = 5.26576E+00
    g5 = 1.91077E+00
    c2 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    g1 = 9.96854E-01
    g2 = 1.63506E-04
    g3 = -4.54753E-02
    g4 = -1.86794E+00
    g5 = 3.81076E+00
    c3 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    g1 = 7.30784E-03
    g2 = 1.93345E-05
    g3 = 2.65840E-03
    g4 = 4.57083E+00
    g5 = 6.71247E+00
    c4 =  g1 + g2*r + g3/(1 + np.exp((r - g4)/g5))

    return c1 * (np.exp(-c2*d) - c3*np.exp(-c4*d))


# In[7]:


#Tomando a média de modulação solar
def EΦ(E,d,r,w):
    return (EϕB(E,d,r)*Local(E,w) + ΦT(E, w))*(ΦN_min(d,r) + ΦN_max(d,r))/2 

#Tomando a média de modulação solar em altitudes mais altas, sem nêutrons térmicos
def EΦalt(E,d,r,w):
    return (EϕB(E,d,r)*Local(E,w))*(ΦN_min(d,r) + ΦN_max(d,r))/2 

