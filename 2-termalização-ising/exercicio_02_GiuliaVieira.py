#!/usr/bin/env python
# coding: utf-8
UNIVERSIDADE FEDERAL DE MINAS GERAIS
INSTUTUTO DE CIÊNCIAS EXATAS
GRADUAÇÃO EM CIÊNCIA DA COMPUTAÇÃO
DISCIPLINA: Introdução a Física Estatística e Computacional

ALUNA: Giulia Monteiro Silva Gomes Vieira
MATRICULA: 2016006492
# #### EXERCÍCIO AVALIATIVO 02: Ising

# In[1]:


import numpy as np
from numba import jit
import matplotlib.pyplot as plt


# In[2]:


# Função fornecida pelo professor
@jit(nopython=True)
def vizinhos(N: int):
    # Define a tabela de vizinhos
    L = int(np.sqrt(N))
    viz = np.zeros((N, 4), dtype=np.int16)
    for k in range(N):
        viz[k, 0] = k + 1
        if (k + 1) % L == 0:
            viz[k, 0] = k + 1 - L
        viz[k, 1] = k + L
        if k > (N - L - 1):
            viz[k, 1] = k + L - N
        viz[k, 2] = k - 1
        if k % L == 0:
            viz[k, 2] = k + L - 1
        viz[k, 3] = k - L
        if k < L:
            viz[k, 3] = k + N - L
    return viz


# In[3]:


@jit(nopython=True)
def metropolis(L, T, steps):
    energy = np.zeros(steps)
    magnetization = np.zeros(steps)

    spins = np.array([-1, 1])

    energy_var = np.array([8.0, 4.0, 0.0, -4.0, -8.0])
    power_of = np.exp(energy_var * 1 / T)

    N = L * L
    S = np.random.choice(spins, N)

    v = vizinhos(N)

    for i in range(steps):
        for k in np.arange(N):
            j = int(S[k] * np.sum(S[v[k]]) * 0.5 + 2)
            if np.random.rand() < power_of[j]:
                S[k] = -1 * S[k]
        energy[i] = -np.sum(S * (S[v[:, 0]] + S[v[:, 1]]))
        magnetization[i] = np.sum(S)

    return energy, magnetization


# In[4]:


L = 32
T = 1.5
steps = 1000


# In[5]:


energy, magnetization = metropolis(L, T, steps)


# ##### Exemplo de comportamento da energia vs passos

# In[6]:


plt.plot(energy, label='Energia')
plt.title('Energia vs. Passos de Monte Carolo')
plt.xlabel('Passos')
plt.ylabel('Energia')
plt.legend()
plt.show()


# ##### Exemplo de comportamento da magnetização vs passos

# In[7]:


plt.plot(magnetization, label='Magnetização')
plt.title('Magnetização vs. Passos de Monte Carlo')
plt.xlabel('Passos')
plt.ylabel('Magnetização')
plt.legend()
plt.show()


# In[8]:


L = [10, 30, 50, 70, 90]
T = [0.5, 1.5, 2.5, 3.5, 4.5]
steps = 1000


# ###### Para tamanhos fixos, vamos ver como energias e magnetizações se comportam em diferentes temperaturas

# In[9]:


l = 10

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for t in T:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Temp {t}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Temp {t}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps (Size {l})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step (Size {l})')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[10]:


l = 30

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for t in T:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Temp {t}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Temp {t}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps (Size {l})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step (Size {l})')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[11]:


l = 50

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for t in T:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Temp {t}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Temp {t}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps (Size {l})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step (Size {l})')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[12]:


l = 70

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for t in T:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Temp {t}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Temp {t}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps (Size {l})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step (Size {l})')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[13]:


l = 90

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for t in T:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Temp {t}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Temp {t}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps (Size {l})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step (Size {l})')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# ###### Para temperaturas fixas, vamos ver como energias e magnetizações se comportam em diferentes tamanhos

# In[24]:


t = 0.5

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through sizes
for l in L:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Size {l}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Size {l}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Size in Temp {t}')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Size in Temp {t}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[25]:


t = 1.5

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for l in L:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Size {l}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Size {l}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Size in Temp {t}')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Size in Temp {t}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[26]:


t = 2.5

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for l in L:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Size {l}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Size {l}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps in Temp {t})')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step in Temp {t}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[27]:


t = 3.5

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for l in L:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Size {l}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Size {l}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps in Temp {t}')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Steps in Temp {t}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[28]:


t = 4.5

# Create a single figure and axes for the comparison
fig, axes = plt.subplots(ncols=2, figsize=(12, 5))


# Iterate through temperatures
for l in L:
    # Simulate the Ising model
    energies, magnetizations = metropolis(l, t, steps)

    # Plot energy vs steps for current temperature
    axes[0].plot(energies, label=f'Size {l}')
    
    # Plot magnetization vs steps for current tempo
    axes[1].plot(magnetizations, label=f'Size {l}')   

# Set plot labels and title
axes[0].set_title(f'Energy vs. Steps in Temp {t}')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Energy')
axes[0].legend()

axes[1].set_title(f'Magnetization vs. Step in Temp {t}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Magnetization')
axes[1].legend()

# Show the plot
plt.show()


# In[ ]:




