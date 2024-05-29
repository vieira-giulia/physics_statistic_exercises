#!/usr/bin/env python
# coding: utf-8
UNIVERSIDADE FEDERAL DE MINAS GERAIS
INSTUTUTO DE CIÊNCIAS EXATAS
GRADUAÇÃO EM CIÊNCIA DA COMPUTAÇÃO
DISCIPLINA: Introdução a Física Estatística e Computacional

ALUNA: Giulia Monteiro Silva Gomes Vieira
MATRICULA: 2016006492
# #### EXERCÍCIO AVALIATIVO 03: Propriedades Ising 2D

# In[1]:


import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def calor_especifico(energies, temperature, size):
    # beta = 1 / temperature
    energy_squared_mean = np.mean(np.array(energies)**2)
    energy_mean_squared = np.mean(energies)**2
    specific_heat = (energy_squared_mean - energy_mean_squared) / (size**2 * temperature**2)
    return specific_heat


# In[5]:


def susceptibilidade(magnetizations, temperature, size):
    # beta = 1 / temperature
    magnetization_squared_mean = np.mean(np.array(magnetizations)**2)
    magnetization_mean_squared = np.mean(magnetizations)**2
    susceptibility = (magnetization_squared_mean - magnetization_mean_squared) / (size**2 * temperature)
    return susceptibility


# In[6]:


def error(arr):
    return np.sqrt(
        np.sum(np.power(arr - np.average(arr), 2)) / (arr.size * (arr.size - 1))
    )


# #### 01. Escolha dos parâmetros: motivação
Tamanho do Sistema (𝐿): Depende da capacidade computacional e do objetivo da simulação. Tamanhos pequenos facilitam simulações inicias, tamanhos maiores são geralmente mais fidedignos a realidade. Dadas as simulações do exercício anterior eu julgaria que tamanhos maiores também compensam ruídos e têm comportamento mais ordenado.Temperatura de Simulação (𝑇): Grandes variações poderiam trazer mais informações sobre os comportamentos da energia e magnetismo. Uma forma de abordar esse problema seria começar próximo a T = 0C e aumentar graduamente.Número de Passos para Termalização (𝑁456): Grande o suficiente para garantir que o sistema alcance o equilíbrio térmico.Número de Passos para Médias Termodinâmicas (𝑁789): Suficiente para garantir boas estatísticas, o que é complicado de definir, sua escolha é possivelmente mais baseada em tentativa e erro.
# #### 02. Comportamento das principais grandezas termodinâmicas
Energia por Spin: Deve diminuir à medida que a temperatura aumenta.Magnetização por Spin: Deve diminuir à medida que a temperatura aumenta.Calor Específico: Pode apresentar picos indicando transições de fase. Geralmente proporcional á temperatura do sistema.Susceptibilidade Magnética: Pode aumentar à medida que a temperatura se aproxima da temperatura crítica.
# ##### 03. Variação com o Tamanho do Sistema
Para uma mesma temperatura, quanto maior o tamanho do sistema mais distante energia está de zero. Isso infere que sistemas maiores não só requerem mais passos, mas também causam maior desequilíbrio de energia.
# In[7]:


t = 0.5
L = [10, 30, 50, 70, 90]
steps = 1000

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

Suponha a instancia abaixo para as questões 04 e 05:
# ##### 04. Comportamento dos erros estatísticos:
Os erros estatísticos crescem com o tamanho do sistema, provavelmente porque sistemas maiores tendem a ser mais caóticos.
# In[8]:


L = 100
T = 5.0
steps = 1000
E, M = metropolis(L, T, steps)


# In[9]:


error(E)


# In[10]:


error(M)


# In[11]:


L = 500
E, M = metropolis(L, T, steps)


# In[12]:


error(E)


# In[13]:


error(M)


# ##### 05. Identificação de Fases do Sistema:
O sistema muda de fase quando muda de estado termodinamico, que neste caso pode ser influenciado pelo magnetismo ou pela temperatura. Caso seja determinado pela temperatura eu entenderia que T=0 marca uma mudanaça de fase em que a energia do sistema está abaixando. Temperaturas mais baixas também parecem inferir queda em magnetismo, o que me direciona ainda mais para esta possível fase do sistema. Na próxima sessão também vemos como um T próximo de zero se comporta de maneira muito diferente dos demais.
# ###### 06. Estimativa da Temperatura de Transição:
Temperatura zero, ao observarmos o gráfico abaixo, o sistema se comporta muito discrepantemente ao se aproximar do zero, comparado ao mesmo intervalo para valores distantes dele.
# In[14]:


l = 200
T = [0.5, 10.5, 20.5, 30.5, 40.5]
steps = 1000

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

