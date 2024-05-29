#!/usr/bin/env python
# coding: utf-8
UNIVERSIDADE FEDERAL DE MINAS GERAIS
INSTUTUTO DE CIÊNCIAS EXATAS
GRADUAÇÃO EM CIÊNCIA DA COMPUTAÇÃO
DISCIPLINA: Introdução a Física Estatística e Computacional

ALUNA: Giulia Monteiro Silva Gomes Vieira
MATRICULA: 2016006492
# #### EXERCÍCIO AVALIATIVO 01: INTEGRAÇÃO DIRETA POR MONTE CARLO

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random

from typing import Callable


# In[2]:


N_SAMPLES = 1000


# ##### Primeiro método

# In[3]:


def first_method(f, lim_inf, lim_sup, n_points):
    points_inside = 0

    for _ in range(n_points):
        x = random.uniform(lim_inf, lim_sup)
        y = random.uniform(0, max(f(lim_inf), f(lim_sup)))

        if 0 <= y <= f(x):
            points_inside += 1

    rectangular_area = (lim_sup - lim_inf) * max(f(lim_inf), f(lim_sup))
    fractional_points_inside = points_inside / n_points

    estimate_integral = rectangular_area * fractional_points_inside
    return estimate_integral


# ##### Segundo método

# In[4]:


def second_method(f, lim_inf, lim_sup, n_points):
    accumulator = 0

    for _ in range(n_points):
        x = random.uniform(lim_inf, lim_sup)
        accumulator += f(x)

    average = accumulator / n_points
    estimate_integral = (lim_sup - lim_inf) * average

    return estimate_integral


# In[5]:


def get_histogram(integral_func, method, lim_inf, lim_sup, n_points, n_samples):
    estimates = []

    for _ in range(n_samples):
        e = method(integral_func, lim_inf, lim_sup, n_points)
        estimates.append(e)

    return estimates


# In[6]:


def plot_histogram(estimates_method_1, estimates_method_2, n_points):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(estimates_method_1, bins=30, color='blue', alpha=0.7, label='Método 1')
    plt.title(f'Histograma - Método 1 (N = {n_points})')
    plt.xlabel('Estimativa da Integral')
    plt.ylabel('Frequência')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(estimates_method_2, bins=30, color='green', alpha=0.7, label='Método 2')
    plt.title(f'Histograma - Método 2 (N = {n_points})')
    plt.xlabel('Estimativa da Integral')
    plt.ylabel('Frequência')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ##### Função 01

# In[7]:


FUNC_1_LIM_INF = 0
FUNC_1_LIM_SUP = 1


# In[8]:


def func_1(x):
    return 1 - x**2


# Para 100 pontos:

# In[9]:


n_points = 100


# In[10]:


first_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[11]:


second_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[12]:


estimates_method_1 = get_histogram(func_1, first_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_1, second_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES)     


# In[13]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 1000 pontos:

# In[14]:


n_points = 1000


# In[15]:


first_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[16]:


second_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[17]:


estimates_method_1 = get_histogram(func_1, first_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_1, second_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES) 


# In[18]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 10000 pontos:

# In[19]:


n_points = 10000


# In[20]:


first_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[21]:


second_method(func_1, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points)


# In[22]:


estimates_method_1 = get_histogram(func_1, first_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_1, second_method, FUNC_1_LIM_INF, FUNC_1_LIM_SUP, n_points, N_SAMPLES)


# In[23]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# ##### Função 02

# In[24]:


FUNC_2_LIM_INF = 0
FUNC_2_LIM_SUP = 1


# In[25]:


def func_2(x):
    return np.exp(x)


# Para 100 pontos:

# In[26]:


n_points = 100


# In[27]:


first_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[28]:


second_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[29]:


estimates_method_1 = get_histogram(func_2, first_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_2, second_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES)     


# In[30]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 1000 pontos:

# In[31]:


n_points = 1000


# In[32]:


first_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[33]:


second_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[34]:


estimates_method_1 = get_histogram(func_2, first_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_2, second_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES) 


# In[35]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 10000 pontos:

# In[36]:


n_points = 10000


# In[37]:


first_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[38]:


second_method(func_2, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points)


# In[39]:


estimates_method_1 = get_histogram(func_2, first_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_2, second_method, FUNC_2_LIM_INF, FUNC_2_LIM_SUP, n_points, N_SAMPLES)


# In[40]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# ##### Função 03

# In[41]:


FUNC_3_LIM_INF = 0
FUNC_3_LIM_SUP = np.pi


# In[42]:


def func_3(x):
    return np.sin(x)**2


# Para 100 pontos:

# In[43]:


n_points = 100


# In[44]:


first_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[45]:


second_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[46]:


estimates_method_1 = get_histogram(func_3, first_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_3, second_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES)     


# In[47]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 1000 pontos:

# In[48]:


n_points = 1000


# In[49]:


first_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[50]:


second_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[51]:


estimates_method_1 = get_histogram(func_3, first_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_3, second_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES) 


# In[52]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# Para 10000 pontos:

# In[53]:


n_points = 10000


# In[54]:


first_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[55]:


second_method(func_3, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points)


# In[56]:


estimates_method_1 = get_histogram(func_3, first_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES)
estimates_method_2 = get_histogram(func_3, second_method, FUNC_3_LIM_INF, FUNC_3_LIM_SUP, n_points, N_SAMPLES)


# In[57]:


plot_histogram(estimates_method_1, estimates_method_2, n_points)


# ##### Função 04

# In[58]:


FUNC_4_LIM_INF = 0
FUNC_4_LIM_SUP = 1


# In[59]:


def func_4(x: list):
    return 1 / ((x[0] + x[1]) * x[2] + (x[3] + x[4]) * x[5] + (x[6] + x[7]) * x[8])


# In[60]:


def second_method_9d(N, func: Callable):
    accumulator = 0
    for _ in range(N):
        accumulator = accumulator + func(np.random.uniform(0, 1, 9))
    return accumulator / N


# In[61]:


def run_second_method_9d(n_points):
    sample = np.zeros(N_SAMPLES)
    for i in range(N_SAMPLES):
        sample[i] = second_method_9d(n_points, func_4)
    return sample.mean(), sample


# Para 1000 pontos:

# In[62]:


n_points = 1000


# In[63]:


mean, sample = run_second_method_9d(n_points)


# In[64]:


plt.hist(sample)
plt.show()


# In[65]:


mean


# ##### Conclusões 

# A distribuição dos valores gerados nos histogramas se aproxima da distribuição normal,
# resultado já esperado, dado o Teorema do Limite Central.
# Além disso, podemos observar que as médias são próximas aos valores analíticos apresentados.
