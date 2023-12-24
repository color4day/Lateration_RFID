#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import scipy as sp
from scipy.optimize import minimize


# In[20]:


phi = []


# In[21]:


#fisher information matrix
# a,b,c,d = 0,0,0,0
# for i in phi:
#     angle = theta+i
#     a += (np.cos(angle))**2
#     b -= (A/2*np.sin(2*(angle)))
#     c = b
#     d -=(A**2*np.sin(angle))


# In[ ]:





# In[22]:


def function(phi):
    A=1
    phi = phi*2*np.pi
    theta = (0.6)*np.pi
    F11,F12,F21,F22 = 0,0,0,0
    for i in phi:
#         print(i)
        angle = theta+i
        F11 += (np.cos(angle))**2
        F12 -= (A/2*np.sin(2*(angle)))
        F21 = F12
        F22 +=(A**2*(np.sin(angle))**2)
        
    FIM = np.array([[F11,F12],[F12,F22]])
    var_theta = FIM[1,1]/np.linalg.det(FIM)
#     var_A = d/det
    return var_theta


# In[23]:


def callback_function(phi):
    value = function(phi)
    function_values.append(value)


# In[24]:


print(function(np.array([0.1,0.2])))


# In[25]:


function_values = []
phi = np.array([0,0,0,0,0.2,0.2,0.2,0.5])
var_theta = minimize(function,phi, method='BFGS', options={'gtol': 1e-3, 'disp': True}, callback=callback_function)


# In[26]:


print(var_theta)
print(var_theta.x*2)


# In[27]:


(1/4)-0.3


# In[28]:


import matplotlib.pyplot as plt


# Plot the function values during optimization
plt.plot(function_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Function Value vs. Iteration")
plt.grid(True)
plt.show()


# In[39]:


def CRLB_fix_other_phi(x):
    A=1
    phi = np.array([0,0,0,0,0.2,0.2,0.2,x])
#     phi = [0,0.1,0.2,0.3,0.5,0.3,0.5,x]
#     phi = phi*2*np.pi
    theta = (0.6)*np.pi
    F11,F12,F21,F22 = 0,0,0,0
    for i in phi:
#         print(i)
        angle = theta+i
        F11 += (np.cos(angle))**2
        F12 -= (A/2*np.sin(2*(angle)))
        F21 = F12
        F22 +=(A**2*(np.sin(angle))**2)
        
    FIM = np.array([[F11,F12],[F12,F22]])
    var_theta = FIM[1,1]/np.linalg.det(FIM)
#     var_A = d/det
    return var_theta


# In[42]:


x = np.linspace(0, 2*np.pi, 1000)
y = []
for i in range(len(x)):
    y.append(CRLB_fix_other_phi(x[i]))

plt.plot(x,y,label = "CRLB_phi8")
plt.axhline(y = np.mean(y),label = "mean")
plt.xlabel("x")
plt.ylabel("CRLB")
plt.title("Plot of CRLB w/ 7 phis fixed")
plt.xticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2), ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

plt.legend()
plt.grid(True)
plt.show()


# In[31]:


# 定义你的函数
def f_heatmap(phi):
    A = 1
#     phi = phi * 2 * np.pi
    theta = (1 / 3) * 2 * np.pi
    F11, F12, F21, F22 = 0, 0, 0, 0
    for i in phi:
        angle = theta + i
        F11 += (np.cos(angle))**2
        F12 -= (A / 2 * np.sin(2 * angle))
        F21 = F12
        F22 += (A**2 * (np.sin(angle))**2)
    FIM = np.array([[F11, F12], [F12, F22]])
    var_theta = FIM[1,1] / np.linalg.det(FIM)
    return var_theta

# 定义 x1 和 x2 的值
x1_values = np.linspace(0, 2 * np.pi, 100)
x2_values = np.linspace(0, 2 * np.pi, 100)

# 计算对应的 y = f(phi) 值
y_values = np.zeros((len(x1_values), len(x2_values)))

for i, x1 in enumerate(x1_values):
    for j, x2 in enumerate(x2_values):
        phi = np.array([0,0,0.2,0.2,0.2,0.5, x1, x2])
        y_values[i, j] = f_heatmap(phi)

# 绘制热图
plt.imshow(y_values, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='y = f(phi)')

plt.xticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2), ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
plt.yticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2), ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

plt.xlabel('x2')
plt.ylabel('x1')
plt.title('Heatmap of y = f(phi) for x1 and x2')
plt.show()


# In[32]:


num_samples = 10000
phi_samples = np.random.uniform(0, 2*np.pi, size=(num_samples, 8))

# 计算对应的 y = f(phi) 值
y_values = np.array([f_heatmap(phi) for phi in phi_samples])

# 绘制直方图
plt.hist(y_values, bins=50, density=False, alpha=0.75, color='blue', edgecolor='black')
#plt.hist(y_values, bins=50, density=True, alpha=0.75, color='blue', edgecolor='black')
plt.xlabel('y = f(phi)')
plt.ylabel('Density')
plt.title('Histogram of y = f(phi) for 10000 random samples')
plt.grid(True)
plt.show()


# In[37]:


num_samples = 10
theta = 0.3
epsilon = 0.0003
phi_samples = np.random.uniform(0, 1, size=(num_samples, 8))
a = np.array([1-theta,1-theta,1-theta,0-theta,0-theta,0-theta,0-theta,0-theta-epsilon])
phi_samples[0] = a#np.array([0.9]*8)
print(phi_samples)
var = [-1]*num_samples
for i in range(len(phi_samples)):
    var[i] = function(phi_samples[i])
print(var)


# In[38]:


x = np.arange(1,11)
plt.plot(x,var)
plt.xlabel("i_th combination")
plt.ylabel("var_theta")
plt.title("var_theta for different combination of phi")

plt.show()


# In[35]:


max(y_values)


# In[36]:


def f_changetheta(phi,theta):
    A = 1
#     phi = phi * 2 * np.pi
#     theta = (1 / 3) * 2 * np.pi
    F11, F12, F21, F22 = 0, 0, 0, 0
    for i in phi:
        angle = theta + i
        F11 += (np.cos(angle))**2
        F12 -= (A / 2 * np.sin(2 * angle))
        F21 = F12
        F22 += (A**2 * (np.sin(angle))**2)
    FIM = np.array([[F11, F12], [F12, F22]])
    var_theta = FIM[1,1] / np.linalg.det(FIM)
    return var_theta

theta = np.array([0,0.3,0.6,0.9])*2*np.pi
# num_samples = 10000
# phi_samples = np.random.uniform(0, 2*np.pi, size=(num_samples, 8))
for i in theta:
# 计算对应的 y = f(phi) 值
    y_values = np.array([f_changetheta(phi,i) for phi in phi_samples])

# 绘制直方图
    plt.hist(y_values, bins=50, density=True, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('y = f(phi)')
    plt.ylabel('Density')
    plt.title('Histogram of y = f(phi) for 10000 random samples,theta = '+str(i))
    plt.grid(True)
    plt.show()


# In[ ]:




