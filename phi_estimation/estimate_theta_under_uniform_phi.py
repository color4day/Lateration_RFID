#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[192]:


def estimate_theta(phi,y):
    C = np.cos(phi)
    S = np.sin(phi)
    H = np.vstack((C,S)).T
#     print(H.shape,y.shape)
    alpha = np.matmul(np.linalg.pinv(H),y)
    print(alpha)
    t_hat = np.arctan(alpha[0]/alpha[1])
    return t_hat


# In[193]:


def function(theta, phi):
    A=1
#     phi = phi*2*np.pi
#     theta = (0.6)*np.pi
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


# In[221]:


def gen_y(theta,phi,sigma=1,A = 10,ch = 8,repeat = 5):
    y = np.random.randn(ch,repeat)
#     print(y.shape)
    for i in range(ch):
        for j in range(repeat):
            y[i][j] += A*np.sin(theta+phi[i])
    return y


# In[222]:


phi = np.linspace(0,2*np.pi,8)
theta_len = 100
theta = np.linspace(0,2*np.pi,theta_len)
theta_hat = np.zeros(theta_len)
crlb_uniformphi = np.zeros(theta_len)
crlb_thetahat = np.zeros(theta_len)
crlb_theta = np.zeros(theta_len)
#noise = np.linspace(-0.1,0.1,8)
noise = 0.1*np.random.randn(8)
for i in range(len(theta)):
    y = gen_y(theta[i],phi)
    theta_hat[i] = estimate_theta(phi,np.mean(y,axis = 1))
#     print(theta[i],theta_hat[i])
    phi_hat = np.ones(8)*(np.pi-theta_hat[i])+noise
    phi_truevalue= np.ones(8)*(np.pi-theta[i])+noise
    print(phi_truevalue)
    crlb_uniformphi[i] = function(theta[i],phi)
    crlb_thetahat[i] = function(theta[i],phi_hat)
    crlb_theta[i] = function(theta[i],phi_truevalue)


# In[223]:


mapped_theta = (theta + np.pi/2) % (np.pi)- np.pi/2
plt.plot(mapped_theta,label = "theta")
plt.plot(theta_hat+np.pi%np.pi, label = "theta_hat")
plt.legend()
plt.show()


# In[225]:


mapped_theta = (theta + np.pi/2) % (np.pi)- np.pi/2
plt.plot(theta,crlb_theta, label = "CRLB_theta")

plt.plot(theta,crlb_thetahat,label = "CRLB_thetahat")
plt.plot(theta,crlb_uniformphi,label = "uniform_phi")
#plt.ylim(-100,100)
plt.legend()
plt.title("repeat 5 times")
plt.show()


# In[226]:


mapped_theta = (theta + np.pi/2) % (np.pi)- np.pi/2
plt.plot(theta,crlb_theta, label = "CRLB_theta")

#plt.ylim(-100,100)
plt.legend()

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# 示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建第一个 y 轴并绘制 y1
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color='blue', label='y1')
ax1.set_xlabel('X轴')
ax1.set_ylabel('y1', color='blue')
ax1.tick_params('y', colors='blue')

# 创建第二个 y 轴并绘制 y2
ax2 = ax1.twinx()
ax2.plot(x, y2, color='red', label='y2')
ax2.set_ylabel('y2', color='red')
ax2.tick_params('y', colors='red')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# 标题
plt.title('yy图')

# 显示图形
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()

ax1.plot(theta)
ax1.set_title('sin(x) and exp(x)')

ax2 = ax1.twinx()
ax2.plot(mapped_theta,'C1')

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the first y-axis and plot y1
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color='blue', label='y1')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('y1', color='blue')
ax1.tick_params('y', colors='blue')

# Create the second y-axis and plot y2
ax2 = ax1.twinx()
ax2.plot(x, y2, color='red', label='y2')
ax2.set_ylabel('y2', color='red')
ax2.tick_params('y', colors='red')

# Adding a legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Title
plt.title('yy Plot')

# Show the plot
plt.show()


# In[ ]:


y = np.random.randn(8)
print(y)
y1 = np.random.randn(8)
print(y)


# In[ ]:




