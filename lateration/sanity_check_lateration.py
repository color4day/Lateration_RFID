#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[46]:


# #get coordinates of anchors:
# def get_points(x, y, d):
#     a = np.random.uniform(-1, 1)
#     b = np.random.uniform(-1, 1)
#     c = a**2 + b**2
#     distance = np.sqrt(c)
#     if c < 1:
#         x = np.append(x, a)
#         print(a)
#         y = np.append(y, b)
#         #d2 = np.append(d2, c)
#         d = np.append(d,distance)
#         #return x, y, d2
#     else:
#         x,y,d = get_points(x,y,d)
#     return x, y, d


# In[48]:


def get_points(x, y, d):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    c = a**2 + b**2
    distance = np.sqrt(c)
    if c < 1:
        x = np.append(x, a)
        print(a)
        y = np.append(y, b)
        #d2 = np.append(d2, c)
        d = np.append(d,distance)
        #return x, y, d2
    else:
        x,y,d = get_points(x,y,d)
    return x, y, d

x = np.array([])
y = np.array([])
d = np.array([])

for i in range(3):
    x, y, d = get_points(x, y, d)

print(x)
print(y)
print("d =", d)
d_matrix = np.tile(d2, (1000, 1))

print(d_matrix)


# In[ ]:





# In[23]:


#get estimated target coordinates with anchor as inputs
def get_A(x,y):
    a_11 = 2*(x[1]-x[0])
    a_12 = 2*(y[1]-y[0])
    a_21 = 2*(x[2]-x[0])
    a_22 = 2*(y[2]-y[0])
    return np.matrix([[a_11, a_12], [a_21, a_22]])


# In[49]:


def get_B(x,y,d):
    b_1 = d[0]**2-d[1]**2+x[1]**2+y[1]**2-x[0]**2-y[0]**2
    b_2 = d[0]**2-d[2]**2+x[2]**2+y[2]**2-x[0]**2-y[0]**2
    return np.matrix([[b_1], [b_2]])


# In[50]:


x = np.array([])
y = np.array([])
d = np.array([])

for i in range(3):
    x, y, d = get_points(x, y, d)

print(x)
print(y)
print("d =", d)
d.size


# for i in range(p_num):
#     newd = d2+np.random.normal(0,0.01)
#     d_noise.append(newd)

# A = get_A(x,y)
# B = get_B(x,y,d2)
# coor = np.linalg.pinv(A)*B
# print(A)
# print(B)
# print(coor)


# In[54]:


iter_num = 1000

d_matrix = np.tile(d,(1000,1))
d_matrix.shape


# Generate independent normal noise
noise_mean = 0
noise_std_dev = 0.01
noise = np.random.normal(noise_mean, noise_std_dev, d2_matrix.shape)
d_noise = d_matrix + noise
d_noise


# In[55]:


d_noise[0]


# In[69]:


co_x = np.array([])
co_y = np.array([])
print(co_x.shape, co_y.shape)

for i in range(iter_num):
    A = get_A(x,y)
    B = get_B(x,y,d_noise[i])
    coor = np.linalg.pinv(A)*B
    co_x = np.append(co_x,coor[0])
    co_y = np.append(co_y,coor[1])
print(co_x.shape, co_y.shape)
type(co_x)


# In[71]:


xpoints = np.array(x)
ypoints = np.array(y)
plt.scatter(xpoints, ypoints)
plt.scatter(co_x, co_y)
plt.show()


# In[72]:


# Create a histogram
plt.hist(co_x, bins=20, color='blue', alpha=0.7)  # Adjust the number of bins as needed

# Add labels and a title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of 1D Array')

# Display the histogram
plt.show()


# In[77]:


variance = np.var(co_x)
std_deviation = np.std(co_x)
print(variance, std_deviation)


# In[79]:


# Create a histogram
plt.hist(co_y, bins=20, color='blue', alpha=0.7)  # Adjust the number of bins as needed

# Add labels and a title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of 1D Array')

# Display the histogram
plt.show()


# In[78]:


variance = np.var(co_y)
std_deviation = np.std(co_y)
print(variance, std_deviation)


# In[80]:


from scipy import integrate

# 定义被积函数
def integrand(x1, x2):
    if x1 < x2:
        return x2 - x1
    else:
        return x1 - x2

# 执行数值积分
result, _ = integrate.dblquad(integrand, -1, 1, lambda x1: -1, lambda x1: 1)

# 期望值为结果
expectation = 0.25 * result

print("期望值 E(|x1 - x2|) =", expectation)


# In[ ]:




