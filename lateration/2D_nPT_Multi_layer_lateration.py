#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[24]:


#generate point at same layer
def gen_layer(layer,points=5):
#     x = np.array([-1,0,1,2])
#     y = np.array([-layer, -layer+1,-layer,-layer+1])
    x = np.array(range(-1, points-1))
    y = np.tile(np.array([-layer, -layer+1]), points//2+1)[:points]
    return x,y
#for each point in next layer, 
#generate distance with target
def getD(x1,y1,xt,yt):
    d = math.sqrt((x1-xt)**2+(y1-yt)**2)
    return d
#generate noised distance
def addNoise(d):
    # Generate independent normal noise
    noise_mean = 0
    noise_std_dev = 0.01
    noise = np.random.normal(noise_mean, noise_std_dev)
    return d+noise


# In[23]:


# points = 5
# layer = 5
# x = np.array([range(-1,points-1)])
# y = np.tile(np.array([-layer, -layer+1]), points//+1)[:points]
# print(y)


# In[25]:


def addNoise(d):
    # Generate independent normal noise
    noise_mean = 0
    noise_std_dev = 0.01
    noise = np.random.normal(noise_mean, noise_std_dev)
    return d+noise
print(addNoise(1))


# In[31]:


print(x_total[0],y_total[0])
def get_A(x,y,points = 5):#几个点就是多少points A dimension:points-1*2
    a = np.matrix([])  # 使用 np.matrix 创建空矩阵
    for i in range(1, points):
        a_curr = 2 * np.array([x[i] - x[0], y[i] - y[0]])  # 使用 np.array 创建数组
#         print(a_curr)
        if a.size == 0:
            a = np.matrix(a_curr)
        else:
            a = np.vstack((a, a_curr))  # 使用 np.vstack 来添加行
#             print(a)
    return a
    #a = a.reshape(-1, 2) 

print(get_A(x_total[0],y_total[0],points))    


# In[32]:


def get_B(x,y,d,points = 4):
    b = np.matrix([]) 
    for i in range(1, points):
        b_curr = np.array([d[0]**2-d[i]**2+x[i]**2+y[i]**2-x[0]**2-y[0]**2])
        print(b_curr)
        if b.size == 0:
            b = np.matrix(b_curr)
        else:
            b = np.vstack((b, b_curr))  # 使用 np.vstack 来添加行
            print(b)
    #b_1 = d[0]**2-d[1]**2+x[1]**2+y[1]**2-x[0]**2-y[0]**2
    #b_2 = d[0]**2-d[2]**2+x[2]**2+y[2]**2-x[0]**2-y[0]**2
    return b
# print(get_B(x_total[0],y_total[0],points))    


# In[33]:


def lateration(x,y,d):
    A = get_A(x,y,points)
    B = get_B(x,y,d,points)
    coor = np.linalg.pinv(A)*B
    coor = coor.A1
    print("coor = ", coor)
    return coor[0], coor[1]


# In[44]:


layer = 10
points = 6 # # of points used for lateration
x = []
y = []
for i in range(layer):
    x_curr = gen_layer(i,points)[0]
    y_curr = gen_layer(i,points)[1]
    x.append(x_curr)
    y.append(y_curr)
x_total = np.array(x)
y_total = np.array(y)
print(x_total)
print(y_total)
d_0 = [[]for i in range(points)]
for i in range(points):#for ith point of d
    for j in range(points):
        d = getD(x[0][j],y[0][j],x[1][i],y[1][i])
#         print("d=", d)
#         d_2 = addNoise(d)
#         print("dnoise = ", d_2)
        d_0[i].append(addNoise(d))
print(d_0)
d_0 = np.array(d_0)
for i in range(layer):
    plt.plot(x_total[i],y_total[i])


# In[45]:


# x = [x1,x2,x3]
# y = [y1,y2,y3]
# d = [[d11,d21,d31],
#     [d12,d22,d32],
#     [d13,d23,d33]]


# In[46]:


output = {}
iteration = 1000
noise_mean = 0
noise_std_dev = 0.01


# In[47]:


for iter in range(iteration):
    x_curr = x_total[0]#3*1000
    y_curr = y_total[0]#3*1000
    noise = np.random.normal(noise_mean, noise_std_dev, d_0.shape)
    d0_noise = d_0 + noise
    d_curr = d0_noise #3*3*1000
    for l in range(layer-1):
        x_next,y_next = [],[]
        d_next = [[]for i in range(points)]
        for i in range(points):
            co_x,co_y = lateration(x_curr,y_curr,d_curr[i])
            #print("co_x,co_y = ",co_x,co_y)
            for j in range(points):
                d_next_single = getD(x_curr[i],y_curr[i],x_total[l+1][j],y_total[l+1][j])
                d_next_single = addNoise(d_next_single)
                d_next[j].append(d_next_single)
            x_next.append(co_x)
            y_next.append(co_y)
        print("x_next = ", x_next)
        print("y_next = ", y_next)
        print("d_next = ", d_next)
        x_curr = x_next
        var_name_x = f'x{l+1}'
        if var_name_x not in output:
            output[var_name_x] = []
        output[var_name_x].append(x_next)
        y_curr = y_next
        var_name_y = f'y{l+1}'
        if var_name_y not in output:
            output[var_name_y] = []
        output[var_name_y].append(y_next)
        d_curr = d_next


# In[49]:


# x1 = output["x1"]
varlist = []
stdlist = []
for l in range(layer-1):
    var_name_x = f'x{l+1}'
    #print(var_name_x)
    tempx = output[var_name_x]
    #print("tx = ",tempx)
    vlist_sub = []
    stdlist_sub = []
    for j in range(points):
        col_tempx = [tempx[i][j]for i in range(iteration)]
        #print("ctx = ", col_tempx)
        variance = np.var(col_tempx)
        vlist_sub.append(variance)
        std_deviation = np.std(col_tempx)
        stdlist_sub.append(std_deviation)
    varlist.append(vlist_sub)
    stdlist.append(stdlist_sub)
print("varlist = ", varlist)
print("stdlist = ", stdlist)


# In[50]:


v0 = [varlist[i][0]for i in range(layer-1)]
v1 = [varlist[i][1]for i in range(layer-1)]
print(v0)


# In[51]:


idx = np.arange(0, len(v0))
plt.plot(idx+1,v0)
plt.plot(idx+1,v1)
plt.xlabel("i_th layer")
plt.ylabel("variance")
plt.title(str(points)+" points")


# In[52]:


# print(v0/v0[0])
timesx = v0/v0[0]
timesy = v1/v1[0]
idx2 = np.arange(0,5)
plt.plot(idx2+1,timesx[0:5],label = "var of x")
plt.plot(idx2+1,timesy[0:5], label = "var of y")
plt.xticks(np.arange(1, 6, 1))
plt.legend()
plt.xlabel("i_th layer")
plt.ylabel("Times of original variance")
plt.title(str(points)+" points")
plt.show()


# In[53]:


plt.plot(np.log(v0))
plt.plot(np.log(v1))


# In[ ]:





# In[ ]:


plt.plot(np.log(np.log(v0)))
plt.plot(np.log(np.log(v1)))


# In[ ]:




