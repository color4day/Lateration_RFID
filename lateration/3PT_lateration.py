#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sympy as sym


# In[5]:


x1 = sym.Symbol("x1")
x2 = sym.Symbol("x2")
x3 = sym.Symbol("x3")
y1 = sym.Symbol("y1")
y2 = sym.Symbol("y2")
y3 = sym.Symbol("y3")
d1 = sym.Symbol("d1")
d2 = sym.Symbol("d2")
d3 = sym.Symbol("d3")

x = sym.Symbol("x")
y = sym.Symbol("y")

A = 2*(sym.Matrix([[x2,y2],[x3,y3]]) - ((sym.ones(2,1)*x1).row_join(sym.ones(2,1)*y1)) )
B = sym.Matrix([d1**2-d2**2+x2**2+y2**2-x1**2-y1**2, d1**2-d3**2+x3**2+y3**2-x1**2-y1**2])
display(A)
display(B)


# In[6]:


target = (A.T*A)**(-1)*A.T * B


# In[7]:


x = target[0].simplify()
y = target[1].simplify()
display(x)
display(y)


# In[8]:


display((sym.diff(x,d1)).simplify())
display((sym.diff(x,d2)).simplify())
display((sym.diff(x,d3)).simplify())


# In[9]:


display((sym.diff(y,d1)).simplify())
display((sym.diff(y,d2)).simplify())
display((sym.diff(y,d3)).simplify())
#d/x


# In[10]:


display((sym.diff(x,x1)).simplify())
display((sym.diff(x,y1)).simplify())
display((sym.diff(y,x1)).simplify())
display((sym.diff(y,y1)).simplify())
#(d/x+d^2/x^2*(d/x))*d^2/x^2 +d/x


# In[11]:


import math
variable = [x1,x2,x3,y1,y2,y3,d1,d2,d3]
value = [0.56775614, 0.77447097, 0.63789222,-0.743638 ,  -0.24222297,  0.38177948, 0.93559847 ,0.81146612, 0.74341244]
#subs_dict = {x1:0.57000136, x2:-0.62848133, x3:0.02130644,y1:-0.14788517, y2:-0.71885565, y3:-0.05859919, d1:0.58887314 , d2: 0.95485194 ,d3:0.06235246}
subs_dict = {}
for i in range(len(variable)):
    subs_dict[variable[i]] = value[i]
    
result_x = [sym.diff(x,d1).subs(subs_dict),sym.diff(x,d2).subs(subs_dict),sym.diff(x,d3).subs(subs_dict)]
result_y = [sym.diff(y,d1).subs(subs_dict),sym.diff(y,d2).subs(subs_dict),sym.diff(y,d3).subs(subs_dict)]

ans_x,ans_y = 0,0
for i in range(len(result_x)):
    print(result_x[i]**2)
    ans_x+= result_x[i]**2*0.0001
    ans_y+= result_y[i]**2*0.0001
    print(ans_x, ans_y)
print(math.sqrt(ans_x),math.sqrt(ans_y))
#print(i**2 for i in result)


# In[ ]:




