#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sympy as sym


# In[3]:


x1 = sym.Symbol("x1")
x2 = sym.Symbol("x2")
x3 = sym.Symbol("x3")
x4 = sym.Symbol("x4")
y1 = sym.Symbol("y1")
y2 = sym.Symbol("y2")
y3 = sym.Symbol("y3")
y4 = sym.Symbol("y4")
d1 = sym.Symbol("d1")
d2 = sym.Symbol("d2")
d3 = sym.Symbol("d3")
d4 = sym.Symbol("d4")

x = sym.Symbol("x")
y = sym.Symbol("y")

A = (sym.Matrix([[x2,y2],[x3,y3],[x4,y4]]) - ((sym.ones(3,1)*x1).row_join(sym.ones(3,1)*y1)) )
B = sym.Matrix([d1**2-d2**2+x2**2+y2**2-x1**2-y1**2, d1**2-d3**2+x3**2+y3**2-x1**2-y1**2,
                d1**2-d4**2+x4**2+y4**2-x1**2-y1**2])
display(A)
display(B)


# In[4]:


target = (A.T*A)**(-1)*A.T * B


# In[5]:


x = target[0].simplify()
y = target[1].simplify()
display(x)
display(y)


# In[6]:


display((sym.diff(x,d1)).simplify())
display((sym.diff(x,d2)).simplify())
display((sym.diff(x,d3)).simplify())


# In[7]:


display((sym.diff(y,d1)).simplify())
display((sym.diff(y,d2)).simplify())
display((sym.diff(y,d3)).simplify())


# In[ ]:




