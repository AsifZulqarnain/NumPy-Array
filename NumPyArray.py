#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np

empty_array = np.empty([2, 3])
print(empty_array)


# In[9]:


import numpy as np


full_array = np.full([2, 3], 2)
print(full_array)


# In[4]:


n=np.ones(4)
print(n)


# In[5]:


o=np.zeros(6)
print(o)


# In[16]:


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_to_find = np.array([4, 5, 6])
if np.any(np.all(np.equal(arr, row_to_find), axis=1)):
    print("The row exists in the array")
else:
    print("The row does not exist in the array")
    


# In[10]:


import numpy as np

empty_array = np.empty([2, 3])
print(empty_array)


# In[15]:


num=np.eye(3,dtype=float)
print(num)


# In[23]:


import numpy as np

arr = np.array([[1, 2, 3], [4, np.nam, 7], [8, 9, 10], [np.nan, np.nan, np.nan]])

mask = np.isnan(arr).any(axis=1)

arr = arr[~mask]

print(arr)


# In[32]:


import numpy as np
arr = np.array([[[1], [2], [3]], [[4], [5], [6]]])

print(arr.shape)

arr = np.squeeze(arr)

print(arr.shape)


# In[33]:


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]])

seq_to_find = np.array([1, 2, 3])

matches = np.where(np.all(arr == seq_to_find, axis=1))[0]

num_matches = len(matches)

print(num_matches)


# In[34]:


import numpy as np

arr = np.array([1, 2, 3, 4, 3, 2, 1, 3, 4, 4, 4])

unique_values, counts = np.unique(arr, return_counts=True)

most_frequent_index = np.argmax(counts)

most_frequent_value = unique_values[most_frequent_index]

print(most_frequent_value)


# In[38]:


import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([[4, 5, 6], [7, 8, 9]])

combined_arr = np.vstack((arr1, arr2))

print(combined_arr)


# In[39]:


import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

grid_arr1, grid_arr2 = np.meshgrid(arr1, arr2)

result = np.hstack((grid_arr1.reshape(-1,1), grid_arr2.reshape(-1,1)))

print(result)


# In[4]:


import numpy as np
h=np.logspace(2, 3, num=3)
print(h)


# In[9]:


h1=np.linspace(1,20,4)
print(h1)


# In[ ]:




