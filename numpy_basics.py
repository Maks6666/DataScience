import numpy as np

# how to create 1D array from python list:

data = [1, 2, 3, 4, 5]

arr = np.array(data)
print(arr)
# to output array shape 
print(arr.shape)
# to output array data type
print(arr.dtype)
#  to output array dimension (1D)
print(arr.ndim)


# shape - the size of the array in the corresponding dimension (1D, 2D, 3D)

# 1D array (vector)
arr_1d = np.array([1, 2, 3, 4, 5])
# array size output 
print(arr_1d.shape) 

# 2D array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
# array size output 
print(arr_2d.shape) 

# 3D array (tensor - usually using in ML and DL)
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# array size output 
print(arr_3d.shape) 


# other way to create an array 

arr2 = np.array([1, 2, 3, 4, 5, 6])
print(arr2)
# we may ouput same parameters as earlier 
print(arr2.shape)
# to output array data type
print(arr2.dtype)
#  to output array dimension (1D)
print(arr2.ndim)



# dtype - datatype showing object

# Creating an Array of integers (data type is int32)
arr_int = np.array([1, 2, 3, 4], dtype='int32')
print(arr_int)

#  Creating an Array of floars (data type is  float64)
arr_float = np.array([1.1, 2.2, 3.3, 4.4], dtype='float64')
print(arr_float)


# changing data type in arrays

# Creating an Array of integers (data type is int32)
arr_int1 = np.array([1, 2, 3, 4], dtype='int32')
print(arr_int1)
# changing data type to float
arr_float1 = arr_int.astype('float64')
print(arr_float1)

# math operations with arrays

# Creating an Array of integers
arr_int2 = np.array([1, 2, 3, 4], dtype='int32')
print(arr_int2)

# Multiplying by a number changes the data type
arr_result = arr_int2 * 1.5  # data type is going to be changed to float64
print(arr_result)

#also way to change data type in array 
arr_int3 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
print(arr_int3)


# ranges using numpy (for i in range analog) - works alos with float data

arr4 = np.arange(0, 12, 1.2)
print(arr4)


# array "linspace" creates an array of evenly distributed values ​​in a given range (in this case - 5).
arr5 = np.linspace(0, 3, 5)
print(arr5)


# to create random array 
random_arr = np.random.random((5,))
print(random_arr)

# other wat to create random array 
random_arr1 = np.random.random_sample((5,))
print(random_arr1)

#  range of random numbers from a to b is b - a * np.random() + a, where b > a


random_arr2 = (10 - -5) * np.random.random_sample((5,)) - 5
print(random_arr2)



# generating random integers in a range
random_arr3 = np.random.randint(-5, 10, 10)
print(random_arr3)

# operations with arrays 

arr3 = np.array([1, 2, 3, 4, 5])
print(arr3)

# to calculate the square root of each element in the array 
arr3 = np.sqrt(arr3)
print(arr3)

# to calculate sinus of each element in the array 
arr3 = np.sin(arr3)
print(arr3)

# to calculate cosinus of each element in the array 
arr3 = np.cos(arr3)
print(arr3)

# to calculate logorifm of each element in the array 
arr3 = np.log(arr3)
print(arr3)


# to calculate exponent of each element in the array 
arr3 = np.exp(arr3)
print(arr3)


# basic math operations with arrays
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

c = a + b 
d = a - b
e = a * b 
f = a / b

print(c)
print(d)
print(e)
print(f)

arr4 = np.array([1, 2, 3, 4]) 
# multiply each array element by a number (in this case - 2)
arr4 = arr4 * 2
print(arr4)
# raise all elements to powers (in this case to 2)

arr5 = np.array([1, 2, 3, 4]) 
arr5 = arr5 ** 2
print(arr5)

# functions&methods

arr6 = np.array([1, 2, 3, 4, 5, 6]) 
# output maximun value from array
print(arr6.max())
# output minimum value from array
print(arr6.min())
# output medium value from array
print(arr6.mean())
# output array summa
print(arr6.sum())
# output standard deviation
print(arr6.std())
# output median
print(np.median(arr6))

# check all array values ​​for compliance with a condition
print(arr6 < 3)


# manipulation with arrays 

arr7 = np.array([1, 2, 3, 4, 5, 6]) 
# to insert value -2 into 2nd place (2nd index) in array 
arr7 = np.insert(arr7, 2, -2)
print(arr7)
# to delate any value from array (in this case 4th element)
arr7 = np.delete(arr7, 3)
print(arr7)
# to sort an array
arr7 = np.sort(arr7)
print(arr7)
# to concatenate two arrays
arr8 = np.array([0, 0, 0])
arr7 = np.concatenate((arr7, arr8))
print(arr7)
# to split an array in some parts (in this case - into 3)
arr8 = np.array_split(arr7, 3)
print(arr8)


# indexes 
arr9 = np.array([1, 2, 3, 4, 5, 6]) 
# to output element using it's index (in this case 4th index)
print(arr9[4])
# to output elements using it's index from a to b(in this case a is index 0, b is 3rd index)
print(arr9[0:3])
# to reverse array
print(arr9[::-1])
# to choose all elements with condition
print(arr[arr<2])
# to choose all elements with duoble condition (condition is "and")
print(arr[(arr<5) & (arr>2)])
# also to choose all elements with duoble condition (condition is "or")
print(arr[(arr<5) | (arr>2)])
# to replace values from a to b based on their indexes 
arr9[1:4] = 0 
print(arr9)

# matrix

# to create 2D matrix with floar data type
matrix = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
print(matrix)

# also way to create matrix
matrix = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=float)
print(matrix)

# to output shape, size and dimension(2D)
print(matrix.shape)
print(matrix.size)
print(matrix.ndim)

# to reshape matrix form 
print(matrix.reshape(1, 9))
# also to reshape matrix form - you mau change reshape parameters
matrix2 = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)], dtype=float)
print(matrix2.reshape(2, 6))

# to insert random values to matrix assigning ammount of columns and strings 
matrix3 = np.random.random((2, 2))
print(matrix3)

# to resize matrix:
matrix4 = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [3, 4, 5]])
# choose string and ammount of values you want to add in resized array  
matrix4 = np.resize(matrix4, (2, 3))
print(matrix4)

# function combination
matrix5 = np.arange(8).reshape(2, 4)
print(matrix5)

# special matrix

# zero matrix 
matrix_0 = np.zeros((2, 3))
print(matrix_0)

# ones matrix
matrix_ones = np.ones((2, 3))
print(matrix_ones)

# 5x5 identity matrix.
matrix_eye = np.eye((5))
print(matrix_eye)

# fill a matrix with one value
matrix_full = np.full((3, 3), 9)
print(matrix_full)

# to create empty matrix
matrix_empty = np.empty((3, 2))
print(matrix_empty)

# operations with matrix
matrix_1 = np.array([(1, 2, 3), (4, 5, 6)], dtype=float)


matrix_2 = np.array([(7, 8, 9), (10, 11, 12)], dtype=float)

res = matrix_1 + matrix_2
print(res)
res = matrix_1 - matrix_2
print(res)
res = matrix_1 * matrix_2
print(res)
res = matrix_1 / matrix_2
print(res)

# scalar product
matrix_1 = np.array([(1, 2), (3, 4)], dtype=float)


matrix_2 = np.array([(5, 6), (7, 8)], dtype=float)


res1 = matrix_1.dot(matrix_2)
print(res1)

# axis
# 0 axis is for columns (VERTICAL)
# 1 axis is for string (HORIZONTAL)


# IMPORTANT - in the context of DELETE functions, axis 0 interacts with ROWS and axis 1 with COLUMNS
matrix5 = np.array([(1, 2, 3), (3, 4, 3), (4, 3, 3)], dtype=float)
# to delate 1st string from matrix
matrix5 = np.delete(matrix5, 1, axis=1)
print(matrix5)
# to delate 1st columns from matrix
matrix5 = np.delete(matrix5, 1, axis=0)
print(matrix5)

# math functions for matrix

matrix_3 = np.array([[1, 2, 3],[2, 3, 4],[5, 6, 7]], dtype=float)
# to calculate the square root of each element in the array 
matrix_3 = np.sqrt(matrix_3)
print(matrix_3)

# to calculate sinus of each element in the array 
matrix_3 = np.sin(matrix_3)
print(matrix_3)

# to calculate cosinus of each element in the array 
matrix_3 = np.cos(matrix_3)
print(matrix_3)

# to calculate logorifm of each element in the array 
matrix_3 = np.log(matrix_3)
print(matrix_3)




matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
# Vertical sum (axis 0 corresponds to column axes)
sum_columns = np.sum(matrix, axis=0)
print("Summa by columns (axis=0):", sum_columns)


# Horizontal sum (axis 1 corresponds to row axes)
sum_rows = np.sum(matrix, axis=1)
print("Summa by strings (axis=1):", sum_rows)

# Vertical Max. value (axis 0 corresponds to column axes)
sum_columns = np.max(matrix, axis=0)
print("Max. value by columns (axis=0):", sum_columns)


# Horizontal Max. value (axis 1 corresponds to row axes)
sum_rows = np.max(matrix, axis=1)
print("Max. value by strings (axis=1):", sum_rows)

# concatenating by horizontal axis 
m_1 = np.array([[1, 2, 3], [4, 5, 6]])
print(m_1)

m_2 = np.array([[7, 8, 9], [10, 11, 12]])
print(m_2)

m_res = np.concatenate((m_1, m_2), axis=1)
print(m_res)

# concatenating by vertical axis 
m_1 = np.array([[1, 2, 3], [4, 5, 6]])
print(m_1)

m_2 = np.array([[7, 8, 9], [10, 11, 12]])
print(m_2)

m_res = np.concatenate((m_1, m_2), axis=0)
print(m_res)

# split by vertical axis into 2 arrays
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
esult = np.array_split(matrix, 2, axis=0)
print(esult)

# split by horizontal axis into 2 arrays
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
esult = np.array_split(matrix, 2, axis=1)
print(esult, "\n")

# indices
matrix = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
# in this case 1 means index of row, 2 means index of value
print(f"2nd element on 1st row - {matrix[1, 2]}")

# to output whole row (2nd in our case)
print(matrix[2])

# to output rows with condition
print(matrix[1:2, 0:2])

# to output rows with logic condition
print(matrix > 2)

# to replace columns and rows

print(matrix.T)

# to put matrix into 1D array
print(matrix.flatten())

# to find reverse matrix (if it exists) - a matrix whose multiplication by the original matrix gives the identity matrix
matrix = np.linalg.inv(matrix)
print(matrix)

# to find matrix trace - the sum of the elements located on its main diagonal
matrix = np.trace(matrix)
print(matrix)

# to find matrix rank - maximum number of linearly independent rows (or columns) in a matrix
matrix = np.linalg.matrix_rank(matrix)
print(matrix)








