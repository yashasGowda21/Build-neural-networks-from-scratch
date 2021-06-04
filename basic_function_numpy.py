import math


# GRADED FUNCTION: basic_sigmoid

def basic_sigmoid(x):
    """
    Compute sigmoid of x.
    
    Using math library

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    # (≈ 1 line of code)
    # s = 
    # YOUR CODE STARTS HERE
    # sigmoid : x = 1/1+e^(-t)
    s = 1/(1+math.exp(-x))
    # YOUR CODE ENDS HERE
    
    return s
print("basic sigmoid for single value x")
print(basic_sigmoid(2))

### One reason why we use "numpy" instead of "math" in Deep Learning ###

x = [1, 2, 3] # x becomes a python list object
try:
    basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
except Exception as e:
    print(e)

print("Numpy ")
import numpy as np

# example of np.exp
t_x = np.array([1, 2, 3])
print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

# GRADED FUNCTION: sigmoid

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    # (≈ 1 line of code)
    # s = 
    # YOUR CODE STARTS HERE
    
    s = 1/(1+np.exp(-x))
    # YOUR CODE ENDS HERE
    
    return s



# GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    #(≈ 2 lines of code)
    # s = 
    # ds = 
    # YOUR CODE STARTS HERE
    s = sigmoid(x)
    ds = s*(1-s)
    
    # YOUR CODE ENDS HERE
    
    return ds

def image2vector(image):
    """Reshape 3D image array to 1D image array.

    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    # (≈ 1 line of code)
    # v =
    # YOUR CODE STARTS HERE
    
    row = image.shape[0]
    col = image.shape[1]
    depth = image.shape[2]
    # In other words, you "unroll", or reshape, the 3D array into a 1D vector.
    v = image.reshape(row*col*depth,1)
    
    
    # YOUR CODE ENDS HERE
    
    return v
    

def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    #(≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    # x_norm =
    # Divide x by its norm.
    # x =
    # YOUR CODE STARTS HERE
    
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    
    x = x/x_norm
    
    
    # YOUR CODE ENDS HERE

    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    # YOUR CODE STARTS HERE
    #(≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True) 
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum  
    
    # YOUR CODE ENDS HERE
    
    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    #(≈ 1 line of code)
    # loss = 
    # YOUR CODE STARTS HERE
    
    loss = np.sum(abs(y-yhat))
    # YOUR CODE ENDS HERE
    
    return loss

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    #(≈ 1 line of code)
    # loss = ...
    # YOUR CODE STARTS HERE
    
    loss = np.sum(np.dot(y-yhat,y-yhat))
    # YOUR CODE ENDS HERE
    
    return loss