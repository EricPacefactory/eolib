#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:39:09 2018

@author: eo
"""


import numpy as np

m = 2

h = 2
x_0 = 0

x = [x_0 - k*h for k in range(1+m)]
x = np.array(x)

b = [0] + [k*(x_0**(k-1)) for k in range(1,1+m)]


V = np.vander(x, increasing=True)
v_inv = np.linalg.inv(V)



w = h*np.dot(b, v_inv)
print(w)

#%%

num_coeffs = 4
backward_difference = True
newest_last = True
style = "central"

def difference_coeffs(num_coeffs, style = "backward", newest_last = True):
    
    # Generates a set of co-effcients for performing multi-polint numerical differentiation
    # The coeffs. are used by performing a dot-product with the desired data set. For example:
    #
    # Given data = [yn, ..., y2, y1, y0]
    # (where y0 is the 'newest' data point, typically due to gathering data using a .append() function)
    #
    # dy/dx = sum(coeffs.*data)    
    
    if style.strip().lower() == "central":        
        x = np.arange(num_coeffs)
        x_0 = np.mean(x)
    
    elif style.strip().lower() == "backward":        
        x = np.arange(num_coeffs)
        x_0 = x[-1]
        
    elif style.strip().lower() == "forward":        
        x = np.arange(num_coeffs)
        x_0 = x[0]
        
    else:
        raise AttributeError("Style must be one of: central, forward, backward")
    
    # Build right-hand-side values
    rhs = [0] + [k*(x_0**(k-1)) for k in range(1, num_coeffs)]
    
    # Build 'powers of x' matrix
    x_mat = np.vander(x, increasing=True)
    x_mat_inv = np.linalg.inv(x_mat)
    
    # Calculate solution to (1/dx) * X * coeffs = rhs
    coeffs = np.dot(rhs, x_mat_inv)
    
    if newest_last:
        coeffs = np.flip(coeffs, axis=0)

    return coeffs



#print(generate_forwardbackward_difference_coeffs(num_coeffs = 3))


#%% Demo


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    num_plot = 50
    inner_term = 2*np.pi*1.5/num_plot
    demo_x_data = [k for k in range(num_plot)]
    demo_y_data = [np.sin(x*inner_term) for x in demo_x_data]
    
    plt.figure("Original Signal")
    plt.plot(demo_x_data, demo_y_data)
    
    
    true_dy_data = [inner_term*np.cos(x*inner_term) for x in demo_x_data]
        
    coeff_check = [2, 3, 5]
    
    back_diffs = {eachNum: difference_coeffs(eachNum, "backward") for eachNum in coeff_check}
    
    
    dydx = {eachNum: [] for eachNum in coeff_check}
    x_points = {eachNum: [] for eachNum in coeff_check}

    
    
    for k in range(num_plot):        
        for numCoeffs, eachDiff in back_diffs.items():            
            if k >= numCoeffs:
                first_index = k - numCoeffs
                last_index = k
                y_data = demo_y_data[first_index:last_index]
                
                dydx[numCoeffs].append( np.dot(eachDiff, y_data) )
                x_points[numCoeffs].append(last_index)
        
    
    plt.figure("Backward Difference")
    plt.plot(demo_x_data, true_dy_data)
    
    for eachNum in coeff_check:
        
        x_data_plot = x_points[eachNum]
        y_data_plot = dydx[eachNum]
        
        plt.plot(x_data_plot, y_data_plot)
    
    plt.legend(["True"] + [eachKey for eachKey in coeff_check])
    
    
    
    
    
    pass