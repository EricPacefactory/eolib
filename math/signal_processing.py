#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:22:52 2018

@author: eo
"""


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes




# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions

def create_smoothing_kernel(num_samples = 3, window = "auto"):
    
    # Can't do anything to a null value
    if num_samples is None:
        return None
    
    # Make sure a valid window type is used (and print these out, since it would be mysterious otherwise)
    valid_window_types = ["auto", "rectangle", "triangle", "blackman"]
    if window not in valid_window_types:
        print("")
        print("Window supplied is not valid:", window)
        print("Must be one of the following:")
        for eachWindow in valid_window_types:
            print("  -", eachWindow)
            
            
    # Make sure number of samples is odd
    num_samples = force_to_odd(num_samples, force_downward = True, zero_maps_to = 1)
    
    if window is "auto":
        if num_samples < 7:
            window = "rectangle"
        elif num_samples < 21:
            window = "triangle"
        else:
            window = "blackman"
    
    if window is "rectangle":    
        return [1/num_samples] * num_samples
    
    if window is "triangle":
        
        # Scale factor is calculated so that the result matches a rectangle convolved with itself!
        count_seq = np.arange(0, num_samples)
        center_shift = (num_samples - 1) / 2
        scale_factor = (num_samples + 1) / 2
        triangle_seq = 1 - abs(count_seq - center_shift) / scale_factor
        norm_values = triangle_seq / sum(triangle_seq)
        
        return norm_values
    
    if window is "blackman":
    
        a0 = 0.42
        a1 = 0.5
        a2 = 0.08
        
        n = np.arange(0, num_samples)
        denom = num_samples - 1        
        
        blackman_seq = a0 - a1*np.cos(2*np.pi*n/denom) + a2*np.cos(4*np.pi*n/denom)
        norm_values = blackman_seq/sum(blackman_seq)
        
        return norm_values
        
# .....................................................................................................................    
    
def safe_smooth(data, kernel, same_size_out = True, data_type_out = np.float32, extrapolation_factor = 2,
                restore_first_value = False,
                restore_last_value = False):
    
    # If no kernel is provided, just return the data as-is
    if kernel is None:
        return data
    
    # Make sure the kernel isn't too big to use on this data
    kernel_size = len(kernel)
    data_len = len(data)
    if data_len < 2*kernel_size:
        return data
    
    # Figure out how many extrapolated values are needed (on both sides!)
    num_ext = int((kernel_size - 1)/2)
    
    # Calculate 'safe' extrapolated values to get back original data length
    smooth_data = np.convolve(data, kernel, mode = "valid")
    if same_size_out:
        smooth_out = list(smooth_data)
        for k in range(num_ext):
            new_front = smooth_out[0] + (smooth_out[0] - smooth_out[1])/extrapolation_factor
            new_back = smooth_out[-1] + (smooth_out[-1] - smooth_out[-2])/extrapolation_factor
            smooth_out = [new_front] + smooth_out + [new_back]
        
        # Restore the first/last values (with no smoothing) if desired
        if restore_first_value:
            smooth_out[0] = data[0]        
        if restore_last_value:
            smooth_out[-1] = data[-1]
            
        return data_type_out(smooth_out)
    
    return data_type_out(smooth_data)

# .....................................................................................................................

def central_diff(data, same_size_out = True, data_type_out = np.float32):
    
    # Special case when data is smaller than difference kernel (i.e. fewer than 3 points), return all zeros
    try: data[2]
    except IndexError: return data_type_out([0] * len(data))
    
    # Apply central difference using convolution (might be more efficient ways to do this!)
    kernel = [1/2, 0, -1/2]
    dd = np.convolve(data, kernel, mode='valid')
    
    # Add missing data end-points by using forward/backward differences
    if same_size_out:
        dd0 = data[1] - data[0]
        ddn = data[-1] - data[-2]    
        dd_out = [dd0] + list(dd) + [ddn]         
        return data_type_out(dd_out)
    
    return data_type_out(dd)

# .....................................................................................................................

# Function which takes any integer and forces it to the nearest (lower by default) odd number
def force_to_odd(input_value, force_downward = True, zero_maps_to = 0):
    
    # Can't do anything to a null value
    if input_value is None:
        return None
    
    # Special case, we don't want negative numbers...
    if input_value <= 0:
        return zero_maps_to
    
    # Determine whether if the number is even/odd, then shift it depending of force direction
    one_if_even_zero_if_odd = (1 - input_value % 2)
    if force_downward:
        return input_value - one_if_even_zero_if_odd
    else:
        return input_value + one_if_even_zero_if_odd

# .....................................................................................................................

# Function which maps all integers to odd values (not the same as forcing to an odd value!)
def map_ints_to_odds(input_value, lowest_valid_input = 0, one_maps_to = 1):
    
    # Can't do anything to a null value
    if input_value is None:
        return None
    
    # Convert to a null output if we don't pass the lowest valid input
    if input_value < lowest_valid_input:
        return None
    
    # Transformation converts:
    # 1 -> one_maps_to
    # 2 -> one_maps_to + 2
    # 3 -> one_maps_to + 4, etc.
    odd_conversion = 2*(input_value - 1)
    odd_conversion = max(0, odd_conversion)
    output_value = odd_conversion + one_maps_to
    
    # Return odd version of input
    return output_value

# .....................................................................................................................

# Function which takes in any integer and create a tuple of odd numbers
def odd_tuplify(input_value, lowest_valid_input = 0, one_maps_to = 1):
    
    odd_value = map_ints_to_odds(input_value, lowest_valid_input = lowest_valid_input, one_maps_to = one_maps_to)
    
    if odd_value is None:
        return None
    
    tuplified = (odd_value, odd_value)
    return tuplified

# .....................................................................................................................
    
# .....................................................................................................................
    
# .....................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Demo

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    kernel_size = 27
    rr = create_smoothing_kernel(kernel_size, "rectangle")
    tt = create_smoothing_kernel(kernel_size, "triangle") 
    bb = create_smoothing_kernel(kernel_size, "blackman")
    
    # Example of kernels
    plt.figure("Kernels")
    plt.plot(rr)
    plt.plot(tt)
    plt.plot(bb)


    # Filter comparison
    t = np.linspace(0, 1, 500)
    noise_mag = 2
    clean_signal = np.sin(2*np.pi*(11)*t) + np.cos(2*np.pi*(7)*t) + np.sin(2*np.pi*(3)*t)    
    noisy_signal = clean_signal + np.random.rand(len(clean_signal))*(noise_mag) - noise_mag/2
    min_val, max_val = np.floor(min(clean_signal)), np.ceil(max(clean_signal))
    
    plt.figure("Clean Signal")
    plt.plot(t, noisy_signal)
    plt.plot(t, clean_signal)
    plt.ylim([min_val, max_val])
    
    # Run filter comparisons
    n_offset = int((kernel_size - 1)/2)
    idx = t[n_offset:-n_offset]
    
    plt.figure("Rectangle Performance")
    plt.plot(idx, np.convolve(noisy_signal, rr, mode="valid"))
    plt.plot(t, clean_signal)
    plt.ylim([min_val, max_val])    
    
    plt.figure("Triangle Performance")
    plt.plot(idx, np.convolve(noisy_signal, tt, mode="valid"))
    plt.plot(t, clean_signal)
    plt.ylim([min_val, max_val])
    
    plt.figure("Blackman Performance")
    plt.plot(idx, np.convolve(noisy_signal, bb, mode="valid"))
    plt.plot(t, clean_signal)
    plt.ylim([min_val, max_val])

    plt.figure("Blackman Safe Performance")
    plt.plot(t, safe_smooth(noisy_signal, bb, extrapolation_factor= 2))
    plt.plot(t, clean_signal)
    plt.ylim([min_val, max_val])

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap

