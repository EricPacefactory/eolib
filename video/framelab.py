#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 10:37:37 2018

@author: eo
"""

#%% Imports

import cv2
import numpy as np

from functools import partial
from collections import deque


#%% Classes

class FrameLab:
    
    
    def __init__(self, name):
        
        self._name = name
        
        self._historyWHC = [(None, None, None)]
        
        self._resources = {}
        self._res_compare = {}
        self._framedeck_list = []
        
        
        self._function_sequence = []
        self._function_index = 0
        
        self._changed = False
        
        self._store_intermediates = False
        self._intermediates = []
        self._frame_storage = {}
        self._frame_count = 0
        self._sample_rate_frames = 1
        
        pass
    
    # .................................................................................................................

    def __repr__(self):
        
        print("")
        outStringList = []
        outStringList.append(" ".join(["FrameLab:", self._name]))
        for eachFunc in self._function_sequence:
            
            if type(eachFunc) is partial:                
                funcString = "".join(["  ", eachFunc.func.__name__, "()"])
            else:
                funcString = "".join(["  ", eachFunc.__name__, "()"])
                
            outStringList.append(funcString)
                
        return "\n".join(outStringList)
    
    # .................................................................................................................
    
    def fetch_dimensions(self, *args):
        
        # Create lut for convenience
        dim_ref = {"width": 0, "height": 1, "channels": 2, "depth": 2}
        
        # Get all the dimensions requests
        dimension_list = [self._historyWHC[-1][dim_ref[eachArg.lower().strip()]] for eachArg in args]
        
        # Reduce output to a single number if only one dimension was requested
        if len(dimension_list) == 1:
            dimension_list = dimension_list[0]
        
        return dimension_list
    
    # .................................................................................................................
    
    def set_input(self, *, frame_dimensions_WHC = None, 
                  image_reference = None, 
                  video_reference = None,
                  frame_lab = None):
        
        # Some feedback, in case this function is called at the wrong time!
        if len(self._historyWHC) > 1:
            print("")
            print("WARNING:")
            print("  set_input() should be called before any frame processing functions!")
            print("  Frame sizing history will be lost")
        
        if frame_lab is not None:
            width, height, channels = frame_lab.fetch_dimensions("width", "height", "channels")
            self._historyWHC = [(width, height, channels)]
            return
        
        if image_reference is not None:
            frame_HWC = image_reference.shape
            width = frame_HWC[1]
            height = frame_HWC[0]
            channels = frame_HWC[2] if len(frame_HWC) > 2 else 1            
            self._historyWHC = [(width, height, channels)]
            return
        
        if video_reference is not None:
            width = int(video_reference.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_reference.get(cv2.CAP_PROP_FRAME_HEIGHT))
            channels = 3
            self._historyWHC = [(width, height, channels)]
            return
        
        if frame_dimensions_WHC is not None:
            self._historyWHC = [tuple(frame_dimensions_WHC)] 
            return
        
        raise AttributeError("Must provide some input for frame sizing")
    
    # .................................................................................................................
    
    def set_sample_rate(self, sample_rate_frames):        
        self._sample_rate_frames = sample_rate_frames
    
    # .................................................................................................................
    
    def enable_intermediates(self, enable = True):
        self._store_intermediates = enable
    
    # .................................................................................................................
    
    def process(self, frame):
        
        # Update frame processing count, mainly for subsampling
        self._frame_count += 1
        
        # Repeatedly apply each function in the func. sequence on the result from the previous function
        prevFrame = frame.copy()
        for idx, eachFunction in enumerate(self._function_sequence):
            
            self._function_index = idx
            prevFrame = eachFunction(prevFrame)
            
            if self._store_intermediates:
                try:
                    self._intermediates[idx] = prevFrame.copy()
                except IndexError:
                    self._intermediates.append(prevFrame.copy())
                
        # Indicate that the processor output has changed
        self._changed = True
        
        return prevFrame
    
    # .................................................................................................................
    
    def store_frame(self, frame_name):
        
        # Check that the storage key doesn't already exist
        if frame_name in self._frame_storage:
            raise KeyError("Frame name already exists!")
        
        def store_frame_(inFrame, store_key):            
            # Store incoming frame with no modifications and then pass it through
            self._frame_storage[store_key] = inFrame.copy()
            return inFrame
        
        store_func = partial(store_frame_, store_key = frame_name)
        
        self._seq_func(store_func)
    
    # .................................................................................................................
    
    def fetch_frame(self, frame_name):
        
        if frame_name not in self._frame_storage:
            raise KeyError("Frame '{}' not found!".format(frame_name))
        
        return self._frame_storage[frame_name]
    
    # .................................................................................................................
    
    def fetch_setting(self, setting_name):
        
        if setting_name not in self._resources:
            raise KeyError("Setting '{}' not found!".format(setting_name))
        
        return self._resources[setting_name]
    
    # .................................................................................................................
    
    def modify(self, setting_name, new_value):
        
        if setting_name not in self._resources:
            raise KeyError("Setting '{}' not found!".format(setting_name))
            
        self._resources[setting_name] = new_value
        
    # .................................................................................................................
    
    def modify_from_dict(self, dictionary, select_key = None):
        
        # Use either the full dictionary or just the selected key
        use_dict = dictionary if select_key is None else {select_key: dictionary[select_key]}
         
        # Update each resource entry, based on dictionary keys/values
        for eachKey, eachValue in use_dict.items():
            
            if eachKey not in self._resources:
                raise KeyError("Setting '{}' not found!".format(eachKey))   
                
            self._resources[eachKey] = eachValue
    
    # .................................................................................................................
    
    def blame(self):
        
        pass
    
    # .................................................................................................................
    
    def collage(self):
        
        pass
    
    # .................................................................................................................
    
    def custom_function(self, custom_func, verbose = True):
        
        # Try to get the existing dimensions
        in_width, in_height, in_channels = self.fetch_dimensions("width", "height", "channels")
        
        # Use defaults if dimensions aren't available
        in_width = in_width if in_width is not None else 100
        in_height = in_height if in_height is not None else 100
        in_channels = in_channels if in_channels is not None else 3
                
        # Figure out what this custom function does to image dimensions by passing a dummy frame through it
        dummy_frame = np.zeros((in_height, in_width, in_channels), dtype=np.uint8)
        try:
            outFrame = custom_func(dummy_frame)
        except Exception as e:
            print("")
            print(self._name)
            print("Error running custom function:", custom_func.__name__)
            print("Tried inputting frame of dimensions:")
            print("WHC:", " x ".join([str(in_width), str(in_height), str(in_channels)]))
            print("")
            raise e
        
        # Update dimension record to account for any resizing this function performs
        out_dimensions = outFrame.shape
        out_height, out_width = out_dimensions[0:2]
        out_channels = out_dimensions[2] if len(out_dimensions) > 2 else 1
        self._historyWHC.append((out_width, out_height, out_channels))
        
        # Some feedback
        if verbose:
            print("")
            print("Custom function:", custom_func.__name__)
            print("  Input size  (WHC):", " x ".join([str(in_width), str(in_height), str(in_channels)]))
            print("  Output size (WHC):", " x ".join([str(out_width), str(out_height), str(out_channels)]))
        
        return self._seq_func(custom_func)
    
    # .................................................................................................................
    
    def grayscale(self, *, output_bgr = False):
        
        # Update frame dimensions if a conversion to 1 channel occurs (i.e. no conversion back to bgr)
        if not output_bgr:
            width, height = self.fetch_dimensions("width", "height")
            self._historyWHC.append((width, height, 1))
        
        # Use different function, depending on whether a 3/1 channel output is desired
        if output_bgr:
            
            def grayscale_(inFrame):                
                return cv2.cvtColor(cv2.cvtColor(inFrame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        else:
            
            def grayscale_(inFrame):                
                return cv2.cvtColor(inFrame, cv2.COLOR_BGR2GRAY)
        
        gray_func = grayscale_
        
        return self._seq_func(gray_func)
    
    # .................................................................................................................
    
    def norm(self, order = np.inf, output_bgr = False):
        
        # Get frame sizing
        width, height, channels = self.fetch_dimensions("width", "height", "channels")
        
        # Handle incorrect input (single channel) by passing the incoming frame through
        if channels == 1:            
            # Some feedback
            print("")
            print("Norm operation skipped, since input is already single channel!")
            
            def skip_norm(inFrame): return inFrame        
            return self._seq_func(skip_norm)
        
        # Store input control data
        order_key, _ = self._store_settings(order)
        
        # Add norming operation (reduce to single channel) to dimension record  
        self._historyWHC.append((width, height, 1))
        
        # Use different function, depending on whether a 3/1 channel output is desired
        if output_bgr:
            
            def norm_(inFrame, res_key):
                # Numpy: np.linalg.norm(x, ord, axis)
                return cv2.cvtColor(np.uint8(np.linalg.norm(inFrame, ord=self._resources[res_key], axis=2)), 
                                    cv2.COLOR_GRAY2BGR)
        else:
            
            def norm_(inFrame, res_key):
                return np.uint8(np.linalg.norm(inFrame, ord=self._resources[res_key], axis=2))
        
        norm_func = partial(norm_, res_key = order_key)
        
        return self._seq_func(norm_func)
    
    # .................................................................................................................
    
    def scale_channels(self, *, red = 1.0, green = 1.0, blue = 1.0):
        
        # Store input control data
        red_key, _ = self._store_settings(red)
        green_key, _ = self._store_settings(green)
        blue_key, _ = self._store_settings(blue)
        
        def scale_channels_(inFrame, red_res_key, green_res_key, blue_res_key):
            
            # Get multipliers for convenience
            red_val = self._resources[red_res_key]
            green_val = self._resources[green_res_key]
            blue_val = self._resources[blue_res_key]
            
            # Perform channel multiplications
            float_frame = np.float16(inFrame) * np.float16((blue_val, green_val, red_val))
            
            return np.uint8(np.round(float_frame))
        
        scale_func = partial(scale_channels_, red_res_key = red_key, green_res_key = green_key, blue_res_key = blue_key)
        
        return self._seq_func(scale_func)
    
    # .................................................................................................................
    
    def invert(self):
        return self._seq_func(cv2.bitwise_not)
    
    # .................................................................................................................
    
    def mask_from_image(self, mask_image):
        
        # Store input control data
        original_mask_key, _ = self._store_settings(mask_image)
        
        def mask_from_image_(inFrame, orig_key):
            
            # Compare shape of incoming frame to original mask frame, if they don't match we need to resize the mask
            frame_shape = inFrame.shape
            mask_shape = self._resources[orig_key].shape
            if frame_shape != mask_shape:
                frameWH = (frame_shape[1], frame_shape[0])
                new_resized_mask = cv2.resize(self._resources[orig_key], dsize = frameWH)
                self._resources[orig_key] = new_resized_mask.copy()
                
            # OpenCV: cv2.bitwise_and(src, src2)
            return cv2.bitwise_and(inFrame, self._resources[orig_key])
        
        mask_func = partial(mask_from_image_,
                            orig_key = original_mask_key)
        
        return self._seq_func(mask_func)
    
    # .................................................................................................................
    
    def mask_from_polygons(self, polygon_list, invert_mask = False, normalized_input = False):
        
        # Store input control data
        poly_key, poly_val = self._store_settings(None)#polygon_list)
        inv_key, inv_val = self._store_settings(None)
        mask_key, _ = self._store_settings(None)
        
        # Store copies of values that need to be checked for changes
        self._store_comparison(poly_key)
        self._store_comparison(inv_key)        
        
        # Quick cheat to force mask drawing on first run by filling in values after storing initial comparison values
        self._resources[poly_key] = polygon_list
        self._resources[inv_key] = invert_mask
        
        def mask_from_polygons_(inFrame, polygon_key, invert_key, mask_image_key):
            
            # Check if the mask needs to be re-generated
            poly_changed = self._compare_changes_arrays(polygon_key)
            inv_changed = self._compare_changes(invert_key)
            if poly_changed or inv_changed:  
                
                # For debugging
                #print("MASK CHANGED")
                
                # Create empty frame for masking
                frame_shape = inFrame.shape
                frameWH = np.array((frame_shape[1], frame_shape[0]))
                mask_bg_fill = 0 if self._resources[invert_key] else 255
                mask_fg_fill = 255 if self._resources[invert_key] else 0
                new_mask_image = np.full(frame_shape, mask_bg_fill, dtype=np.uint8)
                
                # Fill mask region + draw outline around it, since there appears to be aliasing issues
                for each_polygon in self._resources[polygon_key]:                    
                    poly_px = np.int32(frameWH * each_polygon) if normalized_input else np.int32(each_polygon)
                    cv2.fillPoly(new_mask_image, [poly_px], mask_fg_fill)
                    cv2.polylines(new_mask_image, [poly_px], True, mask_fg_fill, 2)
                
                # Store new mask image so we can re-use it
                self._resources[mask_image_key] = new_mask_image.copy()
            
            # OpenCV: cv2.bitwise_and(src, src2)
            return cv2.bitwise_and(inFrame, self._resources[mask_image_key])
        
        mask_func = partial(mask_from_polygons_, 
                            polygon_key = poly_key, 
                            invert_key = inv_key, 
                            mask_image_key = mask_key)
        
        return self._seq_func(mask_func)
    
    # .................................................................................................................
    
    def threshold(self, threshold = 127):
        
        # Store input control data
        thresh_key, _ = self._store_settings(threshold)
        
        # Function for getting the 1-index return argument (frame data) from the OpenCV function
        def threshold_(inFrame, res_key):            
            # OpenCV: cv2.threshold(src, thresh, maxval, type) returns (threshold_value, frame_data)
            return cv2.threshold(inFrame, 
                                 thresh=self._resources[res_key], 
                                 maxval=255, 
                                 type=cv2.THRESH_BINARY)[1]
        
        thresh_func = partial(threshold_, res_key = thresh_key)
        
        return self._seq_func(thresh_func)
    
    # .................................................................................................................
    
    def threshmask(self, threshold):
        
        # Store input control data
        thresh_key, _ = self._store_settings(threshold)
        
        def threshmask_(inFrame, res_key):
            
            # OpenCV: cv2.threshold(src, thresh, maxval, type) returns (threshold_value, frame_data)
            threshold_image = cv2.threshold(inFrame, 
                                            thresh=self._resources[res_key], 
                                            maxval=255, 
                                            type=cv2.THRESH_BINARY)[1]          
            
            # Apply threshold as a maskto the original image
            return cv2.bitwise_and(inFrame, threshold_image)
        
        threshmask_func = partial(threshmask_, res_key = thresh_key)
        
        return self._seq_func(threshmask_func)
    
    # .................................................................................................................
    
    def convert_color(self, color_space_code):
        
        # Store input control data
        cspace_key, _ = self._store_settings(color_space_code)
        
        def convert_color_(inFrame, res_key):            
            # OpenCV: cv2.cvtColor(src, code)
            return cv2.cvtColor(inFrame, self._resources[res_key])
        
        cspace_func = partial(convert_color_, res_key = cspace_key)
        
        return self._seq_func(cspace_func)
    
    # .................................................................................................................
    
    def channel_filter(self, lower_ch1, lower_ch2, lower_ch3, upper_ch1, upper_ch2, upper_ch3):
        
        # Store input control data
        lch1_key, _ = self._store_settings(lower_ch1)
        lch2_key, _ = self._store_settings(lower_ch2)
        lch3_key, _ = self._store_settings(lower_ch3)
        uch1_key, _ = self._store_settings(upper_ch1)
        uch2_key, _ = self._store_settings(upper_ch2)
        uch3_key, _ = self._store_settings(upper_ch3)
        
        def channel_filter_(inFrame, low_ch1_key, low_ch2_key, low_ch3_key, upp_ch1_key, upp_ch2_key, upp_ch3_key):
            
            lower_bound = (self._resources[low_ch1_key], self._resources[low_ch2_key], self._resources[low_ch3_key])
            upper_bound = (self._resources[upp_ch1_key], self._resources[upp_ch2_key], self._resources[upp_ch3_key])
            
            # OpenCV: cv2.inRange(src, lowerb, upperb)
            return cv2.inRange(inFrame, lower_bound, upper_bound)
        
        chfilter_func = partial(channel_filter_, 
                                low_ch1_key = lch1_key,
                                low_ch2_key = lch2_key,
                                low_ch3_key = lch3_key,
                                upp_ch1_key = uch1_key,
                                upp_ch2_key = uch2_key,
                                upp_ch3_key = uch3_key)
        
        return self._seq_func(chfilter_func)
    
    # .................................................................................................................
    
    def morphology(self, kernel_size = (3, 3), kernel_shape = cv2.MORPH_RECT, operation = cv2.MORPH_CLOSE):
        
        # Store input control data
        ksize_key, ksize_val = self._store_settings(kernel_size)
        kshape_key, kshape_val = self._store_settings(kernel_shape)
        op_key, _ = self._store_settings(operation)
        
        # Store copies of values that need to be checked for changes
        self._store_comparison(ksize_key)
        self._store_comparison(kshape_key)
        
        # Build the initial kernel
        kernel_val = cv2.getStructuringElement(kshape_val, ksize_val)
        kernel_key, _ = self._store_settings(kernel_val)
        
        def morphology_(inFrame, kernel_size_key, kernel_shape_key, operation_key, kernel_key):
            
            # Check if the kernel needs to be re-generated
            ksize_changed = self._compare_changes(kernel_size_key)
            kshape_changed = self._compare_changes(kernel_shape_key)
            if ksize_changed or kshape_changed:
                kernel_val = cv2.getStructuringElement(self._resources[kernel_shape_key],
                                                       self._resources[kernel_size_key])
                self._resources[kernel_key] = kernel_val
            
            # OpenCV: cv2.morphologyEx(src, op, kernel)
            return cv2.morphologyEx(inFrame, self._resources[operation_key], self._resources[kernel_key])
        
        morph_func = partial(morphology_, 
                             kernel_size_key = ksize_key,
                             kernel_shape_key = kshape_key,
                             operation_key = op_key,
                             kernel_key = kernel_key)
        
        return self._seq_func(morph_func)
    
    # .................................................................................................................
    
    def blur(self, kernel_size = (3, 3), kernel_sigma = (0, 0)):
        
        # Store input control data
        ksize_key, _ = self._store_settings(kernel_size)
        ksigma_key, _ = self._store_settings(kernel_sigma)
        
        def blur_(inFrame, kernel_size_key, kernel_sigma_key):   
            # OpenCV: cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
            return cv2.GaussianBlur(src = inFrame, 
                                    ksize = self._resources[kernel_size_key], 
                                    sigmaX = self._resources[kernel_sigma_key][0], 
                                    sigmaY = self._resources[kernel_sigma_key][1],
                                    borderType = cv2.BORDER_DEFAULT)
        
        blur_func = partial(blur_, kernel_size_key = ksize_key, kernel_sigma_key = ksigma_key)
        
        return self._seq_func(blur_func)
    
    # .................................................................................................................
    
    def fast_blur(self, kernel_size = (3, 3)):
        
        # Store input control data
        ksize_key, _ = self._store_settings(kernel_size)
        
        def fast_blur_(inFrame, kernel_size_key):
            # OpenCV: cv2.blur(src, ksize)
            return cv2.blur(src = inFrame,
                            ksize = self._resources[kernel_size_key])
            
        fast_blur_func = partial(fast_blur_, kernel_size_key = ksize_key)
        
        return self._seq_func(fast_blur_func)
    
    # .................................................................................................................
    
    def median_blur(self, kernel_size = 3):
        
        # Store input control data
        ksize_key, _ = self._store_settings(kernel_size)
        
        def median_blur_(inFrame, kernel_size_key):            
            # OpenCV: cv2.medianBlur(src, ksize)
            return cv2.medianBlur(src = inFrame, 
                                  ksize = self._resources[kernel_size_key])
        
        median_blur_func = partial(median_blur_, kernel_size_key = ksize_key)
        
        return self._seq_func(median_blur_func)
    
    # .................................................................................................................
    
    def crop_tl_br(self, top_left = (0.0, 0.0), bottom_right = (1.0, 1.0), normalized_inputs = True):
        
        # Store input control data
        tl_key, tl_val = self._store_settings(top_left)
        br_key, br_val = self._store_settings(bottom_right)
        
        # Get frame sizing
        width, height, channels = self.fetch_dimensions("width", "height", "channels")
        width_scale = np.array(width) - 1
        height_scale = np.array(height) - 1        
        
        # Add resizing to dimension record
        new_width = np.int32(np.round(width_scale*(br_val[0] - tl_val[0])))
        new_height = np.int32(np.round(height_scale*(br_val[1] - tl_val[1])))
        self._historyWHC.append((new_width, new_height, channels))
        
        def crop_(inFrame, top_left_key, bot_right_key):
            
            # Get frame sizing
            height_scale, width_scale = inFrame.shape[0:2] - np.int32((1,1))
            
            # Retrieve settings
            top_left_x, top_left_y = self._resources[top_left_key]
            bot_right_x, bot_right_y = self._resources[bot_right_key]
            
            tl_x = np.int32(np.round(width_scale*top_left_x))
            tl_y = np.int32(np.round(height_scale*top_left_y))
            br_x = np.int32(np.round(width_scale*bot_right_x))
            br_y = np.int32(np.round(height_scale*bot_right_y))
            
            return inFrame[tl_y:br_y, tl_x:br_x]
        
        crop_func = partial(crop_, top_left_key = tl_key, bot_right_key = br_key)
        
        return self._seq_func(crop_func)
    
    # .................................................................................................................
    
    def resize_WH(self, new_WH, auto_disable = True):
        
        # Store input control data
        size_key, new_WH_val = self._store_settings(new_WH)
        
        # Catch floating point numbers, which will cause errors later. Caller should handle typecasting
        if type(new_WH_val[0]) is float or type(new_WH_val[1]) is float:
            raise TypeError("Cannot use floating point values for resize_WH, must use integers!")
        
        # Get frame sizing
        width, height, channels = self.fetch_dimensions("width", "height", "channels")
        
        # Automatically skip resizing if the frame dimensions match prior dimensions
        # Disabling option is helpful in case resize is handled dynamically (i.e. user can modify at runtime)
        if auto_disable:
            if new_WH_val[0] == width and new_WH_val[1] == height:
                
                # Some feedback
                print("")
                print("No resizing performed, since input already matches target dimensions!")
                
                # Define pass-through function (not a lambda, since the function name seems helpful for feedback)
                def skip_resize_WH(inFrame): return inFrame 
                return self._seq_func(skip_resize_WH)
            
        # Add resizing to dimension record  
        self._historyWHC.append((new_WH_val[0], new_WH_val[1], channels))
        
        # Define resize function, with modifiable resizing dimensions
        def resize_(inFrame, res_key):
            return cv2.resize(inFrame, dsize=self._resources[res_key])
        
        resize_func = partial(resize_, res_key = size_key)
        
        return self._seq_func(resize_func)
    
    # .................................................................................................................
    
    def resize_XY(self, new_XY, auto_disable = True):
        
        # Store input control data
        scale_key, new_XY_val = self._store_settings(new_XY)
        
        # Get frame sizing in pixels
        width, height, channels = self.fetch_dimensions("width", "height", "channels")
        new_width = int(np.round(width*new_XY_val[0]))
        new_height = int(np.round(height*new_XY_val[1]))
        
        # Automatically skip resizing if the frame dimensions match prior dimensions
        # Disabling option is helpful in case resize is handled dynamically (i.e. user can modify at runtime)
        if auto_disable:
            if new_width == width and new_height == height:
                
                # Some feedback
                print("")
                print("No resizing performed, since resized dimensions match existing frame dimensions!")
                
                # Define pass-through function (not a lambda, since the function name seems helpful for feedback)
                def skip_resize_XY(inFrame): return inFrame 
                return self._seq_func(skip_resize_XY)
            
        # Add resizing to dimension record
        self._historyWHC.append((new_width, new_height, channels))
        
        # Define resize function, with modifiable scaling factors
        def resize_(inFrame, res_key):
            return cv2.resize(inFrame, dsize=None, fx = self._resources[res_key][0], fy = self._resources[res_key][1])
        
        resize_func = partial(resize_, res_key = scale_key)
        
        return self._seq_func(resize_func)
    
    # .................................................................................................................
    
    def gamma(self, gamma_factor):
        
        # Allocate space for gamma factor and the corresponding lut
        gamma_key, _ = self._store_settings(None)
        table_key, _ = self._store_settings(None)
        
        # Store a copy of the gamma value so we can check for comparison
        self._store_comparison(gamma_key)
        
        # Quick cheat to force table creation on first run by filling in gamma after storing initial comparison values
        self._resources[gamma_key] = gamma_factor
        
        def gamma_(inFrame, gamma_res_key, table_res_key):
            
            # Update the look-up table if the gamma value changes
            gamma_changed = self._compare_changes(gamma_res_key)
            if gamma_changed:
                gamma_cor = 1/self._resources[gamma_res_key]
                self._resources[table_res_key] = np.uint8(np.round(255*np.power(np.linspace(0, 1, 256), gamma_cor)))
                
            return cv2.LUT(inFrame, self._resources[table_res_key])
        
        gamma_func = partial(gamma_, gamma_res_key = gamma_key, table_res_key = table_key)
        
        return self._seq_func(gamma_func)
    
    # .................................................................................................................
    
    def pixelate(self, scale_factor):
        
        # Store input scaling factor
        key_name, _ = self._store_settings(scale_factor)
        
        def pixelate_(inFrame, res_key):
            
            # Get scaling factor
            scale = 1/self._resources[res_key]
            inputWH = inFrame.shape[1::-1]
            
            # Downscale then re-upscale to create pixelated look
            downscale = cv2.resize(inFrame, dsize=None, fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
            upscale = cv2.resize(downscale, dsize=inputWH, interpolation = cv2.INTER_NEAREST)
            
            return upscale
        
        pixelate_func = partial(pixelate_, res_key = key_name)
        
        return self._seq_func(pixelate_func)
    
    # .................................................................................................................
    
    def canny(self, low_threshold = 50, high_threshold = 200, kernel_size = 3, output_bgr = False):
        
        # Store input control data
        lth_key, _ = self._store_settings(low_threshold)
        hth_key, _ = self._store_settings(high_threshold)
        ksize_key, _ = self._store_settings(kernel_size)
        
        if output_bgr:
            
            def canny_(inFrame, low_threshold_key, high_threshold_key, kernel_size_key):
                # OpenCV: cv2.Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient)
                return cv2.cvtColor(cv2.Canny(image = inFrame, 
                                              threshold1 = self._resources[low_threshold_key],
                                              threshold2 = self._resources[high_threshold_key],
                                              apertureSize = self._resources[kernel_size_key]), cv2.COLOR_GRAY2BGR)
        else:
            
            def canny_(inFrame, low_threshold_key, high_threshold_key, kernel_size_key):
                # OpenCV: cv2.Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient)
                return cv2.Canny(image = inFrame, 
                                 threshold1 = self._resources[low_threshold_key],
                                 threshold2 = self._resources[high_threshold_key],
                                 apertureSize = self._resources[kernel_size_key])
                             
        canny_func = partial(canny_, 
                             low_threshold_key = lth_key, 
                             high_threshold_key = hth_key, 
                             kernel_size_key = ksize_key)
        
        return self._seq_func(canny_func)
    
    # .................................................................................................................
    
    def absdiff(self, image):
        
        # Store input image data
        key_name, _ = self._store_settings(image)
        
        def absdiff_(inFrame, res_key):
            return cv2.absdiff(inFrame, self._resources[res_key])
        
        diff_func = partial(absdiff_, res_key = key_name)
        
        return self._seq_func(diff_func)
    
    # .................................................................................................................
    
    def absdiff_self(self, backstep = 1, max_backstep = None):
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for getting an absolute difference 
        def absdiff_self_(inFrame, deck_index, stepsize_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get backward-step frame
            prev_frame = self._read_from_deck(deck_index, self._resources[stepsize_key])
            
            # OpenCV: cv2.absdiff(src1, src2)
            return cv2.absdiff(inFrame, prev_frame)
        
        diff_func = partial(absdiff_self_, deck_index = deck_idx, stepsize_key = backstep_key)
        
        return self._seq_func(diff_func)
    
    # .................................................................................................................
    
    def sum(self, image, output_dtype = np.uint8):
        
        # Store input image data
        key_name, _ = self._store_settings(image)
        type_key, _ = self._store_settings(output_dtype)
        
        def sum_(inFrame, image_key, out_type_key):
            return np.sum([inFrame, self._resources[image_key]], dtype = self._resources[out_type_key])
        
        sum_func = partial(sum_, image_key = key_name, out_type_key = type_key)
        
        return self._seq_func(sum_func)
    
    # .................................................................................................................
    
    def sum_self(self, num_to_sum = 1, max_frames_to_sum = None, summing_dtype = np.uint16):
        
        # Store summation count info as a controllable variable
        numsum_key, max_frames_to_sum = self._store_settings(num_to_sum)
        type_key, _ = self._store_settings(summing_dtype)
        
        # Set up the depth of storage needed to store previous frames
        if max_frames_to_sum is None:
            max_frames_to_sum = max_frames_to_sum + 1
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_frames_to_sum)
        
        # Function for summing previous frames
        def sum_self_(inFrame, deck_index, num_to_sum_key, sum_type_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get list of frames to sum
            num_frames = self._resources[num_to_sum_key]
            frame_list = [self._read_from_deck(deck_index, each_idx) for each_idx in range(num_frames)] 
            
            # Numpy: np.sum(a, axis, dtype)
            summed_frame = np.sum(frame_list, axis=0, dtype = self._resources[sum_type_key])
            return np.uint8(np.clip(summed_frame, 0, 255))
        
        sum_func = partial(sum_self_, deck_index = deck_idx, num_to_sum_key = numsum_key, sum_type_key = type_key)
        
        return self._seq_func(sum_func)
    
    # .................................................................................................................
    
    def add(self, image, intensity = 1.0):
        
        # Store input image data
        image_key, _ = self._store_settings(image)
        intensity_key, _ = self._store_settings(intensity)
        
        def add_(inFrame, image_res_key, intensity_res_key):            
            # OpenCV: cv2.addWeighted(src1, alpha, src2, beta, gamma)
            return cv2.addWeighted(src1 = inFrame, 
                                   alpha = self._resources[intensity_res_key],
                                   src2 = self._resources[image_res_key],
                                   beta = 1.0,
                                   gamma = 0.0)
        
        add_func = partial(add_, image_res_key = image_key, intensity_res_key = intensity_key)
        
        return self._seq_func(add_func)
    
    # .................................................................................................................
    
    def add_self(self, backstep = 1, max_backstep = None, weighting = 0.5):
        
        # Store input control data
        weight_key, _ = self._store_settings(weighting)
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        def add_self_(inFrame, deck_index, stepsize_key, weighting_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get backward-step frame
            prev_frame = self._read_from_deck(deck_index, self._resources[stepsize_key])
            
            # OpenCV: cv2.addWeighted(src1, alpha, src2, beta, gamma)
            return cv2.addWeighted(src1 = inFrame, 
                                   alpha = 1 - self._resources[weighting_key],
                                   src2 = prev_frame,
                                   beta = self._resources[weighting_key],
                                   gamma = 0.0)
            
        add_func = partial(add_self_, deck_index = deck_idx, stepsize_key = backstep_key, weighting_key = weight_key)
        
        return self._seq_func(add_func)
    
    # .................................................................................................................
    
    def subtract(self, image, dtype = np.int16):
        # OpenCV: cv2.subtract(src1, src2, dtype)
        
        # Store input image data
        image_key, _ = self._store_settings(image)

        def subtract_(inFrame, key_name, data_type):
            return np.subtract(inFrame, self._resources[key_name], dtype = data_type)
        
        subtract_func = partial(subtract_, key_name = image_key, data_type = dtype)
        
        return self._seq_func(subtract_func)
    
    # .................................................................................................................
    
    def subtract_self(self, backstep = 1, max_backstep = None, dtype = np.int16):
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for directly subtracting frames (as opposed to absdiff, which cannot return negative values) 
        def subtract_self_(inFrame, deck_index, stepsize_key, data_type):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get backward-step frame
            prev_frame = self._read_from_deck(deck_index, self._resources[stepsize_key])
            
            return np.subtract(inFrame, prev_frame, dtype = data_type)
        
        subtract_func = partial(subtract_self_, deck_index = deck_idx, stepsize_key = backstep_key, data_type = dtype)
        
        return self._seq_func(subtract_func)
    
    # .................................................................................................................
    
    def and_with(self, image):
        # NOT WELL TESTED!
        
        # Store input image data
        image_key, _ = self._store_settings(image)
        
        def and_with_(inFrame, res_key):
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(inFrame, self._resources[res_key])
        
        and_func = partial(and_with_, res_key = image_key)
        
        return self._seq_func(and_func)
    
    # .................................................................................................................
    
    def and_with_self(self, backstep = 1, max_backstep = None):
        # NOT WELL TESTED!
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for bitwise ANDing with previous frames
        def and_with_self_(inFrame, deck_index, stepsize_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get backward-step frame
            prev_frame = self._read_from_deck(deck_index, self._resources[stepsize_key])
            
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(inFrame, prev_frame)
        
        and_self_func = partial(and_with_self_, deck_index = deck_idx, stepsize_key = backstep_key)
        
        return self._seq_func(and_self_func)
    
    # .................................................................................................................
    
    def andsum_self(self, backstep = 1, max_backstep = None):
        # NOT WELL TESTED!
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for bitwise ANDing with multiple previous frames
        def andsum_self_(inFrame, deck_index, stepsize_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
        
            # Get list of frames to AND together
            num_frames = self._resources[stepsize_key]
            andsum_frame = inFrame.copy() #np.full(inFrame.shape, 255, dtype=np.uint8)
            for each_idx in range(num_frames):
                andsum_frame = cv2.bitwise_and(andsum_frame, self._read_from_deck(deck_index, each_idx))                
            return andsum_frame
        
        andsum_self_func = partial(andsum_self_, deck_index = deck_idx, stepsize_key = backstep_key)
        
        return self._seq_func(andsum_self_func)    
    
    # .................................................................................................................
    
    def or_with(self, image):
        # NOT WELL TESTED!
        
        # Store input image data
        image_key, _ = self._store_settings(image)
        
        def or_with_(inFrame, res_key):
            # OpenCV: cv2.bitwise_or(src1, src2)
            return cv2.bitwise_or(inFrame, self._resources[res_key])
        
        or_func = partial(or_with_, res_key = image_key)
        
        return self._seq_func(or_func)
    
    # .................................................................................................................
    
    def or_with_self(self, backstep = 1, max_backstep = None):
        # NOT WELL TESTED!
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for bitwise ORing with previous frames
        def or_with_self_(inFrame, deck_index, stepsize_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
            
            # Get backward-step frame
            prev_frame = self._read_from_deck(deck_index, self._resources[stepsize_key])
            
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_or(inFrame, prev_frame)
        
        or_self_func = partial(or_with_self_, deck_index = deck_idx, stepsize_key = backstep_key)
        
        return self._seq_func(or_self_func)
    
    # .................................................................................................................
    
    def orsum_self(self, backstep = 1, max_backstep = None):
        # NOT WELL TESTED!
        
        # Store backward step info as a controllable variable
        backstep_key, backstep_val = self._store_settings(backstep)
        
        # Set up the depth of storage needed to store previous frames
        if max_backstep is None:
            max_backstep = backstep_val + 1
            
        # Make sure we have the minimum number of storage space needed (in case user enters a bad value)
        max_backstep = max(max_backstep, backstep_val + 1)
        
        # Create storage for the frame data
        deck_idx = self._create_new_framedeck(max_deck_size = max_backstep)
        
        # Function for bitwise ORing with multiple previous frames
        def orsum_self_(inFrame, deck_index, stepsize_key):
            
            # Add new frame to the deck
            self._add_to_deck(deck_index, inFrame)
        
            # Get list of frames to OR together
            num_frames = self._resources[stepsize_key]
            orsum_frame = inFrame.copy()#np.zeros(inFrame.shape, dtype=np.uint8)
            for each_idx in range(num_frames):
                orsum_frame = cv2.bitwise_or(orsum_frame, self._read_from_deck(deck_index, each_idx))                
            return orsum_frame
        
        orsum_self_func = partial(orsum_self_, deck_index = deck_idx, stepsize_key = backstep_key)
        
        return self._seq_func(orsum_self_func)        
    
    # .................................................................................................................
    
    def _seq_func(self, in_func):        
        self._function_sequence.append(in_func)
    
    # .................................................................................................................
    
    def _store_settings(self, setting):
        
        # If a set is supplied, assume that we're referencing an existing resource
        if type(setting) is set:
            
            # Make sure only a single setting was supplied
            if len(setting) > 1:
                raise AttributeError("Can't supply more than 1 setting at a time!")
                
            # Check that the key name exists
            key_name = setting.pop()
            if key_name not in self._resources:
                raise KeyError("Key not found: {}".format(key_name))
            
            return key_name, self._resources[key_name]
        
        
        # If a dictionary is supplied 
        if type(setting) is dict:
            
            # Make sure only a single setting was supplied
            if len(setting) > 1:
                raise AttributeError("Can't supply more than 1 setting at a time!")
            
            # Extract data from dictionary
            key_name = list(setting.keys())[0]
            key_val = setting[key_name]  
            
            # Add resource
            self._resources[key_name] = key_val
            
            return key_name, key_val
            
            
        # Default case - just generate a key name and store the resource automatically
        
        # Create a (hopefully unique?) key name for the data
        res_size = len(self._resources)
        key_name = "_{:0>3}".format(res_size)
        key_val = setting
        
        # Check that the key doesn't already exist
        if key_name in self._resources:
            raise KeyError("Setting name already exists!")
            
        # Add resource to storage
        self._resources[key_name] = key_val
        
        return key_name, key_val
    
    # .................................................................................................................
    
    def _store_comparison(self, key_name):
        
        try:
            self._res_compare[key_name] = self._resources[key_name].copy()
        except AttributeError:
            self._res_compare[key_name] = self._resources[key_name]
            
    # .................................................................................................................
    
    def _compare_changes(self, key_name):
        
        value_changed = (self._res_compare[key_name] != self._resources[key_name])
        if value_changed:
            self._store_comparison(key_name)
        
        return value_changed
    
    # .................................................................................................................
    
    def _compare_changes_arrays(self, key_name):
        
        value_changed = np.array_equal(np.array(self._res_compare[key_name]), np.array(self._resources[key_name]))
        if value_changed:
            self._store_comparison(key_name)
        
        return value_changed
    
    # .................................................................................................................
    
    def _create_new_framedeck(self, max_deck_size):        
        
        # Add new (empty) deque to framedeck list, then return the index to that list entry
        self._framedeck_list.append(deque([], maxlen=max_deck_size))
        return len(self._framedeck_list) - 1
    
    # .................................................................................................................
    
    def _add_to_deck(self, deck_index, frame):
        self._framedeck_list[deck_index].append(frame)
    
    # .................................................................................................................
    
    def _read_from_deck(self, deck_index, backward_index):
        
        # Make sure there is data in the deck, otherwise return the closest index
        last_idx = len(self._framedeck_list[deck_index]) - 1
        if backward_index > last_idx:
            return self._framedeck_list[deck_index][0]
        
        # Grab the requested index (counting from the end of the list)
        back_idx_offset = -backward_index - 1
        return self._framedeck_list[deck_index][back_idx_offset]
    
    # .................................................................................................................
    
    # .................................................................................................................
    
    # .................................................................................................................
    


#%% Demo

if __name__ == "__main__":
    
    import datetime as dt
    
    #bgImg = cv2.imread("/home/eo/Desktop/PythonData/Shared/backgrounds/Dortec/dtb_4_bgImage.png")
    
    
    
    videoObj = cv2.VideoCapture("/home/eo/Desktop/PythonData/Shared/videos/dtb_4.avi")
    #videoObj = cv2.VideoCapture(0)    
    
    rec, bgImg = videoObj.read()
    blurBG = cv2.GaussianBlur(bgImg, (11,11), 0)
    
    
    vv = FrameLab("SUMONE")
    vv.set_input(video_reference = videoObj)
    vv.set_sample_rate(1)
    vv.enable_intermediates()
    vv.absdiff_self(4)
    vv.sum_self(20)
    
    
    
    
    gg = FrameLab("Outline Stage")
    
    gg.set_input(video_reference = videoObj)
    gg.set_sample_rate(1)
    gg.enable_intermediates()
    
    gg.blur((11,11))
    gg.absdiff(blurBG)
    gg.grayscale(output_bgr = True)
    gg.threshold(30)
    gg.canny(50, 220, output_bgr = True)
    #gg.morphology(operation=cv2.MORPH_DILATE, kernel_shape=cv2.MORPH_ELLIPSE)
    gg.invert()
    #gg.pixelate(3)
    #gg.and_with(bgImg)
    #gg.add(bgImg, intensity = 2.5)
    #gg.resize_WH((1280,720))
    
    zz = FrameLab("Ghost Stage")
    zz.set_input(video_reference = videoObj)
    zz.set_sample_rate(1)
    zz.enable_intermediates()
    zz.blur((11,11))
    zz.absdiff(blurBG)
    zz.grayscale(output_bgr = True)
    zz.threshold(30)
    #gg.morphology((11,11))
    #zz.pixelate(3)
    zz.add(bgImg, intensity = 2.5)
    
    frameWH = gg.fetch_dimensions("width", "height")
    vidFPS = videoObj.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    total_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #videoOut = cv2.VideoWriter("/home/eo/Desktop/testRec2.avi", fourcc, vidFPS, tuple(frameWH))
    
    
    st_time = dt.datetime.now()
    
    for k in range(total_frames):
        
        (rec, inFrame) = videoObj.read()
        if not rec: break
        
        '''
        qq = gg.process(inFrame)
        pp = zz.process(inFrame)
        nn = cv2.bitwise_and(qq, pp)
        
        cv2.imshow("TEST", qq)
        cv2.imshow("TEST2", pp)
        cv2.imshow("Test3", nn)
        '''
        
        vv_img = vv.process(inFrame)
        cv2.imshow("SUMTEST", vv_img)
        
        keyPress = cv2.waitKey(1) & 0xFF
        if keyPress == ord('q') or keyPress == 27:
            break
        
        
    videoObj.release()
    #videoOut.release()
    cv2.destroyAllWindows()
    
    en_time = dt.datetime.now()
    dt = en_time - st_time
    print("")
    print("Took", dt.total_seconds())

