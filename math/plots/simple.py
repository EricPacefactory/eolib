#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:39:03 2019

@author: eo
"""


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes

class Positive_Line_Plot:
    
    # .................................................................................................................
    
    def __init__(self, frame_width, frame_height, x_pad = 20, y_pad = 20, use_metric_prefix = True):
        
        # Store sizing info
        self.width = frame_width
        self.height = frame_height
        self.x_pad = x_pad
        self.y_pad = y_pad
        self.plot_width = frame_width - (2 * x_pad)
        self.plot_height = frame_height - (2 * y_pad)
        self.use_metric_prefix = use_metric_prefix
        self.metric_prefix_strs = {0: "", 
                                   1: "k", 2: "M", 3: "G", 4: "T",
                                   -1: "m", -2: "u", -3: "n", -4: "p"}
        
        # Set up border drawing co-ords
        self.border_tl = (self.x_pad, self.y_pad)
        self.border_br = (self.x_pad + self.plot_width - 1, self.y_pad + self.plot_height - 1)
        
        # Set up default text configuration
        self.text_x_offset = 3
        self.min_text_pos = (0, 0)
        self.max_text_pos = (0, 0)
        self.text_config = {}
        self.change_text_config()        
        
        # Set up default plotting background colors
        self.plot_bg_color = None
        self.padding_color = None
        self.blank_plot_frame = None
        self.change_plot_bg_colors()
        
        # Set up default plotting line config
        self.plot_line_color = None
        self.plot_line_thickness = None
        self.plot_line_type = None
        self.change_plot_line()
    
    # .................................................................................................................
    
    def change_plot_line(self, 
                         plot_line_color = (50, 100, 170), 
                         plot_line_thickness = 1, 
                         plot_line_type = cv2.LINE_AA):
        
        self.plot_line_color = plot_line_color
        self.plot_line_thickness = plot_line_thickness
        self.plot_line_type = plot_line_type
    
    # .................................................................................................................
    
    def change_plot_bg_colors(self, plot_bg_color = (0, 0, 0), padding_color = (40, 40, 40)):
        
        # Update plotting colors
        self.plot_bg_color = plot_bg_color
        self.padding_color = padding_color
                               
        # Create & store the new blank plotting frame
        self.blank_plot_frame = np.full((self.plot_height, self.plot_width, 3), self.plot_bg_color, dtype=np.uint8)
    
    # .................................................................................................................
    
    def change_text_config(self,
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 0.4, 
                           color = (220, 220, 220), 
                           thickness = 1, 
                           lineType = cv2.LINE_AA):
        
        self.text_config = {"fontFace": fontFace,
                            "fontScale": fontScale,
                            "color": color,
                            "thickness": thickness,
                            "lineType": lineType}
        
        # Figure out how big the text is with these settings
        (x_size, y_size), y_base = cv2.getTextSize("Test Text", fontFace, fontScale, thickness)
        
        # Set up positioning of min/max value text
        text_x = self.x_pad + self.text_x_offset
        self.max_text_pos = (text_x, self.y_pad + y_size + y_base)
        self.min_text_pos = (text_x, self.y_pad - y_base + self.plot_height - 1)
    
    # .................................................................................................................
    
    def plot(self, y_data, max_plot_y = None, display_units = ""):
        
        # Create a copy of the blank plotting frame, so we don't mess up the original
        plot_frame = self.blank_plot_frame.copy()
        
        # Bail 
        if len(y_data) < 2:
            y_data = [0, 0]
        
        # Figure out the maximum value to use for display
        max_plot_y = max_plot_y if (max_plot_y is not None) else self.pick_max_y(y_data)
        rounded_max, max_display_string = self.round_to_factors_of_thousand(max_plot_y, display_units)
        
        # Scale the plot data, so that it is normalized relative to the max value
        normalized_data = np.float32(y_data) / rounded_max
        
        # Figure out plotting co-ords for the frame timing plot
        x_plot = np.linspace(0, self.plot_width - 1, len(y_data))
        y_plot = self.plot_height*(1 - normalized_data)
        xy_plot = np.int32(np.round(np.vstack((x_plot, y_plot)).T))
        
        # Draw the frame timing as a line plot
        is_closed = False
        cv2.polylines(plot_frame, [xy_plot], is_closed, 
                      self.plot_line_color, self.plot_line_thickness, self.plot_line_type)
        
        # Add padded borders to create the final image size
        output_frame = cv2.copyMakeBorder(plot_frame, 
                                          top = self.y_pad, 
                                          bottom = self.y_pad, 
                                          left = self.x_pad, 
                                          right = self.x_pad, 
                                          borderType = cv2.BORDER_CONSTANT,
                                          value = self.padding_color)
    
        # Draw a border around the plot
        cv2.rectangle(output_frame, self.border_tl, self.border_br, (200, 200, 200), 2, cv2.LINE_4)
        
        # Draw min/max plot values for reference
        cv2.putText(output_frame, "0 {}".format(display_units), self.min_text_pos, **self.text_config)
        cv2.putText(output_frame, max_display_string, self.max_text_pos, **self.text_config)
    
        return output_frame
    
    # .................................................................................................................
    
    def imshow(self, window_name, y_data, max_plot_y = None, display_units = "", frame_delay_ms = None):
        
        # Helper function, which just generates a plotted graph in a window for display
        plot_frame = self.plot(y_data, max_plot_y, display_units)
        cv2.imshow(window_name, plot_frame)
        if frame_delay_ms is not None:
            cv2.waitKey(frame_delay_ms)
    
    # .................................................................................................................
    
    def round_to_factors_of_thousand(self, raw_max_value, units = ""):
        
        '''
        Function which takes a maximum value and converts it into a rounded value between 0 and 999,
        also provides a scaling factor (powers of 1000)
        '''
        
        # Figure out the maximum value, in terms of factors of 1000, so that the scaled max is between 0 and 999
        factors_of_thousand = int(np.floor(np.log10(raw_max_value) / 3)) if raw_max_value > 0 else 0
        scaling_factor = 1 / (1000 ** factors_of_thousand)
        scaled_max = raw_max_value * scaling_factor
        
        # Round up to the nearest  factor of 2 or 10, depending on how big the max value is
        round_to = 2 if scaled_max < 10 else 10
        rounded_max = int(round_to * np.ceil(scaled_max / round_to))
        unscaled_rounded_max = rounded_max / scaling_factor
        
        # Create a string representation of the scaled/rounded value
        if self.use_metric_prefix:
            scaling_string = self.metric_prefix_strs.get(factors_of_thousand, "?")
            unit_string = "{}{}".format(scaling_string, units)
        else:
            scaling_string = "" if factors_of_thousand == 0 else "x 1E{:.0f}".format(3*factors_of_thousand)
            unit_string = "{} {}".format(scaling_string, units)
        rounded_max_value_string = "{} {}".format(rounded_max, unit_string)
        
        return unscaled_rounded_max, rounded_max_value_string
    
    # .................................................................................................................
    
    @staticmethod
    def pick_max_y(y_data, percentile_max = 95, absolute_to_percentile_max_ratio = 20):
        
        '''
        Function which tries to choose a good (i.e. ignoring outliers) 'max' value from a dataset
        '''
        
        # Get the absolute and percentile-based maximums
        abs_max = np.max(y_data)
        per_max = np.percentile(y_data, percentile_max)
        
        # Decide if the maximum value should be taken from the absolute max, or using percentiles (to avoid outliers)
        max_plot_y_raw = abs_max if ((abs_max / per_max) < absolute_to_percentile_max_ratio) else per_max
        
        return max_plot_y_raw
    
    # .................................................................................................................
    # .................................................................................................................
    
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Demo
    
if __name__ == "__main__":
    
    # Get rid of any existing windows
    cv2.destroyAllWindows()
    
    # Create some random data for plotting demo
    random_data = np.random.randint(3, 1722, 350)
    
    # Plot using the simple positive line plot
    pos_line = Positive_Line_Plot(800, 450, use_metric_prefix = False)
    pos_line.imshow("Random data plot", random_data, display_units = "unitless", frame_delay_ms = 400)
    
    pass

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap


