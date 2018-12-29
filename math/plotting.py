#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:38:08 2018

@author: eo
"""

import cv2
import numpy as np


class Simple_XY:
    
    def __init__(self, width, height, padding_tlbr = 10, title = "", x_label = "", y_label = "", gfx = {}):
        
        # Get overall plot size
        self.width = width
        self.height = height
        
        # Get padding values
        pad_helper = lambda x, idx: int(x[idx % len(x)]) if type(x) in (list, tuple) else int(x)
        self._top_pad = pad_helper(padding_tlbr, 0)
        self._left_pad = pad_helper(padding_tlbr, 1)
        self._bot_pad = pad_helper(padding_tlbr, 2)
        self._right_pad = pad_helper(padding_tlbr, 3)
        
        # Get inner plotting size and positions
        self.plot_width = self.width - self._left_pad - self._right_pad
        self.plot_height = self.height - self._top_pad - self._bot_pad
        self._tl = np.int32((self._left_pad, self._top_pad))
        self._br = np.int32((self.width - self._right_pad - 1, self.height - self._bot_pad - 1))
        
        # Get plotting origin for convenience
        self.origin = np.int32((self._tl[0], self._br[1]))
        self._origin_x = self.origin[0]
        self._origin_y = self.origin[1]
        
        # Set up plot scaling functions
        self._xscale = lambda x: x
        self._yscale = lambda y: y
        self._plotx_func = lambda x: np.int32(self._origin_x + (self.plot_width-1)*self._xscale(x))
        self._ploty_func = lambda y: np.int32(self._origin_y - (self.plot_height-1)*self._yscale(y))
        self._min_x = None
        self._max_x = None
        self._min_y = None
        self._max_y = None
        
        # Graphics settings
        self.gfx = {**self._default_gfx(), **gfx}
        
        # Allocate space for an empty frame and a (possible) template
        self._empty_frame = self._draw_empty_frame()
        self._template = None
        
        # Storage for labels
        self._title = None
        self._x_label = None
        self._y_label = None
        self._x_ticks = None
        self._y_ticks = None
        self._grid_x = None
        self._grid_y = None

        
        # Store title and axis labels if provided
        if title: self.title(title)
        if x_label: self.x_label(x_label)
        if y_label: self.y_label(y_label)
    
    # .................................................................................................................    
    
    def plot(self, y_data, x_data = None, *, 
             auto_scale_x = False, 
             auto_scale_y = False,
             round_nearest_x = None, 
             round_nearest_y = None):
        
        # First handle case on empty x-data
        x_data = np.linspace(0, 1, len(y_data)) if x_data is None else x_data
        
        # Set up axis scaling if needed
        if auto_scale_x is not False:
            if auto_scale_x is True:
                self._min_x = None
                self._max_x = None
            
            if None in [self._min_x, self._max_x]:
                min_x = np.min(x_data) if self._min_x is None else self._min_x
                max_x = np.max(x_data) if self._max_x is None else self._max_x                
                self.xlim(min_x, max_x, round_nearest_x)
                
        if auto_scale_y is not False:             
            if auto_scale_y is True:
                self._min_y = None
                self._max_y = None
            
            if None in [self._min_y, self._max_y]:
                min_y = np.min(y_data) if self._min_y is None else self._min_y
                max_y = np.max(y_data) if self._max_y is None else self._max_y
                self.ylim(min_y, max_y, round_nearest_y)
        
        
        # Get a copy of the empty frome to draw in to
        plot_image = self._empty_frame.copy()
        
        # Draw text labels
        self._draw_title(plot_image)
        self._draw_x_label(plot_image)
        self._draw_y_label(plot_image)
        
        # Draw grid lines in plotting area
        self._draw_grid_x(plot_image)
        self._draw_grid_y(plot_image)
        
        # Draw axis tick marks and numbers
        self._draw_x_ticks(plot_image)
        self._draw_y_ticks(plot_image)
            
        # Draw data
        self._draw_xy_data(plot_image, x_data, y_data)
        
        # Finally draw axes and bounding rectangle over top of plot to clean things up
        self._draw_axes(plot_image)
        self._draw_bounding_frame(plot_image)
        
        return plot_image
    
    # ................................................................................................................. 
    
    def plot_template(self, y_data, x_data = None):
        
        if self._template is not None:            
            plot_image = self._template.copy()            
            self._draw_xy_data(plot_image, x_data, y_data)
            self._draw_bounding_frame(plot_image)            
            return plot_image
        
        raise AttributeError("No template available for plotting!")
    
    # ................................................................................................................. 
    
    def imshow(self, window_title, y_data, x_data = None, *,
               auto_scale_x = None, 
               auto_scale_y = None,
               round_nearest_x = None, 
               round_nearest_y = None):
        
        if self._template is not None:
            plot_image = self.plot_template(y_data, x_data)            
        else:            
            plot_image = self.plot(y_data, x_data, 
                                   auto_scale_x = auto_scale_x,
                                   auto_scale_y = auto_scale_y, 
                                   round_nearest_x = round_nearest_x, 
                                   round_nearest_y = round_nearest_y)
            
        cv2.imshow(window_title, plot_image)
    
    # ................................................................................................................. 
    
    def add_point(self, plot_image, point, radius = None, color = None, thickness = None, line_type = None, outline_color = None):
        
        # Map data points into plotting co-ords
        point_x = self._plotx_func(point[0])
        point_y = self._ploty_func(point[1])
        center = (point_x, point_y)
        radius = 3 if radius is None else radius
        color = self.gfx["line_color"] if color is None else color
        thickness = self.gfx["line_thickness"] if thickness is None else thickness
        line_type = self.gfx["line_type"] if line_type is None else line_type
        
        cv2.circle(plot_image, center, radius, color, thickness, line_type)
        if outline_color is not None:
            cv2.circle(plot_image, center, radius, outline_color, 1, line_type)
    
    # ................................................................................................................. 
    
    def add_points(self, plot_image, points_list):
        
        
        return plot_image
    
    # .................................................................................................................    
    
    @staticmethod
    def interpolate(x_interp, x_data, y_data):        
        y_interp = np.interp(x_interp, x_data, y_data)        
        return x_interp, y_interp
    
    # .................................................................................................................    
    
    @staticmethod
    def smooth(raw_data, num_samples = 5, mode = "same"):        
        smooth_data = np.convolve(raw_data, 
                                  [1/num_samples]*num_samples, 
                                  mode = mode)
        return smooth_data
    
    # .................................................................................................................    
    
    def generate_template(self, 
                          plot_width = None,
                          plot_height = None,
                          title = None, 
                          x_label = None, 
                          y_label = None, 
                          x_data = None, 
                          y_data = None):
        
        
        # First take any inputs needed to generate plot
        
        # Draw frame
        # Draw gride lines
        # Draw y-label
        # Draw x-label
        # Draw draw title
        # (optional) draw x-ticks
        # (Optional) draw y ticks
        
        pass
    
    # .................................................................................................................  
    
    def xlim(self, min_x = None, max_x = None, round_nearest = None):

        if min_x is None:
            self._min_x = None
        else:
            self._min_x = min_x if round_nearest is None else np.floor(min_x/round_nearest)*round_nearest
            
        if max_x is None:
            self._max_x = None
        else:
            self._max_x = max_x if round_nearest is None else np.ceil(max_x/round_nearest)*round_nearest
            
        self._generate_normalizing_funcs()
    
    # .................................................................................................................  
    
    def ylim(self, min_y = None, max_y = None, round_nearest = None):
        
        if min_y is None:
            self._min_y = None
        else:
            self._min_y = min_y if round_nearest is None else np.floor(min_y/round_nearest)*round_nearest
            
        if max_y is None:
            self._max_y = None
        else:
            self._max_y = max_y if round_nearest is None else np.ceil(max_y/round_nearest)*round_nearest
            
        self._generate_normalizing_funcs()
    
    # .................................................................................................................  
    
    def _generate_normalizing_funcs(self):
        
        # Create scaling functions that map smallest values to 0 and largest to 1 (i.e. normalizing)
        if None not in [self._min_x, self._max_x]:
            self._xscale = lambda x: (x - self._min_x) / (self._max_x - self._min_x)
        if None not in [self._min_y, self._max_y]:
            self._yscale = lambda y: (y - self._min_y) / (self._max_y - self._min_y)
    
    # .................................................................................................................  
    
    def _draw_xy_data(self, plot_frame, x_data, y_data, line_color = None, line_thickness = None, line_type = None):
        
        # Map data points into plotting co-ords
        plot_x = self._plotx_func(x_data)
        plot_y = self._ploty_func(y_data)
        color = self.gfx["line_color"] if line_color is None else line_color
        thickness = self.gfx["line_thickness"] if line_thickness is None else line_thickness
        line_type = self.gfx["line_type"] if line_type is None else line_type
        
        # Plot data        
        plot_xy = np.vstack((plot_x, plot_y)).T
        cv2.polylines(plot_frame, [plot_xy], False, 
                      color, 
                      thickness, 
                      line_type)
        
        return plot_frame
    
    # .................................................................................................................   
    
    def x_ticks(self, ticks, *, labels = True, tick_length = 10, decimal_places = 0, 
                fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        # Get custom text settings for tick labels
        text_settings = self._customized_text(fontFace, fontScale, color, thickness, lineType)   
        
        # Calculate the tick positions (or take direct, if input isn't a scalar)
        norm_pos = np.linspace(0, 1, ticks) if type(ticks) in [int, float] else ticks
        norm_pos = np.float32(norm_pos)
        
        # Calculate tick positions on the plot itself (to save time figuring them out when drawing)
        plot_pos_x = np.array(self._origin_x + (self.plot_width - 1)*norm_pos)
        plot_pos_y = self._origin_y 
        
        # Get drawing values for convenience
        plot_pos_y1 = plot_pos_y - int(tick_length/2)
        plot_pos_y2 = plot_pos_y + int(tick_length/2)
        text_pos_y = plot_pos_y2 + 5
        
        self._x_ticks = {"norm_pos_x": norm_pos,
                         "plot_pos_x": plot_pos_x,
                         "plot_pos_y1": plot_pos_y1,
                         "plot_pos_y2": plot_pos_y2,
                         "enable_labels": labels,
                         "text": {"text_pos_y": text_pos_y, 
                                  "decimal_places": decimal_places,
                                  **text_settings}}
    
    # .................................................................................................................   
    
    def _draw_x_ticks(self, plot_frame):
        
        if self._x_ticks is not None:
            
            # Draw tick marks (i.e. little lines on x-axis)
            for eachPosX in self._x_ticks["plot_pos_x"]:
                
                y1 = self._x_ticks["plot_pos_y1"]
                y2 = self._x_ticks["plot_pos_y2"]
                
                cv2.line(plot_frame, (eachPosX, y1), (eachPosX, y2), 
                         self.gfx["fg_color"], self.gfx["line_thickness"], cv2.LINE_4)
                
            # Draw tick labels (i.e. numbers corresponding to plotting position)
            if None not in [self._min_x, self._max_x] and self._x_ticks["enable_labels"]:
                
                # Get some convenient variables...
                dx = self._max_x - self._min_x                
                text_pos_y = self._x_ticks["text"]["text_pos_y"]
                decimal_places = self._x_ticks["text"]["decimal_places"]
                font = self._x_ticks["text"]["fontFace"]
                scale = self._x_ticks["text"]["fontScale"]
                color = self._x_ticks["text"]["fontColor"]
                thickness = self._x_ticks["text"]["fontWeight"]
                line_type = self._x_ticks["text"]["line_type"]
                
                # Draw each of the tick labels
                for eachNormX, eachPosX in zip(self._x_ticks["norm_pos_x"], self._x_ticks["plot_pos_x"]):
                    
                    # Get x value at the tick
                    x_tick_val = self._min_x + dx*eachNormX
                    x_tick_str = "{0:.{1}f}".format(x_tick_val, decimal_places)
                    
                    # Find text sizing and positioning
                    text_size = cv2.getTextSize(x_tick_str, font, scale, thickness)[0]                    
                    px = int(eachPosX - text_size[0]/2)
                    py = int(text_pos_y + text_size[1])
                    text_pos = (px, py)

                    cv2.putText(plot_frame, x_tick_str, text_pos, font, scale, color, thickness, line_type)
                    
        return plot_frame
    
    # .................................................................................................................   
    
    def y_ticks(self, ticks, *, labels = True, tick_length = 10, decimal_places = 0, 
                fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        # Get custom text settings for tick labels
        text_settings = self._customized_text(fontFace, fontScale, color, thickness, lineType)   
        
        # Calculate the tick positions (or take direct, if input isn't a scalar)
        norm_pos = np.linspace(0, 1, ticks) if type(ticks) in [int, float] else ticks
        norm_pos = np.float32(norm_pos)
        
        # Calculate tick positions on the plot itself (to save time figuring them out when drawing)
        plot_pos_x = self._origin_x
        plot_pos_y = self._origin_y - (self.plot_height - 1)*norm_pos
        
        # Get drawing values for convenience
        plot_pos_x1 = plot_pos_x - int(tick_length/2)
        plot_pos_x2 = plot_pos_x + int(tick_length/2)
        text_pos_x = plot_pos_x1 - 5
        
        self._y_ticks = {"norm_pos_y": norm_pos,
                         "plot_pos_x1": plot_pos_x1,
                         "plot_pos_x2": plot_pos_x2,
                         "plot_pos_y": plot_pos_y,
                         "enable_labels": labels,
                         "text": {"text_pos_x": text_pos_x, 
                                  "decimal_places": decimal_places,
                                  **text_settings}}
    
    # .................................................................................................................  
    
    def _draw_y_ticks(self, plot_frame):
        
        if self._y_ticks is not None:
            
            for eachPosY in self._y_ticks["plot_pos_y"]:
                
                x1 = self._y_ticks["plot_pos_x1"]
                x2 = self._y_ticks["plot_pos_x2"]
                
                cv2.line(plot_frame, (x1, eachPosY), (x2, eachPosY), 
                         self.gfx["fg_color"], self.gfx["line_thickness"], cv2.LINE_4)
                
            # Draw tick labels (i.e. numbers corresponding to plotting position)
            if None not in [self._min_y, self._max_y] and self._y_ticks["enable_labels"]:
                
                # Get some convenient variables...
                dy = self._max_y - self._min_y                
                text_pos_x = self._y_ticks["text"]["text_pos_x"]
                decimal_places = self._y_ticks["text"]["decimal_places"]
                font = self._y_ticks["text"]["fontFace"]
                scale = self._y_ticks["text"]["fontScale"]
                color = self._y_ticks["text"]["fontColor"]
                thickness = self._y_ticks["text"]["fontWeight"]
                line_type = self._y_ticks["text"]["line_type"]
                
                # Draw each of the tick labels
                for eachNormY, eachPosY in zip(self._y_ticks["norm_pos_y"], self._y_ticks["plot_pos_y"]):
                    
                    # Get x value at the tick
                    y_tick_val = self._min_y + dy*eachNormY
                    y_tick_str = "{0:.{1}f}".format(y_tick_val, decimal_places)
                    
                    # Find text sizing and positioning
                    text_size = cv2.getTextSize(y_tick_str, font, scale, thickness)[0]
                    px = int(text_pos_x - text_size[0])
                    py = int(eachPosY + text_size[1]/2)
                    
                    text_pos = (px, py)

                    cv2.putText(plot_frame, y_tick_str, text_pos, font, scale, color, thickness, line_type)
                
        return plot_frame
    
    # .................................................................................................................  
    
    def x_label(self, x_label, fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        text_settings = self._customized_text(fontFace, fontScale, color, thickness, lineType)   
        text_sizing = cv2.getTextSize(x_label,  
                                      text_settings["fontFace"], 
                                      text_settings["fontScale"], 
                                      text_settings["fontWeight"])
        
        base_line = max(text_sizing[1], text_sizing[0][1]/2)
        text_size = text_sizing[0]
        text_pos_x = (self.width - text_size[0])/2
        text_pos_y = self.height - base_line - 1
        text_pos = (int(text_pos_x), int(text_pos_y))
        
        self._x_label = {"text": x_label,
                         "position": text_pos,
                         **text_settings}
    
    # .................................................................................................................
    
    def _draw_x_label(self, plot_frame):
        
        if self._x_label is not None:
            cv2.putText(plot_frame, 
                        text = self._x_label["text"], 
                        org = self._x_label["position"], 
                        fontFace = self._x_label["fontFace"],
                        fontScale = self._x_label["fontScale"],
                        color = self._x_label["fontColor"],
                        thickness = self._x_label["fontWeight"],
                        lineType = self._x_label["line_type"])
        
        return plot_frame
    
    # .................................................................................................................  
    
    def y_label(self, y_label, fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        text_settings = self._customized_text(fontFace, fontScale, color, thickness, lineType) 
        text_sizing = cv2.getTextSize(y_label, 
                                      text_settings["fontFace"], 
                                      text_settings["fontScale"], 
                                      text_settings["fontWeight"])
        
        text_size = text_sizing[0]
        base_line = text_sizing[1]
        
        text_pos_x = (self.height - text_size[0])/2
        text_pos_y = text_size[1] + 5
        text_pos = (int(text_pos_x), int(text_pos_y))
        
        self._y_label = {"text": y_label,
                         "position": text_pos,
                         "rotated_dimensions": (text_pos_y + base_line + 1, self.height, 3),
                         **text_settings}
        
    # .................................................................................................................  
    
    def _draw_y_label(self, plot_frame):
        
        if self._y_label is not None:
            
            # Draw y-label onto a rotated image, which will be unrotated (to get vertical writing orientation)
            rot_frame = np.full(self._y_label["rotated_dimensions"], self.gfx["bg_color"], dtype=np.uint8)
            cv2.putText(rot_frame, 
                        text = self._y_label["text"], 
                        org = self._y_label["position"], 
                        fontFace = self._y_label["fontFace"],
                        fontScale = self._y_label["fontScale"],
                        color = self._y_label["fontColor"],
                        thickness = self._y_label["fontWeight"],
                        lineType = self._y_label["line_type"])
            
            # Add rotated label into the original image
            plot_frame[:, 0:self._y_label["rotated_dimensions"][0], :] = np.rot90(rot_frame)
        
        return plot_frame
    
    # .................................................................................................................  
    
    def title(self, title, fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        text_settings = self._customized_text(fontFace, fontScale, color, thickness, lineType)            
        text_size = cv2.getTextSize(title, 
                                    text_settings["fontFace"], 
                                    text_settings["fontScale"], 
                                    text_settings["fontWeight"])[0]
        
        text_pos_x = (self.width - text_size[0])/2
        text_pos_y = text_size[1] + 5
        text_pos = (int(text_pos_x), int(text_pos_y))
        
        self._title = {"text": title,
                       "position": text_pos,
                       **text_settings}
    
    # .................................................................................................................
    
    def _draw_title(self, plot_frame):
        
        if self._title is not None:
            cv2.putText(plot_frame, 
                        text = self._title["text"], 
                        org = self._title["position"], 
                        fontFace = self._title["fontFace"],
                        fontScale = self._title["fontScale"],
                        color = self._title["fontColor"],
                        thickness = self._title["fontWeight"],
                        lineType = self._title["line_type"])
        
        return plot_frame   
    
    # .................................................................................................................
    
    def grid_x(self, rate, grid_color = None, grid_thickness = None, grid_line_type = None):
        
        if rate is not None:
            
            # Get grid custom graphics
            grid_settings = self._customized_grid(grid_color, grid_thickness, grid_line_type)
            
            # Calculate new (normalized) set of points for drawing grid lines
            norm_x_ticks = self._x_ticks["norm_pos_x"]
            num_ticks = len(norm_x_ticks)
            num_new_ticks = int(1 + rate*(num_ticks - 1))
            norm_grid_pos = np.linspace(norm_x_ticks[0], norm_x_ticks[-1], num_new_ticks)
            norm_grid_pos = np.float32(norm_grid_pos[1:-1])

            # Figure out the plotting positions of the grid lines
            plot_pos_x = np.array(self._origin_x + (self.plot_width - 1)*norm_grid_pos)
            plot_pos_y1 = self._tl[1]
            plot_pos_y2 = self._br[1]
            
            # Store grid line data
            self._grid_x = {"norm_pos_x": norm_grid_pos,
                            "plot_pos_x": plot_pos_x,
                            "plot_pos_y1": plot_pos_y1,
                            "plot_pos_y2": plot_pos_y2,
                            **grid_settings}

    
    # .................................................................................................................
    
    def _draw_grid_x(self, plot_frame):
        
        if self._grid_x is not None:            
            for eachPosX in self._grid_x["plot_pos_x"]:                
                cv2.line(plot_frame, 
                         (eachPosX, self._grid_x["plot_pos_y1"]),
                         (eachPosX, self._grid_x["plot_pos_y2"]),
                         self._grid_x["grid_color"], 
                         self._grid_x["grid_thickness"],
                         cv2.LINE_4)
                
        return plot_frame
    
    # .................................................................................................................
    
    def grid_y(self, rate, grid_color = None, grid_thickness = None, grid_line_type = None):
        
        if rate is not None:
            
            # Get grid custom graphics
            grid_settings = self._customized_grid(grid_color, grid_thickness, grid_line_type)
            
            # Calculate new (normalized) set of points for drawing grid lines
            norm_y_ticks = self._y_ticks["norm_pos_y"]
            num_ticks = len(norm_y_ticks)
            num_new_ticks = int(1 + rate*(num_ticks - 1))
            norm_grid_pos = np.linspace(norm_y_ticks[0], norm_y_ticks[-1], num_new_ticks)
            norm_grid_pos = np.float32(norm_grid_pos[1:-1])

            # Figure out the plotting positions of the grid lines
            plot_pos_y = np.array(self._origin_y - (self.plot_height - 1)*norm_grid_pos)
            plot_pos_x1 = self._tl[0]
            plot_pos_x2 = self._br[0]
            
            # Store grid line data
            self._grid_y = {"norm_pos_y": norm_grid_pos,
                            "plot_pos_y": plot_pos_y,
                            "plot_pos_x1": plot_pos_x1,
                            "plot_pos_x2": plot_pos_x2,
                            **grid_settings}
    
    # .................................................................................................................  
    
    def _draw_grid_y(self, plot_frame):
        
        if self._grid_y is not None:            
            for eachPosY in self._grid_y["plot_pos_y"]:                
                cv2.line(plot_frame, 
                         (self._grid_y["plot_pos_x1"], eachPosY),
                         (self._grid_y["plot_pos_x2"], eachPosY),
                         self._grid_y["grid_color"], 
                         self._grid_y["grid_thickness"],
                         cv2.LINE_4)
                
        return plot_frame
    
    # .................................................................................................................
    
    def _customized_text(self, fontFace = None, fontScale = None, color = None, thickness = None, lineType = None):
        
        font = self.gfx["fontFace"] if fontFace is None else fontFace
        scale = self.gfx["fontScale"] if fontScale is None else fontScale
        color = self.gfx["fg_color"] if color is None else color
        thickness = self.gfx["fontWeight"] if thickness is None else thickness
        line_type = self.gfx["line_type"] if lineType is None else lineType
        
        # Covenient bump for grayscale colors
        color = tuple([color] * 3) if type(color) is int else color
        
        return {"fontFace": font,
                "fontScale": scale,
                "fontColor": color,
                "fontWeight": thickness,
                "line_type": line_type}
    
    # .................................................................................................................
    
    def _customized_grid(self, grid_color = None, grid_thickness = None, grid_line_type = None):
        
        color = self.gfx["grid_color"] if grid_color is None else grid_color
        thickness = self.gfx["line_thickness"] if grid_thickness is None else grid_thickness
        line_type = self.gfx["line_type"] if grid_line_type is None else grid_line_type
        
        # Covenient bump for grayscale colors
        color = tuple([color] * 3) if type(color) is int else color
        
        return {"grid_color": color,
                "grid_thickness": thickness,
                "grid_line_type": line_type}
    
    # .................................................................................................................  
    
    def _draw_empty_frame(self):
        empty_frame = np.full((self.height, self.width, 3), self.gfx["bg_color"], dtype=np.uint8)
        cv2.rectangle(empty_frame, tuple(self._tl), tuple(self._br), self.gfx["plot_bg_color"], -1)
        return empty_frame
    
    # .................................................................................................................  
    
    def _draw_bounding_frame(self, plot_frame):
        cv2.rectangle(plot_frame, tuple(self._tl), tuple(self._br), 
                      self.gfx["fg_color"], self.gfx["fg_thickness"], self.gfx["line_type"])
        
    # .................................................................................................................
    
    def _draw_axes(self, plot_frame):
        
        if self._min_x < 0 and self._max_x > 0:
            x_data = (0, 0)
            y_data = (self._min_y, self._max_y)            
            self._draw_xy_data(plot_frame, x_data, y_data, 
                               line_color = [int(eachChannel*0.75) for eachChannel in self.gfx["fg_color"]], 
                               line_thickness = 2*self.gfx["line_thickness"],
                               line_type = cv2.LINE_8)
        
        if self._min_y < 0 and self._max_y > 0:
            x_data = (self._min_x, self._max_x)
            y_data = (0, 0)  
            self._draw_xy_data(plot_frame, x_data, y_data, 
                               line_color = [int(eachChannel*0.75) for eachChannel in self.gfx["fg_color"]], 
                               line_thickness = 2*self.gfx["line_thickness"],
                               line_type = cv2.LINE_8)
    
    # .................................................................................................................  
    
    @staticmethod
    def _default_gfx():
        
        gfx = {"bg_color": (255, 255, 255),
               "plot_bg_color": (255, 255, 255),
               "fg_color": (150, 150, 150),
               "grid_color": (235, 235, 235),
               "line_color": (0, 0, 0),
               "fg_thickness": 1,
               "line_thickness": 1,
               "line_type": cv2.LINE_AA,
               "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
               "fontScale": 0.5,
               "fontWeight": 1}
        
        return gfx
    
    # .................................................................................................................    
    
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Demo
    
    
if __name__ == "__main__":
    
    cv2.destroyAllWindows()
    
    gg = Simple_XY(600, 320, [35, 70, 50, 50])
    gg.title("Example Title", fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.75, thickness = 1, color = 80)
    gg.x_label("X Axis Label (no units)", color = 80)
    gg.y_label("Y Axis Label (Sideways!)", color = (80, 80, 80))
    gg.x_ticks(13, decimal_places = 0)
    gg.y_ticks(4, decimal_places = 1)    
    gg.grid_x(2)
    gg.grid_y(4)
    
    xx = np.linspace(0, 6*np.pi, 100)    
    yy = 0.95*np.sin(xx*2*np.pi*(1/20)) + 0.45
    
    gg.xlim(0, 24, 1)
    #gg.ylim(0, 1.0, 0.1)
    
    pp = gg.plot(yy, xx, auto_scale_x = False, auto_scale_y = True, round_nearest_y = 1)
    
    cv2.imshow("Demo Graph", pp)


