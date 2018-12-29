#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:49:50 2018

@author: eo
"""


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np

from collections import deque

# ---------------------------------------------------------------------------------------------------------------------
#%% Callback classes

class Mouse_Follower:
    
    def __init__(self, 
                 state_list, 
                 borderWH = (0, 0), 
                 initial_state = None,
                 follow_states = None,
                 no_follow_states = None):
        
        # Store mouse co-ords and any frame border offsets
        self.mouse_xy = np.array((0, 0), dtype=np.int32)
        self.borderWH = np.array(borderWH, dtype=np.int32)
        
        # Store state list and count, for indexing
        self.state_list = state_list
        self._state_count = len(state_list)
        
        # Set the initial state index
        self._state_idx = 0        
        if initial_state is not None:            
            if initial_state in state_list:
                self._state_idx = state_list.index(initial_state) 
            else:
                print("")
                print("Initial state '{}' is not in state list!".format(initial_state))
                print("Defaulting to {}".format(state_list[0]))        
        
        # Check that all follow states are valid (remove non-valids)
        follow_check = state_list if follow_states is None else follow_states
        follow_check = follow_check if type(follow_check) in [list, tuple] else [follow_check]
        for idx, each_state in enumerate(reversed(follow_check)):
            if each_state not in state_list:
                print("")
                print("Follow state: '{}' is not in state list!".format(each_state))
                del follow_check[idx]
        
        # Check that all non-follow states are valid (remove non-valids)
        no_follow_check = [] if no_follow_states is None else no_follow_states
        no_follow_check = no_follow_check if type(no_follow_check) in [list, tuple] else [no_follow_check]
        for idx, each_state in enumerate(reversed(no_follow_check)):
            if each_state not in state_list:
                print("")
                print("No follow state: '{}' is not in state list!".format(each_state))
                del no_follow_check[idx]
        
        # Remove no-follows from the follow list
        follow_set = set(follow_check)
        no_follow_set = set(no_follow_check)
        self.follow_states = list(follow_set.difference(no_follow_set)) 
        

    # .................................................................................................................                
                
    def callback(self, event, mx, my, flags, param):
        
        # Record mouse xy position
        if self._enable_following():
            self.mouse_xy = np.array((mx, my)) - self.borderWH
        
        # Increment state on left click
        if event == cv2.EVENT_LBUTTONDOWN:
            self._state_idx = (1 + self._state_idx) % self._state_count
        
        # Reset state on double right click
        if event == cv2.EVENT_RBUTTONDBLCLK:
            self._state_idx = 0
    
    # .................................................................................................................        
            
    def in_state(self, *states_to_check):
        
        current_state = self.state_list[self._state_idx]
        for each_state in states_to_check:            
            
            # Sanity check. Make sure state to check is actually in the state list
            if each_state not in self.state_list:
                print("State: '{}' is not in state list!".format(each_state))
                return False
            
            # Check if we're currently in the target state
            if each_state == current_state:
                return True
            
        return False
    
    # .................................................................................................................
    
    def xy(self):
        return self.mouse_xy
        
    # .................................................................................................................  
    
    def draw_mouse_xy(self, display_frame, 
                      font = cv2.FONT_HERSHEY_SIMPLEX, 
                      scale = 0.5, 
                      color = (255, 255, 255), 
                      thickness = 1, 
                      line_type = cv2.LINE_AA,
                      with_background = True):
        
        xy = tuple(self.mouse_xy)
        text_string = "({}, {})".format(*xy) 
        
        if with_background:
            bg_color = [255 - each_col for each_col in color]
            bg_thickness = 2*thickness
            cv2.putText(display_frame, text_string, xy, font, scale, bg_color, bg_thickness, line_type)        
        cv2.putText(display_frame, text_string, xy, font, scale, color, thickness, line_type)
    
    # .................................................................................................................   
    
    def _enable_following(self):
        current_state = self.state_list[self._state_idx]
        return (current_state in self.follow_states)
    
    # .................................................................................................................


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

class Polygon_Drawer:
    
    # .................................................................................................................
    
    def __init__(self, 
                 frameWH,
                 borderWH = (0, 0), 
                 hover_distance_threshold = 50, 
                 max_zones = 20, 
                 min_points_per_poly = 3,
                 max_points_per_poly = 1000):
        
        # Get proper frame size and scaling
        self.frameWH = np.array(frameWH)
        self._frame_scaling = self.frameWH - np.array((1,1))
        self._changed = True
        
        self.mouse_xy = np.array((0, 0), dtype=np.int32)
        self.borderWH = np.array(borderWH, dtype=np.int32)
        
        self._max_zones = max_zones
        self._min_poly_pts = min_points_per_poly
        self._max_poly_pts = max_points_per_poly
        
        self.zone_hover = None
        self.point_hover = None
        self.zone_select = None
        self.point_select = None
        self.zone_list = None
        self.new_points = None
        
        # Initialize zones and points
        self._new_zone_list()
        self._clear_points_in_progress()
        
        self.hover_distance_sq = hover_distance_threshold**2
        
        self.points_in_progress = False
    
    # .................................................................................................................
    
    def mouse_hover(self):
        
        # Check for the closest point
        minSqDist = 1E9
        
        # Loop through each zone, then loop through every point in each zone
        # Find which point is closest to the mouse
        for zoneIdx, eachZone in enumerate(self.zone_list):
            for pointIdx, eachPoint in enumerate(eachZone):
                
                # Calculate the distance between mouse and point
                distSq = np.sum(np.square(self.mouse_xy - eachPoint))
                
                # Record the closest point
                if distSq < minSqDist:
                    minSqDist = distSq
                    best_match_zone = zoneIdx
                    best_match_point = pointIdx
                    
        # Record the closest zone and point if it's close enough to the mouse
        if minSqDist < self.hover_distance_sq:
            self.zone_hover = best_match_zone
            self.point_hover = best_match_point
        else:
            self.zone_hover = None
            self.point_hover = None
        
    # .................................................................................................................
    
    def left_click(self):
        
        if not self.points_in_progress:                              
            # Select the point nearest to the mouse when left clicking (and not in the middle of drawing new points)
            self.zone_select = self.zone_hover
            self.point_select = self.point_hover
            
    # .................................................................................................................
            
    def left_double_click(self):
        
        if self.points_in_progress:
            if len(self.new_points) >= self._min_poly_pts:
                self._create_zone_from_points()
                self._flag_change()
                
        # Request a return from main callback
        return True

    # .................................................................................................................
    
    def shift_left_click(self):
        
        # Add new point to list
        self.new_points.append(self.mouse_xy)
        
        # If we pass the polygon point limit, create a new zone
        if len(self.new_points) >= self._max_poly_pts:
            self._create_zone_from_points()
            self._flag_change()
       
    # .................................................................................................................
        
    def ctrl_left_click(self):
        
        if not self.points_in_progress:
            # Insert points into existing zones
            pass
    
    # .................................................................................................................
    
    def left_drag(self):
        
        # Update the dragged points based on the mouse position
        if None not in [self.zone_select, self.point_select]:            
            self.zone_list[self.zone_select][self.point_select] = self.mouse_xy
            self._flag_change()
    
    # .................................................................................................................
    
    def left_release(self):
        
        # Unselect points when releasing left click
        self.zone_select = None
        self.point_select = None
        
        # Request a return from main callback
        return True
    
    # .................................................................................................................
    
    def middle_click(self):
        
        pass
    
    # .................................................................................................................
    
    def middle_double_click(self):
        
        pass
    
    # .................................................................................................................
    
    def shift_middle_click(self):
        
        pass
    
    # .................................................................................................................
    
    def ctrl_middle_click(self):
        
        pass
    
    # .................................................................................................................
    
    def middle_drag(self):
        
        pass
    
    # .................................................................................................................
    
    def middle_release(self):
        
        # Request a return from main callback
        return True
    
    # .................................................................................................................
    
    def right_click(self):
        
        if self.points_in_progress:            
            self._clear_points_in_progress()     
        else:
            
            # Clear zone that are moused over, but only if we aren't currently drawing a new region
            new_zone_list = [eachZone for eachZone in self.zone_list
                             if (cv2.pointPolygonTest(eachZone, tuple(self.mouse_xy), measureDist=False) < 0)]
            self._new_zone_list(new_zone_list)
            self._flag_change()
                
            # Clear selections, since the zone indexing could be off
            self.zone_select = None
            self.point_select = None
            self.zone_hover = None
            self.point_hover = None
    
    # .................................................................................................................
    
    def right_double_click(self):
        
        # Clear all zones on double right click
        if not self.points_in_progress:
            self._new_zone_list([])
            self._flag_change()
            
        return True
    
    # .................................................................................................................
    
    def shift_right_click(self):
        
        pass
    
    # .................................................................................................................
    
    def ctrl_right_click(self):
        
        if not self.points_in_progress:
            # Delete nearest point from zone, assuming it doesn't shrink below minimum point count
            
            self._flag_change()
            pass
        pass
    
    # .................................................................................................................
    
    def right_drag(self):
        
        pass
    
    # .................................................................................................................
    
    def right_release(self):
        
        # Request a return from main callback
        return True
    
    # .................................................................................................................
    
    def mouse_move(self):
        
        #print("XY: {}\nEVENT: {}\nFLAGS: {}\n".format(self.mouse_xy, self.event, self.flags))
        pass
    
    # .................................................................................................................
    
    def mouse_check(self, event, mx, my, flags):
        
        # Record events & corrected flag
        self.points_in_progress = len(self.new_points) > 0
        self.flags = flags & 0x1F # Mask numbers >= 32 (alt and numlock both report 32)
        self.event = event
        
        # Record mouse x and y positions at all times for hovering
        self.mouse_xy = np.array((mx, my)) - self.borderWH
        
        # Left mouse events
        self.is_left_click = (self.event == cv2.EVENT_LBUTTONDOWN)
        self.is_left_double_click = (self.event == cv2.EVENT_LBUTTONDBLCLK)
        self.is_left_drag = ((self.flags & cv2.EVENT_FLAG_LBUTTON) > 0)
        self.is_left_release = (self.event == cv2.EVENT_LBUTTONUP)
        
        # Right mouse events
        self.is_right_click = (self.event == cv2.EVENT_RBUTTONDOWN)
        self.is_right_double_click = (self.event == cv2.EVENT_RBUTTONDBLCLK)
        self.is_right_drag = ((self.flags & cv2.EVENT_FLAG_RBUTTON) > 0)
        self.is_right_release = (self.event == cv2.EVENT_RBUTTONUP)
        
        # Middle mouse events
        self.is_middle_click = (self.event == cv2.EVENT_MBUTTONDOWN)
        self.is_middle_double_click = (self.event == cv2.EVENT_MBUTTONDBLCLK)
        self.is_middle_drag = ((self.flags & cv2.EVENT_FLAG_MBUTTON) > 0)
        self.is_middle_release = (self.event == cv2.EVENT_MBUTTONUP)
        
        # Modifiers
        self.is_shift = ((self.flags & cv2.EVENT_FLAG_SHIFTKEY) > 0)
        self.is_ctrl = ((self.flags & cv2.EVENT_FLAG_CTRLKEY) > 0)
        # Alt key is not properly recognized! Gets confused with numlock
        
        # Passive events
        self.is_mouse_move = (self.event == cv2.EVENT_MOUSEMOVE)
        self.is_hovering = (self.is_mouse_move
                            and not self.is_left_click
                            and not self.is_middle_click
                            and not self.is_right_click
                            and not self.points_in_progress)
    
    # .................................................................................................................
    
    def callback(self, event, mx, my, flags, param):
        
        # Figure out what events are occurring
        self.mouse_check(event, mx, my, flags)
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Check for passive events   
        
        if self.is_mouse_move:
            self.mouse_move()
        
        if self.is_hovering:
            self.mouse_hover()
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Check for left click events          
        
        if self.is_left_click:            
            if self.is_shift:
                req_return = self.shift_left_click()
            elif self.is_ctrl:
                req_return = self.ctrl_left_click()
            else:
                req_return = self.left_click()
                
            # Stop if needed
            if req_return: return
            
        if self.is_left_double_click:
            self.left_double_click()
            
        if self.is_left_drag:
            self.left_drag()
            
        if self.is_left_release:
            req_return = self.left_release()
            if req_return: return
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Check for right click events    
        
        if self.is_middle_click:            
            if self.is_shift:
                req_return = self.shift_middle_click()
            elif self.is_ctrl:
                req_return = self.ctrl_middle_click()
            else:
                req_return = self.middle_click()
                
            # Stop if needed
            if req_return: return
            
        if self.is_middle_double_click:
            self.middle_double_click()
            
        if self.is_middle_drag:
            self.middle_drag()
            
        if self.is_middle_release:
            req_return = self.middle_release()
            if req_return: return
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Check for right click events    
        
        if self.is_right_click:            
            if self.is_shift:
                req_return = self.shift_right_click()
            elif self.is_ctrl:
                req_return = self.ctrl_right_click()
            else:
                req_return = self.right_click()
                
            # Stop if needed
            if req_return: return
            
        if self.is_right_double_click:
            self.right_double_click()
            
        if self.is_right_drag:
            self.right_drag()
            
        if self.is_right_release:
            req_return = self.right_release()
            if req_return: return
    
    # .................................................................................................................
    
    def arrow_keys(self, key_press):
        
        # Check for arrow key presses (left: 81, up: 82, right: 83, down: 84)
        arrow_pressed = (80 < key_press < 85)        
        if arrow_pressed:
            
            if None not in (self.zone_hover, self.point_hover):
                multiplier = 1 + 19*int(self.is_shift)
                arrowX = (int(key_press == 83) - int(key_press == 81))*multiplier
                arrowY = (int(key_press == 84) - int(key_press == 82))*multiplier
                self.zone_list[self.zone_hover][self.point_hover] += (arrowX, arrowY)
                self._flag_change()
                
    # .................................................................................................................
    
    def snap_to_border(self, key_press, snap_key = 'b', snap_threshold_distance = 0.05):
        
        # Only snap if the snap key is pressed
        snap_key_pressed = (key_press == ord(snap_key))
        if snap_key_pressed:
            
            # Make sure a point is being hovered (the point that will be snapped)
            if None not in (self.zone_hover, self.point_hover):
                
                # Get hovered point position and frame bounaries for convenience
                point_x, point_y  = self.zone_list[self.zone_hover][self.point_hover]
                frameW, frameH = self._frame_scaling
                
                # Set the threshold for how close a point can be 'inside' the frame and still be snapped
                distance_threshold = 0.05*min(frameW, frameH)
                
                # Check if point is snappable in x
                snapXleft = (point_x < distance_threshold)
                snapXright = (point_x > frameW - distance_threshold)
                if snapXleft: point_x = 0    
                if snapXright: point_x = frameW
                
                # Check if point is snappable in y
                snapYleft = (point_y < distance_threshold)
                snapYright = (point_y > frameH - distance_threshold)
                if snapYleft: point_y = 0    
                if snapYright: point_y = frameH
                
                # Update the quad points if any of the co-ords were snapped
                if any([snapXleft, snapXright, snapYleft, snapYright]):
                    self.zone_list[self.zone_hover][self.point_hover] = np.array((point_x, point_y))
                    self._flag_change()
    
    # .................................................................................................................
    
    def draw_poly_in_progress(self, display_frame, 
                              line_color = (255, 255, 0), 
                              line_thickness = 1, 
                              line_type = cv2.LINE_AA,
                              circle_radius = 5,
                              show_circles = True):
        
        # Don't draw anything if there aren't any new points!
        if len(self.new_points) < 1: return
        
        # Set up points for drawing
        points_and_mouse_list = self.new_points.copy()
        points_and_mouse_list.append(self.mouse_xy)
        draw_points = np.array(points_and_mouse_list, dtype=np.int32) + self.borderWH
        
        # Draw region in-progress
        cv2.polylines(display_frame, [draw_points], True, 
                      line_color, 
                      line_thickness, 
                      line_type)
        
        # Draw circles at all new points (not counting mouse)
        if show_circles:            
            for each_point in draw_points[:-1]:                
                cv2.circle(display_frame, tuple(each_point), circle_radius, line_color, -1, line_type)

    # .................................................................................................................
    
    def draw_zones(self, display_frame, 
                   line_color = (0, 255, 255), 
                   line_thickness = 1, 
                   line_type = cv2.LINE_AA,
                   circle_radius = 5,
                   show_circles = True):
        
        # Don't draw anything if there aren't any zones!
        if len(self.zone_list) < 1: return
        
        # Draw zones
        for each_zone in self.zone_list:
            
            # Add border offsets before drawing
            draw_points = np.int32(each_zone) + self.borderWH
            
            # Draw each zone as a closed polygon
            cv2.polylines(display_frame, 
                          [draw_points], 
                          True, 
                          line_color, 
                          line_thickness, 
                          line_type)
        
            # Draw circles at all points
            if show_circles:      
                for each_point in draw_points:
                    cv2.circle(display_frame, tuple(each_point), circle_radius, line_color, -1, line_type)

    # .................................................................................................................
    
    def add_zone(self, point_list, normalized_input = True):
        if point_list is not None:
            if normalized_input:
                self.zone_list.append(np.int32(np.round(np.array(point_list) * self._frame_scaling)))
            else:
                self.zone_list.append(np.int32(point_list))
            
            # Trigger change flag
            self._flag_change()
            
    # .................................................................................................................
    
    def add_zone_list(self, zone_list, normalized_input = True):
        if zone_list is not None:
            for each_point_list in zone_list:
                self.add_zone(each_point_list, normalized_input)

    # .................................................................................................................

    def add_frame_borders(self, display_frame, border_color = (20,20,20), border_type = cv2.BORDER_CONSTANT):
        # Add borders to the frame for drawing 'out-of-bounds'
        return cv2.copyMakeBorder(display_frame, 
                                  top=self.borderWH[1], 
                                  bottom=self.borderWH[1], 
                                  left=self.borderWH[0],
                                  right=self.borderWH[0],
                                  borderType=border_type,
                                  value=border_color)
    
    # .................................................................................................................
    
    def fetch_zone_list(self, normalize = False, force_frame_boundaries = True):
        
        # Create copy for output (don't want to modify internal zone list)
        output_zone_list = self.zone_list.copy()
        
        # Force values to fit properly in frame, if needed
        if force_frame_boundaries:
            
            min_x, min_y = 0, 0
            max_x, max_y= self._frame_scaling
            
            for zone_idx, each_zone in enumerate(self.zone_list):
                
                for point_idx, each_point in enumerate(each_zone):
                    
                    # Force each point to be within the frame
                    x_pt = self.zone_list[zone_idx][point_idx][0]
                    y_pt = self.zone_list[zone_idx][point_idx][1]
                    output_zone_list[zone_idx][point_idx][0] = min(max_x, max(min_x, x_pt))
                    output_zone_list[zone_idx][point_idx][1] = min(max_y, max(min_y, y_pt))
            
        # Normalize if needed
        if normalize:            
            for zone_idx, each_zone in enumerate(self.zone_list):
                output_zone_list[zone_idx] = each_zone / self._frame_scaling
        
        return list(output_zone_list)
    
    # .................................................................................................................
    
    def changed(self):
        changed_state = self._changed
        self._clear_change_flag()
        return changed_state
    
    # .................................................................................................................
    
    def _flag_change(self, changed = True):
        self._changed = True
        
    # .................................................................................................................
    
    def _clear_change_flag(self):
        self._changed = False
    
    # .................................................................................................................
    
    def _create_zone_from_points(self):
        self.zone_list.append(np.array(self.new_points))
        self._clear_points_in_progress()

    # .................................................................................................................
    
    def _new_zone_list(self, new_list = []):        
        self.zone_list = deque(new_list, maxlen = self._max_zones)
        
    # .................................................................................................................
    
    def _clear_points_in_progress(self):
        self.new_points = deque([], maxlen = self._max_poly_pts)
        
    # .................................................................................................................

# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

class Interactive_Quadrilateral(Polygon_Drawer):
    
    def __init__(self, 
                 frameWH,
                 borderWH = (0, 0), 
                 hover_distance_threshold = 50, 
                 max_zones = 1, 
                 min_points_per_poly = 4,
                 max_points_per_poly = 4,
                 initial_quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]):
        
        # Fill out base polygon drawing object
        super().__init__(frameWH, 
                         borderWH, 
                         hover_distance_threshold, 
                         max_zones, 
                         min_points_per_poly, 
                         max_points_per_poly)
        
        # Add initial quadrilateral zone
        self.add_zone(initial_quad, normalized_input = True)
    
    # .................................................................................................................
    
    # Remove undesirable left click functions
    
    def left_double_click(self): pass    
    def shift_left_click(self): pass    
    def ctrl_left_click(self): pass

    # .................................................................................................................
    
    # Remove undesirable middle click functions
    
    def middle_click(self): pass    
    def middle_double_click(self): pass    
    def shift_middle_click(self): pass    
    def ctrl_middle_click(self): pass
    
    # .................................................................................................................
    
    # Remove undesirable right click functions
    
    def right_click(self): pass    
    def shift_right_click(self): pass    
    def ctrl_right_click(self): pass

    # .................................................................................................................
    
    # Reset quad with double right click
    def right_double_click(self): 
        reset_zone = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.zone_list[0] = np.int32(np.round(np.float32(reset_zone) * self._frame_scaling))
        self._flag_change()
    
    # .................................................................................................................
    
    
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
    
class Interactive_Rectangle(Interactive_Quadrilateral):
    
    def __init__(self, 
                 frameWH,
                 borderWH = (0, 0), 
                 hover_distance_threshold = 50, 
                 max_zones = 1, 
                 min_points_per_poly = 4,
                 max_points_per_poly = 4,
                 initial_rectangle = [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]):
        
        # Fill out base polygon drawing object
        super().__init__(frameWH, 
                         borderWH, 
                         hover_distance_threshold, 
                         max_zones, 
                         min_points_per_poly, 
                         max_points_per_poly,
                         initial_rectangle)
        
        # Add initial rectangular zone
        self.add_zone(initial_rectangle, normalized_input = True)
    
    # .................................................................................................................
    
    # Override the regular point dragging
    def left_drag(self):
        
        # Update the dragged points based on the mouse position
        if None not in [self.zone_select, self.point_select]:            
            self._rectangularize(self.zone_select, self.point_select)
            
    # .................................................................................................................
    
    def fetch_tl_br(self, normalize = False, force_frame_boundaries = True):
        
        # Get the rectangle for easier reference
        rect_zone = self.zone_list[0]
        
        # Get the corner points
        min_x, min_y = np.min(rect_zone, axis = 0)
        max_x, max_y = np.max(rect_zone, axis = 0)
        
        # Pull out the convenient corners for describing the rectangle
        top_left = [min_x, min_y]
        bot_right = [max_x, max_y]
        
        # Force values to fit properly in frame, if needed
        if force_frame_boundaries:
            top_left[0] = min(self._frame_scaling[0], max(0, top_left[0]))
            top_left[1] = max(0, min(self._frame_scaling[1], top_left[1]))
            bot_right[0] = min(self._frame_scaling[0], max(0, bot_right[0]))
            bot_right[1] = max(0, min(self._frame_scaling[1], bot_right[1]))
        
        # Normalize if needed
        if normalize:
            top_left = top_left/self._frame_scaling
            bot_right = bot_right/self._frame_scaling
        
        return tuple(top_left), tuple(bot_right)
    
    # .................................................................................................................
        
    def _rectangularize(self, zone_select, modified_point_index):
    
        # Get the two opposing corners from the quadrilateral
        opposite_index = (modified_point_index + 2) % 4
        opposite_corner = self.zone_list[zone_select][opposite_index]
        far_corner_list = [self.mouse_xy, opposite_corner]
        
        # Figure out the bounding rectangle based on the corner points
        min_x, min_y = np.min(far_corner_list, axis = 0)
        max_x, max_y = np.max(far_corner_list, axis = 0)
        
        # Build the new quad corner points
        tl = (min_x, min_y)
        tr = (max_x, min_y)
        br = (max_x, max_y)
        bl = (min_x, max_y)
        rect_zone = np.int32([tl, tr, br, bl])
        
        # Update closest point selection to allow for cross-overs
        self.point_select = np.argmin(np.sqrt(np.sum(np.square(rect_zone - self.mouse_xy), axis=1)))
        
        # Replace the existing zone with updated rectangular zone
        self.zone_list[zone_select] = rect_zone
        self._flag_change()
    
    # .................................................................................................................
        
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
    
class Interactive_Line(Polygon_Drawer):
    
    def __init__(self, 
                 frameWH,
                 borderWH = (0, 0), 
                 hover_distance_threshold = 50, 
                 max_zones = 1, 
                 min_points_per_poly = 2,
                 max_points_per_poly = 2,
                 initial_line = [(0.15, 0.5), (0.85, 0.5)]):
        
        # Fill out base polygon drawing object
        super().__init__(frameWH, 
                         borderWH, 
                         hover_distance_threshold, 
                         max_zones, 
                         min_points_per_poly, 
                         max_points_per_poly)
        
        # Add initial line
        self.add_zone(initial_line, normalized_input = True)
        
    # .................................................................................................................
    
    # Remove undesirable left click functions
    
    def left_double_click(self): pass    
    def shift_left_click(self): pass    
    def ctrl_left_click(self): pass

    # .................................................................................................................
    
    # Remove undesirable middle click functions
    
    def middle_click(self): pass    
    def middle_double_click(self): pass    
    def shift_middle_click(self): pass    
    def ctrl_middle_click(self): pass
    
    # .................................................................................................................
    
    # Remove undesirable right click functions
    
    def right_click(self): pass    
    def shift_right_click(self): pass    
    def ctrl_right_click(self): pass

    # .................................................................................................................
    
    # Reset quad with double right click
    def right_double_click(self): 
        reset_zone = [(0.15, 0.5), (0.85, 0.5)]
        self.zone_list[0] = np.int32(np.round(np.float32(reset_zone) * self._frame_scaling))
        self._flag_change()
    
    # .................................................................................................................
    
    pass
    
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

class Button_Grid:
    
    
    def __init__(self, buttonWH, num_rows, num_cols, 
                 padding = 5, 
                 bg_color = (0, 0, 0),
                 hover_color = (200, 200, 200),
                 click_color = (255, 255, 255)):
        
        # Make sure we have sane values
        buttonWH = (int(round(buttonWH[0])), int(round(buttonWH[1])))
        num_rows = max(1, num_rows)
        num_cols = max(1, num_cols)
        
        self._buttonWH = buttonWH
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._padding = padding
        
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._click_color = click_color
        
        self._button_row_col_indices = {}
        self._button_configs = {}
        self._button_tlbr_record = {}        
        self._button_hover_record = {}
        self._button_pressed_events = {}

        # Get button frame sizing
        frame_w = (buttonWH[0] * num_cols) + (num_cols + 1)*padding
        frame_h = (buttonWH[1] * num_rows) + (num_rows + 1)*padding
        self.frameWH = (frame_w, frame_h)
        
        # Create blank frame for drawing buttons
        self._blank_frame = np.full((frame_h, frame_w, 3), bg_color, dtype=np.uint8)
        
        # Allocate space for an image with buttons drawn in to it and storage for each button image
        self._button_frame = self._blank_frame.copy()
        self._button_images = {}
        
        # Store default config of all text
        self._text_config = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                             "fontScale": 0.75,
                             "thickness": 1}
        
    # .................................................................................................................

    def add_button(self, button_label, row_index = 0, col_index = 0,
                   button_color = (80,80,80),
                   text_color = (230, 230, 230),
                   draw_text_shadow = True):
        
        if (1 + row_index > self._num_rows):
            raise ValueError("Cannot index row {}, only {} rows available!".format(row_index, self._num_rows))
            
        if (1 + col_index > self._num_cols):
            raise ValueError("Cannot index column {}, only {} columns available!".format(col_index, self._num_cols))
            
        # First get the button bounding box
        tlx = self._padding*(1 + col_index) + self._buttonWH[0]*(col_index)
        tly = self._padding*(1 + row_index) + self._buttonWH[1]*(row_index)
        
        brx = tlx + self._buttonWH[0]
        bry = tly + self._buttonWH[1]
        
        # Add button bounding box to record
        self._button_tlbr_record[button_label] = [(tlx, tly), (brx, bry)]
        
        # Add button label to press event dictionary
        self._button_pressed_events[button_label] = False
        self._button_hover_record[button_label] = False
        
        # Draw button image and store it for use later
        b_w, b_h = self._buttonWH
        button_image = np.full((b_h, b_w, 3), button_color, dtype=np.uint8)        
        label_conf = self._get_label_config(button_label, text_color)
        
        # Draw text, with shadow if desired
        if draw_text_shadow:
            shadow_conf = label_conf.copy()
            shadow_conf["color"] = [255 - each_col for each_col in text_color]
            shadow_conf["thickness"] = 2
            cv2.putText(button_image, **shadow_conf)
        cv2.putText(button_image, **label_conf)
        
        # Store button index and config settings
        self._button_row_col_indices[button_label] = (row_index, col_index)
        self._button_configs[button_label] = {"button_color": button_color, 
                                              "text_color": text_color, 
                                              "draw_text_shadow": draw_text_shadow}
        
        # Store button image
        self._button_images[button_label] = button_image
        
        # Draw button into button frame
        self._button_frame[tly:bry, tlx:brx, :] = button_image
        
    # .................................................................................................................
    
    def rename_button(self, old_label, new_label, 
                      new_button_color = None,
                      new_text_color = None,
                      new_draw_text_shadow = None):
        
        
        row_idx, col_idx = self._button_row_col_indices[old_label]
        old_config = self._button_configs[old_label]
        
        new_config = {}
        if new_button_color is None:
            new_config["button_color"] = old_config["button_color"]
        if new_text_color is None:
            new_config["text_color"] = old_config["text_color"]
        if new_draw_text_shadow is None:
            new_config["draw_text_shadow"] = old_config["draw_text_shadow"]
        
        self.remove_button(old_label)
        
        self.add_button(new_label, 
                        row_index = row_idx,
                        col_index = col_idx,
                        **new_config)
    
    # .................................................................................................................
    
    def remove_button(self, button_label = None, row_index = None, col_index = None):
        
        # Can supply either a button label or row/col indices to get button
        if button_label is None and row_index is None and col_index is None:
            raise AttributeError("Must supply either a button label or row/col indices to delete a button!")
        
        # Use row/col indices to select button, if provided
        if row_index is not None and col_index is not None:
            button_label = self._button_row_col_indices[button_label]            
        
        # Erase button image from the button frame
        (tlx, tly), (brx, bry) = self._button_tlbr_record[button_label]
        self._button_frame[tly:bry, tlx:brx, :] = self._blank_frame[tly:bry, tlx:brx, :] 
        
        # Delete all recorded info about button
        del self._button_tlbr_record[button_label]
        del self._button_pressed_events[button_label]
        del self._button_hover_record[button_label]
        del self._button_row_col_indices[button_label]
        del self._button_images[button_label]
    
    # .................................................................................................................
    
    def callback(self, event, mx, my, flags, param):
        
        left_clicked = (event == cv2.EVENT_LBUTTONDOWN)
        for each_button, each_tlbr in self._button_tlbr_record.items():        
            (tlx, tly), (brx, bry) = each_tlbr       
            
            # Figure out in the mouse is hovering over the button
            in_x = (tlx < mx < brx)
            in_y = (tly < my < bry)
            hovering = (in_x and in_y)
            
            # Record hovering and click events
            self._button_hover_record[each_button] = hovering            
            self._button_pressed_events[each_button] = left_clicked and hovering
                
    # .................................................................................................................
    
    def draw_buttons(self):
        
        # Get copy of button frame so we can draw without ruining the original image
        button_frame = self._button_frame.copy()
        
        for each_button, each_image in self._button_images.items():
            
            # Get button bounding boxes so we can draw stuff
            tl, br = self._button_tlbr_record[each_button]            
            
            # Get each button bounding box so we can draw outlines
            if self._button_hover_record[each_button]: 
                cv2.rectangle(button_frame, tl, br, self._hover_color, 1, cv2.LINE_AA)
                
            if self._button_pressed_events[each_button]:          
                cv2.rectangle(button_frame, tl, br, self._click_color, -1, cv2.LINE_AA)
        
        return button_frame
    
    # .................................................................................................................
    
    def button_pressed(self, button_label, clear_on_read = True, error_if_missing = True):
        
        if button_label not in self._button_pressed_events:
            if error_if_missing:
                raise KeyError("Button '{}' is not in button grid!".format(button_label))
            else:
                return False
        
        button_is_pressed = self._button_pressed_events[button_label]        
        if clear_on_read:
            self._button_pressed_events[button_label] = False        
        return button_is_pressed
        
    # .................................................................................................................
    
    def read_all_buttons(self, clear_on_read = True):
        button_states = self._button_pressed_events.copy()
        
        if clear_on_read:
            self._button_pressed_events = {each_key: False for each_key in self._button_pressed_events.keys()}
        return button_states

    # .................................................................................................................
    
    def _get_label_config(self, button_label, text_color = (230, 230, 230)):
        
        text_is_too_big = True
        text_config = self._text_config.copy()
        while text_is_too_big:
            
            # OpenCV: cv2.getTextSize(text, font, scale, thickness)
            text_xy, text_base = cv2.getTextSize(button_label, **text_config)
            
            # Stop if we can fit the text in the button
            if ((text_xy[0] + 5) < self._buttonWH[0]) and ((text_xy[1] + 5) < self._buttonWH[1]):
                text_is_too_big = False
            
            # If we can't fit the text in the button, shrink it a bit and try again
            if text_is_too_big:
                text_config["fontScale"] -= 0.25
            
            # Stop text from shrinking forever
            if text_config["fontScale"] <= 0.15:
                text_config["fontScale"] = 0.15
                text_is_too_big = False
                
        # Get text positioning
        text_x = int(round(self._buttonWH[0] - text_xy[0])/2)
        text_y = int(round(self._buttonWH[1]/2)) + text_base
            
        # Add remaining config data
        text_config["text"] = button_label
        text_config["org"] = (text_x, text_y)
        text_config["color"] = text_color
        text_config["lineType"] = cv2.LINE_AA
        
        return text_config
    
    # .................................................................................................................
    
    # .................................................................................................................
    
    # .................................................................................................................

# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


# ---------------------------------------------------------------------------------------------------------------------
#%% Useful functions
    
# .....................................................................................................................
    
def crop_drawer_util(video_obj_ref, 
                     initial_crop_tl_br = None, 
                     borderWH = (40, 40), 
                     displayWH = None, 
                     normalize_output = True,
                     window_title = "Crop Frame"):
    
    _add_eolib_to_path()
    from eolib.video.windowing import SimpleWindow, breakByKeypress
    
    # Get important video parameters
    initial_frame_position = video_obj_ref.get(cv2.CAP_PROP_POS_FRAMES)
    vid_width = int(video_obj_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(video_obj_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = video_obj_ref.get(cv2.CAP_PROP_FPS)
    vid_frame_delay_ms = 1000/vid_fps
    frame_delay = max(1, min(1000, int(vid_frame_delay_ms)))
    
    # Set the initial cropping area if needed
    if initial_crop_tl_br is None:
        initial_rectangle = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    else:
        (left_x, top_y), (right_x, bot_y) = initial_crop_tl_br
        initial_rectangle = [(left_x, top_y), (right_x, top_y), (right_x, bot_y), (left_x, bot_y)]
    
    # Figure out frame (re-)sizing
    resize_frame = (displayWH is not None)
    frameWH = displayWH if resize_frame else (vid_width, vid_height)
        
    # Build cropping object to get drawing callback
    cropper = Interactive_Rectangle(frameWH = frameWH, 
                                    borderWH = borderWH, 
                                    initial_rectangle=initial_rectangle)
    
    # Create window for display and attach cropping callback function
    cropWindow = SimpleWindow(window_title)
    cropWindow.attachCallback(cropper.callback)
    
    # Video loop
    while True:
        
        (rec, inFrame) = video_obj_ref.read()
        if not rec:
            video_obj_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Resize the frame if needed
        if resize_frame:
            inFrame = cv2.resize(inFrame, dsize = frameWH)
        
        # Add borders to help with drawing out of bounds
        inFrame = cropper.add_frame_borders(inFrame)
        
        cropper.draw_zones(inFrame, show_circles=False)
        winExists = cropWindow.imshow(inFrame)
        if not winExists:
            break
        
        # Get keypresses
        reqBreak, keyPress = breakByKeypress(frame_delay, break_on_enter = True)
        if reqBreak:
            break
    
    # Clean up. Reset video back to initial frame and close windows
    video_obj_ref.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_position)
    cv2.destroyAllWindows()
    
    return cropper.fetch_tl_br(normalize = normalize_output)

# .....................................................................................................................
    
def multizone_drawer_util(video_obj_ref, 
                          max_zones = 100,
                          initial_zone_list = None, 
                          borderWH = (40, 40), 
                          displayWH = None, 
                          normalize_output = True,
                          show_zone_circles = True,
                          window_title = "Drawing Frame"):
    
    _add_eolib_to_path()
    from eolib.video.windowing import SimpleWindow, breakByKeypress
    
    # Get important video parameters
    initial_frame_position = video_obj_ref.get(cv2.CAP_PROP_POS_FRAMES)
    vid_width = int(video_obj_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(video_obj_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = video_obj_ref.get(cv2.CAP_PROP_FPS)
    vid_frame_delay_ms = 1000/vid_fps
    frame_delay = max(1, min(1000, int(vid_frame_delay_ms)))
    
    # Figure out frame (re-)sizing
    resize_frame = (displayWH is not None)
    frameWH = displayWH if resize_frame else (vid_width, vid_height)
        
    # Build cropping object to get drawing callback
    zone_drawer = Polygon_Drawer(frameWH = frameWH, 
                                 borderWH = borderWH,
                                 max_zones = max_zones)
    
    # Add any initial zones
    zone_drawer.add_zone_list(initial_zone_list)
    
    # Create window for display and attach cropping callback function
    zoneWindow = SimpleWindow(window_title)
    zoneWindow.attachCallback(zone_drawer.callback)
    
    # Video loop
    while True:
        
        (rec, inFrame) = video_obj_ref.read()
        if not rec:
            video_obj_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Resize the frame if needed
        if resize_frame:
            inFrame = cv2.resize(inFrame, dsize = frameWH)
        
        # Add borders to help with drawing out of bounds
        inFrame = zone_drawer.add_frame_borders(inFrame)
        
        zone_drawer.draw_poly_in_progress(inFrame)
        zone_drawer.draw_zones(inFrame, show_circles=show_zone_circles)
        winExists = zoneWindow.imshow(inFrame)
        if not winExists:
            break
        
        # Get keypresses
        reqBreak, keyPress = breakByKeypress(frame_delay, break_on_enter = True)
        if reqBreak:
            break
        
        # Nudge mask points with arrow keys 
        zone_drawer.arrow_keys(keyPress)
        
    
    # Clean up. Reset video back to initial frame and close windows
    video_obj_ref.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_position)
    cv2.destroyAllWindows()
    
    return zone_drawer.fetch_zone_list(normalize=normalize_output, force_frame_boundaries=False)

# .....................................................................................................................
    
def _add_eolib_to_path():
    
    import os
    import sys        
        
    # Use the location of this file as the starting search path. The root of eolib is a few folders backwards from here
    search_directory = os.path.dirname(os.path.abspath(__file__))
    for k in range(5):
        if "eolib" in os.listdir(search_directory):
            if search_directory not in sys.path:
                sys.path.append(search_directory)
            break
        search_directory = os.path.dirname(search_directory)
        
# .....................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Demo

if __name__ == "__main__":
    
    '''
    video_source = "/home/eo/Desktop/PythonData/Shared/videos/pl_part1_rot720.mp4"
    videoObj = cv2.VideoCapture(video_source)
    #crop_tl_br = crop_drawer_util(videoObj, displayWH=(320, 180))
    
    ini_list = [[[ 0.45454545,  0.37988827], [-0.06583072,  0.36871508], [-0.00626959, -0.06145251], [ 0.67398119, -0.06145251]]]
                
    #zone_list = multizone_drawer_util(videoObj, displayWH=(320,180), initial_zone_list=ini_list)
    
    one_zone_list = singlezone_drawer_util(videoObj, displayWH=(640, 360))

    videoObj.release()
    cv2.destroyAllWindows()
    '''
    
    
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Button grid demo
    
    cv2.destroyAllWindows()    
    
    # Build button grid object
    btn_grid = Button_Grid((120, 40), 2, 3,)
    btn_grid.add_button("ABC", 0, 0, (255, 0, 0))
    btn_grid.add_button("Hello", 0, 1, (0, 255, 0), (0, 0, 255))
    btn_grid.add_button("Goodbye", 1, 2, (200, 200, 200), (20, 20, 20))
    btn_grid.add_button("Close", 1, 1)
    
    # Add button callback to window
    win_name = "Button Demo"
    demo_window = cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, btn_grid.callback, {})
    
    for k in range(500):
        
        btn_img = btn_grid.draw_buttons()
        
        cv2.imshow(win_name, btn_img)
        
        # Read a specific button
        if btn_grid.button_pressed("Close"):
            cv2.destroyAllWindows()
            break
        
        # Read all buttons (as a dictionary with values of True/False depending on press state)
        for each_button, each_state in btn_grid.read_all_buttons().items():
            if each_state:
                print("Button pressed:", each_button)
                
        cv2.waitKey(20)
        
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .