# occupancy_grid.py
import numpy as np
import math
from threading import RLock
import time

class OccupancyGrid:
    def __init__(self, field_size_cm, cell_size_cm):
        self.field_size_cm = field_size_cm
        self.cell_size_cm = cell_size_cm
        self.grid_rows = int(field_size_cm / cell_size_cm)
        self.grid_cols = int(field_size_cm / cell_size_cm)
        self.log_odds_grid = np.full((self.grid_rows, self.grid_cols), -2.0, dtype=float)
        self.robot_x = field_size_cm / 2.0  # Start at center
        self.robot_y = 4 * field_size_cm / 5.0
        self.yaw = 0.0      # Yaw angle in degrees
        self.latest_forward = None  # Latest forward distance for person tracking
        self.previous_distances = {}  # To detect movement
        self.lock = RLock()
        
        # Person tracking
        self.person_markers = []  # List to store all detected persons
        self.person_cells = set()  # Set of (row, col) cells that contain persons
        self.current_detected_ids = set()  # Track IDs currently being detected
        
        # Tuning parameters
        self.log_odds_occupied_inc = 0.9
        self.log_odds_free_dec = 0.4
        self.max_log_odds = 5.0
        self.min_log_odds = -5.0
        self.tolerance = 5.0  # cm threshold to detect movement
        self.max_speed_cm_s = 50.0
        
        # Default movement speed in cm/s (0.5334 m/s = 53.34 cm/s)
        self.default_speed_cm_s = 53.34

    def clear_all_person_markers(self):
        """Clear all person markers from the grid"""
        with self.lock:
            # Reset all person cells to unoccupied
            for row, col in self.person_cells:
                self.log_odds_grid[row, col] = -2.0  # Reset to initial value
            
            # Clear person markers and cells
            self.person_markers = []
            self.person_cells = set()
            self.current_detected_ids = set()  # Also clear the set of currently detected IDs

    def clamp_log_odds(self, value):
        return max(self.min_log_odds, min(self.max_log_odds, value))

    def update_cell_log_odds(self, row, col, delta):
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            with self.lock:
                # Skip updating the cell if it contains a person
                if (row, col) in self.person_cells:
                    return
                
                current = self.log_odds_grid[row, col]
                new_val = self.clamp_log_odds(current + delta)
                self.log_odds_grid[row, col] = new_val

    def world_to_grid(self, x_cm, y_cm):
        row = int(y_cm // self.cell_size_cm)
        col = int(x_cm // self.cell_size_cm)
        return max(0, min(self.grid_rows - 1, row)), max(0, min(self.grid_cols - 1, col))

    def get_robot_cell(self):
        return self.world_to_grid(self.robot_x, self.robot_y)

    def log_odds_to_probability(self):
        with self.lock:
            return 1.0 / (1.0 + np.exp(-self.log_odds_grid))

    def bresenham_line(self, start_row, start_col, end_row, end_col):
        """Returns list of (row, col) cells along the line."""
        r0, c0 = start_row, start_col
        r1, c1 = end_row, end_col
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        cells = []
        while True:
            cells.append((r0, c0))
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
        return cells

    def mark_ray(self, start_row, start_col, angle_deg, distance_cells):
        """Mark cells along a ray based on angle and distance."""
        angle_rad = math.radians(angle_deg)
        d_col = math.sin(angle_rad)  # Yaw=0 is up (+y), so sin is x-component
        d_row = -math.cos(angle_rad) # -cos because row increases downward
        end_col = start_col + distance_cells * d_col
        end_row = start_row + distance_cells * d_row
        end_col_int = round(end_col)
        end_row_int = round(end_row)
        cells = self.bresenham_line(start_row, start_col, end_row_int, end_col_int)
        for i, (r, c) in enumerate(cells):
            if (r, c) in self.person_cells:
                # Skip cells with persons - don't update them
                continue
                
            if i < len(cells) - 1:
                self.update_cell_log_odds(r, c, -self.log_odds_free_dec)
            else:
                self.update_cell_log_odds(r, c, self.log_odds_occupied_inc)

    def update_robot_position(self, distances, dt, move_command="stop"):
        with self.lock:
            # Only compare if every key in the current readings exists in previous measurements.
            if all(key in self.previous_distances for key in distances) and move_command not in ["forward", "reverse"]:
                if all(abs(distances[key] - self.previous_distances[key]) < self.tolerance for key in distances):
                    return  # No significant change; assume the robot is stationary.

            # Retrieve sensor distances.
            f = distances.get('forward')
            b = distances.get('back')
            r = distances.get('right')
            l = distances.get('left')

            # Check if each sensor reading is valid.
            valid_f = f is not None and f <= 500 and f > self.cell_size_cm
            valid_b = b is not None and b <= 500 and b > self.cell_size_cm
            valid_r = r is not None and r <= 500 and r > self.cell_size_cm
            valid_l = l is not None and l <= 500 and l > self.cell_size_cm

            # Start with the current robot position.
            new_x = self.robot_x
            new_y = self.robot_y
            
            # ONLY update position if we're explicitly moving forward or reverse
            if move_command in ["forward", "reverse"]:
                # Calculate displacement based on default speed and time elapsed
                displacement = self.default_speed_cm_s * dt
                
                # Calculate new position based on move command and yaw
                angle_rad = math.radians(self.yaw)
                
                if move_command == "forward":
                    # Move in the direction of yaw
                    new_x -= displacement * math.sin(angle_rad)
                    new_y += displacement * math.cos(angle_rad)
                    print(f"Moving forward: dx={-displacement * math.sin(angle_rad)}, dy={displacement * math.cos(angle_rad)}")
                elif move_command == "reverse":
                    # Move opposite to the direction of yaw
                    new_x += displacement * math.sin(angle_rad)
                    new_y -= displacement * math.cos(angle_rad)  # Negative because row 0 is top
                    print(f"Moving reverse: dx={displacement * math.sin(angle_rad)}, dy={-displacement * math.cos(angle_rad)}")
                    
                # Clamp new positions within the field.
                new_x = max(0, min(self.field_size_cm, new_x))
                new_y = max(0, min(self.field_size_cm, new_y))

                # Limit the position update to the maximum allowed speed.
                dx = new_x - self.robot_x
                dy = new_y - self.robot_y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist <= self.max_speed_cm_s * dt:
                    self.robot_x = new_x
                    self.robot_y = new_y
                    print(f"New position: ({self.robot_x}, {self.robot_y})")
                else:
                    # Scale the movement to respect max speed
                    scale_factor = (self.max_speed_cm_s * dt) / dist
                    self.robot_x += dx * scale_factor
                    self.robot_y += dy * scale_factor
                    print(f"Speed limited position: ({self.robot_x}, {self.robot_y})")
            else:
                # If move_command is not "forward" or "reverse", don't update position
                print(f"Not moving: keeping position at ({self.robot_x}, {self.robot_y})")

            # Store the current distances for future comparisons.
            self.previous_distances = distances.copy()
            
    # In OccupancyGrid class, optimize update frequency with a max_updates_per_second limiter
    def update_from_sensors(self, distances, yaw, dt, move_command="stop"):
        # Add rate limiting
        current_time = time.time()
        if hasattr(self, 'last_update_time') and current_time - self.last_update_time < 0.02:  # Max 50 updates/second
            return  # Skip this update if too soon after the last one
        
        self.last_update_time = current_time
        
        with self.lock:
            self.yaw = yaw
            self.latest_forward = distances.get('forward') if (distances.get('forward', 501) <= 500) else None
            
            # Always update robot position regardless of move_command
            self.update_robot_position(distances, dt, move_command)
            
            # Only if we're not moving, update the grid with sensor information
            start_row, start_col = self.get_robot_cell()
            for sensor_name, distance in distances.items():
                if sensor_name.endswith('_angle'):
                    continue
                if distance > 500 or distance <= self.cell_size_cm:
                    continue
                distance_cells = distance / self.cell_size_cm
                absolute_angle = distances[sensor_name + '_angle']
                self.mark_ray(start_row, start_col, absolute_angle, distance_cells)

    def add_person_marker(self, grid_row, grid_col, status, track_id=None):
        """Add or update a person marker on the grid"""
        with self.lock:
            # Only update if we have a track_id (which means we can actually see the person)
            if track_id is not None:
                # Check if this track_id already exists in any marker
                existing_marker_index = None
                for i, marker in enumerate(self.person_markers):
                    if marker.get("track_id") == track_id:
                        existing_marker_index = i
                        break
                
                if existing_marker_index is not None:
                    # Found the same person - update their position
                    marker = self.person_markers[existing_marker_index]
                    old_row = marker["grid_row"]
                    old_col = marker["grid_col"]
                    
                    # Only update position if this track_id is actually being detected by the camera now
                    # This prevents updating position of non-visible people when the robot rotates
                    if track_id in self.current_detected_ids:
                        # Check if position changed
                        if old_row != grid_row or old_col != grid_col:
                            # Remove old position from person cells
                            if (old_row, old_col) in self.person_cells:
                                self.person_cells.remove((old_row, old_col))
                            
                            # Reset old position in log_odds_grid
                            if 0 <= old_row < self.grid_rows and 0 <= old_col < self.grid_cols:
                                self.log_odds_grid[old_row, old_col] = -2.0  # Reset to initial unoccupied value
                                print(f"Reset log_odds_grid at old position ({old_row},{old_col})")
                        
                        # Determine status (conscious/unconscious logic)
                        new_status = status
                        if marker["status"] == "Unconscious" and status == "Conscious":
                            # Keep as unconscious until we're sure they're conscious
                            if marker.get("conscious_counter", 0) < 10:
                                marker["conscious_counter"] = marker.get("conscious_counter", 0) + 1
                                new_status = "Unconscious"
                            else:
                                new_status = "Conscious"
                        elif marker["status"] == "Conscious" and status == "Unconscious":
                            # Immediately mark as unconscious
                            new_status = "Unconscious"
                            marker["conscious_counter"] = 0
                        
                        # Update the marker with new position
                        self.person_markers[existing_marker_index] = {
                            "grid_row": grid_row,
                            "grid_col": grid_col,
                            "status": new_status,
                            "timestamp": time.time(),
                            "track_id": track_id,
                            "conscious_counter": marker.get("conscious_counter", 0)
                        }
                        
                        # Add new position to person cells
                        self.person_cells.add((grid_row, grid_col))
                        
                        # Mark this cell as occupied in the log odds grid
                        self.log_odds_grid[grid_row, grid_col] = self.max_log_odds
                        print(f"Updated marker for track_id {track_id} to position ({grid_row},{grid_col})")
                    else:
                        # This person is in our records but not currently visible
                        # Just update the status if needed, but keep the old position
                        print(f"Track_id {track_id} not in current detections, keeping at ({old_row},{old_col})")
                else:
                    # If no existing marker with this track_id, add a new one
                    self.person_markers.append({
                        "grid_row": grid_row, 
                        "grid_col": grid_col, 
                        "status": status,
                        "timestamp": time.time(),
                        "track_id": track_id,
                        "conscious_counter": 0
                    })
                    
                    # Add to person cells set to prevent overwriting
                    self.person_cells.add((grid_row, grid_col))
                    
                    # Mark this cell as occupied in the log odds grid
                    self.log_odds_grid[grid_row, grid_col] = self.max_log_odds
                    print(f"Added new marker for track_id {track_id} at position ({grid_row},{grid_col})")
    
    def update_person_markers(self):
        """Update person markers - people not currently detected should remain in place"""
        with self.lock:
            updated_cells = set()
            
            for marker in self.person_markers:
                grid_row = marker["grid_row"] 
                grid_col = marker["grid_col"]
                updated_cells.add((grid_row, grid_col))
                
                # Ensure the cell is still marked as occupied in the log_odds_grid
                if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                    self.log_odds_grid[grid_row, grid_col] = self.max_log_odds
            
            # Update the person_cells set
            self.person_cells = updated_cells

    def get_grid_data(self):
        """Return data for visualization."""
        with self.lock:
            # Update person markers before returning data
            self.update_person_markers()
            
            return {
                "grid": self.log_odds_to_probability().tolist(),
                "robot_x": self.robot_x,
                "robot_y": self.robot_y,
                "robot_row": self.get_robot_cell()[0],
                "robot_col": self.get_robot_cell()[1],
                "yaw": self.yaw,
                "person_markers": self.person_markers  # Include the persistent markers
            }
            
    def reset_position(self):
        """Reset robot position to center of the field, reset yaw to 0 degrees, and clear the occupancy grid and person detections."""
        with self.lock:
            # Reset robot position and orientation
            self.robot_x = self.field_size_cm / 2.0
            self.robot_y = 4 * self.field_size_cm / 5.0
            self.yaw = 0.0
            self.latest_forward = None
            self.previous_distances = {}  # Reset distance history
            
            # Clear all person markers
            self.clear_all_person_markers()
            
            # Reset the occupancy grid data - set all cells back to the initial log odds value
            self.log_odds_grid = np.full((self.grid_rows, self.grid_cols), -2.0, dtype=float)
            
            print(f"[INFO] Robot position reset to ({self.robot_x}, {self.robot_y}) with yaw=0 and occupancy grid cleared including all person markers")