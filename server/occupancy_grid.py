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
        self.robot_y = field_size_cm / 2.0
        self.yaw = 0.0      # Yaw angle in degrees
        self.latest_forward = None  # Latest forward distance for person tracking
        self.previous_distances = {}  # To detect movement
        self.lock = RLock()
        
        # New: Add persistent person markers list
        self.person_markers = []  # List to store all detected persons
        self.person_cells = set()  # Set of (row, col) cells that contain persons
        
        # Tuning parameters
        self.log_odds_occupied_inc = 0.9
        self.log_odds_free_dec = 0.4
        self.max_log_odds = 5.0
        self.min_log_odds = -5.0
        self.tolerance = 5.0  # cm threshold to detect movement
        self.max_speed_cm_s = 50.0

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

    def update_robot_position(self, distances, dt):
        with self.lock:
            # Only compare if every key in the current readings exists in previous measurements.
            if all(key in self.previous_distances for key in distances):
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

            # Update Y position based on forward and back sensor readings.
            if valid_f and valid_b:
                new_y = ((self.field_size_cm - f) + b) / 2.0
            elif valid_f:
                new_y = self.field_size_cm - f
            elif valid_b:
                new_y = b

            # Update X position based on right and left sensor readings.
            if valid_r and valid_l:
                new_x = ((self.field_size_cm - r) + l) / 2.0
            elif valid_r:
                new_x = self.field_size_cm - r
            elif valid_l:
                new_x = l

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
            # Only update x,y position if a translation command is active.
            
            if move_command in ["forward", "reverse"]:
                self.update_robot_position(distances, dt)
            
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
            # First check if this track_id already exists in any marker
            if track_id is not None:
                for i, marker in enumerate(self.person_markers):
                    if marker.get("track_id") == track_id:
                        # Found the same person - update their position
                        old_row = marker["grid_row"]
                        old_col = marker["grid_col"]
                        
                        # Remove old position from person cells
                        if (old_row, old_col) in self.person_cells:
                            self.person_cells.remove((old_row, old_col))
                            
                        # Only change status if it's becoming unconscious or if it was previously unknown
                        new_status = status
                        if marker["status"] == "Unconscious" and status == "Conscious":
                            # Keep the unconscious status unless they've been conscious for a while
                            if marker.get("conscious_counter", 0) < 10:  # Require 10 consecutive conscious detections
                                marker["conscious_counter"] = marker.get("conscious_counter", 0) + 1
                                new_status = "Unconscious"  # Keep as unconscious until we're sure
                            else:
                                new_status = "Conscious"  # Finally change to conscious
                        elif marker["status"] == "Conscious" and status == "Unconscious":
                            # Immediately mark as unconscious
                            new_status = "Unconscious"
                            marker["conscious_counter"] = 0
                        
                        # Update the marker
                        self.person_markers[i] = {
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
                        return
            
            # Check if a marker already exists at this position (without track_id)
            for i, marker in enumerate(self.person_markers):
                if marker["grid_row"] == grid_row and marker["grid_col"] == grid_col:
                    # If there's no track_id, just update the status
                    if marker["status"] != status:
                        self.person_markers[i]["status"] = status
                    if track_id is not None:
                        self.person_markers[i]["track_id"] = track_id
                    self.person_markers[i]["timestamp"] = time.time()
                    return
            
            # If no existing marker, add a new one
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
            # Use a high value to ensure it stays occupied
            self.log_odds_grid[grid_row, grid_col] = self.max_log_odds

    def update_person_markers(self):
        """Remove expired person markers (only for conscious persons)"""
        current_time = time.time()
        with self.lock:
            updated_markers = []
            updated_cells = set()
            
            for marker in self.person_markers:
                # Keep unconscious persons indefinitely
                if marker["status"] == "Unconscious":
                    updated_markers.append(marker)
                    updated_cells.add((marker["grid_row"], marker["grid_col"]))
                # For conscious persons, keep only recent ones (within last 30 seconds)
                elif current_time - marker.get("timestamp", 0) < 30:
                    updated_markers.append(marker)
                    updated_cells.add((marker["grid_row"], marker["grid_col"]))
            
            self.person_markers = updated_markers
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
        """Reset robot position to center of the field, reset yaw to 0 degrees, and clear the occupancy grid."""
        with self.lock:
            # Reset robot position and orientation
            self.robot_x = self.field_size_cm / 2.0
            self.robot_y = self.field_size_cm / 2.0
            self.yaw = 0.0
            self.latest_forward = None
            self.previous_distances = {}  # Reset distance history
            
            # Reset the occupancy grid data - set all cells back to the initial log odds value
            # But preserve person marker cells
            new_grid = np.full((self.grid_rows, self.grid_cols), -2.0, dtype=float)
            
            # Restore the person cells to max occupancy
            for row, col in self.person_cells:
                new_grid[row, col] = self.max_log_odds
                
            self.log_odds_grid = new_grid
            
            print(f"[INFO] Robot position reset to ({self.robot_x}, {self.robot_y}) with yaw=0 and occupancy grid cleared while preserving person markers")