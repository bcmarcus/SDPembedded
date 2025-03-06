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



    def get_grid_data(self):
        """Return data for visualization."""
        with self.lock:
            return {
                "grid": self.log_odds_to_probability().tolist(),
                "robot_x": self.robot_x,
                "robot_y": self.robot_y,
                "robot_row": self.get_robot_cell()[0],
                "robot_col": self.get_robot_cell()[1],
                "yaw": self.yaw
            }