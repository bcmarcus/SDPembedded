# ultrasonic_sensor.py

class UltrasonicSensor:
    def __init__(self, name, relative_angle_deg, min_valid_distance=25.0):
        """
        Args:
            name (str): Sensor identifier ('forward', 'back', 'right', 'left')
            relative_angle_deg (float): Angle relative to robot's forward direction (degrees)
            min_valid_distance (float): Minimum valid distance (in cm) for sensor readings.
        """
        self.name = name
        self.relative_angle_deg = relative_angle_deg
        self.distance = None  # Latest distance in cm
        self.min_valid_distance = min_valid_distance

    def update_distance(self, distance):
        """Update the sensor's distance reading."""
        self.distance = distance

    def get_absolute_angle(self, yaw):
        """Calculate absolute angle based on robot's Yaw."""
        return (yaw + self.relative_angle_deg) % 360

    def get_data(self, yaw):
        """Return distance and absolute angle if valid."""
        # Use the dynamic threshold (min_valid_distance) instead of hard-coding 25.0.
        if self.distance is None or self.distance > 500 or self.distance <= self.min_valid_distance:
            return None
        return {'distance': self.distance, 'angle': self.get_absolute_angle(yaw)}
