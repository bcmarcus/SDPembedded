# person_tracker.py
import numpy as np
import time

class PersonTracker:
    def __init__(self):
        self.tracks = {}  # track_id: track_info
        self.next_id = 1
        self.max_age = 5  # frames
        self.max_distance = 50  # pixels
        self.determination_time = 1  # seconds

    def update(self, detections):
        updated_tracks = {}
        for detection in detections:
            detection_centroid = detection['centroid']
            min_distance = float('inf')
            matched_track_id = None
            for track_id, track in self.tracks.items():
                track_centroid = track['centroid']
                distance = np.linalg.norm(np.array(detection_centroid) - np.array(track_centroid))
                if distance < self.max_distance and distance < min_distance:
                    min_distance = distance
                    matched_track_id = track_id
            if matched_track_id is not None:
                track = self.tracks[matched_track_id]
                track.update(detection)
                updated_tracks[matched_track_id] = track
            else:
                track_id = self.next_id
                self.next_id += 1
                timestamp = time.time()
                track = {
                    'id': track_id,
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'age': 0,
                    'start_time': timestamp,
                    'data': [{
                        'timestamp': timestamp,
                        'pose': detection['pose'],
                        'eye_status': detection['eye_status']
                    }],
                    'unconscious': False,
                    'unconscious_since': None
                }
                updated_tracks[track_id] = track
        for track_id in self.tracks.keys():
            if track_id not in updated_tracks:
                track = self.tracks[track_id]
                track['age'] += 1
                if track['age'] <= self.max_age:
                    updated_tracks[track_id] = track
        self.tracks = updated_tracks

    def trim_data(self, track):
        current_time = time.time()
        track['data'] = [entry for entry in track['data'] 
                         if current_time - entry['timestamp'] <= self.determination_time]

    def check_unconscious(self, track):
        data = track['data']
        if not data:
            return False
        poses = [entry['pose'] for entry in data]
        eye_statuses = [entry['eye_status'] for entry in data]
        sitting_or_lying = [pose in ['Sitting', 'Lying Down'] for pose in poses]
        eyes_closed = [eye_status == 'Eyes Closed' for eye_status in eye_statuses]
        percent_sitting_or_lying = sum(sitting_or_lying) / len(data)
        percent_eyes_closed = sum(eyes_closed) / len(data)
        unconscious_criteria_met = percent_sitting_or_lying >= 0.8 and percent_eyes_closed >= 0.8
        current_time = time.time()
        if unconscious_criteria_met:
            if not track.get('unconscious'):
                track['unconscious'] = True
                track['unconscious_since'] = current_time
        else:
            track['unconscious'] = False
            track['unconscious_since'] = None
        return track['unconscious']
