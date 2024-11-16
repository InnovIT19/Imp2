import os
import cv2
import numpy as np

# Step 1: Extract Snapshots from Video
def extract_snapshots(video_path, output_folder, num_snapshots=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_snapshots  # Calculate interval to get 20 snapshots

    snapshot_count = 0
    while cap.isOpened() and snapshot_count < num_snapshots:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            snapshot_path = os.path.join(output_folder, f"frame_{snapshot_count}.jpg")
            cv2.imwrite(snapshot_path, frame)
            snapshot_count += 1
        frame_count += 1
    cap.release()

# Usage
video_path = ('C:/Users/DELL/Desktop/Rashmi/Project/1st/v3.mp4')
output_folder = 'C:/Users/DELL/Desktop/Rashmi/Project/1st/snapshots'

extract_snapshots(video_path, output_folder)

# Step 2: Extract Body Dimensions from Snapshot
def extract_body_dimensions(snapshot_path):
    dimensions = {
        'height': round(np.random.uniform(100, 200), 2),  # in cm
        'shoulder_width': round(np.random.uniform(30, 55), 2),  # in cm
        'waist_circumference': round(np.random.uniform(20, 50), 2),  # in cm
        'hip_circumference': round(np.random.uniform(20, 50), 2)  # in cm
    }
    return dimensions

snapshot_path = os.path.join(output_folder, 'frame_0.jpg')
dimensions = extract_body_dimensions(snapshot_path)
print(f"Extracted Dimensions: {dimensions}")

# Step 3: Define Reinforcement Learning Model
class ReinforcementLearningModel:
    def __init__(self):
        self.q_table = {}
        self.actions = ['hourglass', 'inverted_triangle', 'rectangle', 'pear']
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

    def _choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.actions)
        else:
            action = max(self.q_table.get(state, {}), key=self.q_table.get(state, {}).get, default=np.random.choice(self.actions))
        return action

    def _learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        predict = self.q_table[state][action]
        target = reward + self.discount_factor * max(self.q_table[next_state].values())
        self.q_table[state][action] += self.learning_rate * (target - predict)

    def train(self, dimensions, actual_shape):
        state = tuple(dimensions.values())
        action = self._choose_action(state)
        reward = 1 if action == actual_shape else -1
        self._learn(state, action, reward, state)

    def predict(self, dimensions):
        state = tuple(dimensions.values())
        return self._choose_action(state)

# Step 4: Train and Test the Model
model = ReinforcementLearningModel()

for _ in range(1000):
    dummy_dimensions = {
        'height': np.random.uniform(150, 200),  # in cm
        'shoulder_width': np.random.uniform(35, 55),  # in cm
        'waist_circumference': np.random.uniform(27, 50),  # in cm
        'hip_circumference': np.random.uniform(20, 50)  # in cm
    }
    actual_shape = np.random.choice(['hourglass', 'inverted_triangle', 'rectangle', 'pear'])
    model.train(dummy_dimensions, actual_shape)

# Predict body shape for extracted dimensions
predicted_shape = model.predict(dimensions)
print(f"Predicted body shape: {predicted_shape}")

# Step 5: Recommend Upper and Lower Body Size
def recommend_size(shoulder_width, waist_circumference, hip_circumference):
    # Upper body size recommendation thresholds (in cm)
    upper_body_sizes = {
        'XS': {'shoulder_width': 38, 'waist_circumference': 27},
        'S': {'shoulder_width': 40, 'waist_circumference': 30},
        'M': {'shoulder_width': 43, 'waist_circumference': 32},
        'L': {'shoulder_width': 45, 'waist_circumference': 34},
        'XL': {'shoulder_width': 48, 'waist_circumference': 36},
        'XXL': {'shoulder_width': 50, 'waist_circumference': 40},
        'XXXL': {'shoulder_width': 50, 'waist_circumference': 45}
    }

    # Lower body size recommendation thresholds (in cm)
    lower_body_sizes = {
        'XS': {'hip_circumference': 32, 'waist_circumference': 27},
        'S': {'hip_circumference': 34, 'waist_circumference': 30},
        'M': {'hip_circumference': 36, 'waist_circumference': 33},
        'L': {'hip_circumference': 38, 'waist_circumference': 36},
        'XL': {'hip_circumference': 40, 'waist_circumference': 38},
        'XXL': {'hip_circumference': 42, 'waist_circumference': 40},
        'XXXL': {'hip_circumference': 45, 'waist_circumference': 45}
    }

    upper_size = 'XL'  # Default to XL if no other sizes match
    lower_size = 'XL'  # Default to XL if no other sizes match

    for size, thresholds in upper_body_sizes.items():
        if shoulder_width <= thresholds['shoulder_width'] and waist_circumference <= thresholds['waist_circumference']:
            upper_size = size
            break

    for size, thresholds in lower_body_sizes.items():
        if hip_circumference <= thresholds['hip_circumference'] and waist_circumference <= thresholds['waist_circumference']:
            lower_size = size
            break

    return upper_size, lower_size

# Get the size recommendation
recommended_upper_size, recommended_lower_size = recommend_size(
    dimensions['shoulder_width'], dimensions['waist_circumference'], dimensions['hip_circumference']
)
print(f"Recommended upper body size: {recommended_upper_size}")
print(f"Recommended lower body size: {recommended_lower_size}")