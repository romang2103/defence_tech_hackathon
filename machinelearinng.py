import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def extract_psd_features(audio, sr, n_fft=2048):
    fft = np.fft.fft(audio, n=n_fft)
    psd = np.abs(fft)**2
    return psd

def load_data(directory, label):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            # Extract PSD features
            psd = extract_psd_features(audio, sr)
            features.append(psd)
            labels.append(label)
    return features, labels

# Load datasets
yes_features, yes_labels = load_data('/Users/paulpaul/Downloads/DroneAudioDataset-master/Binary_Drone_Audio/yes_drone', 1)
unknown_features, unknown_labels = load_data('/Users/paulpaul/Downloads/DroneAudioDataset-master/Binary_Drone_Audio/unknown', 0)

# Combine datasets
X = np.array(yes_features + unknown_features)
y = np.array(yes_labels + unknown_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

def predict_audio_segments(file_path, model, segment_duration=1):
    # Load the entire audio file
    audio, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    
    # Calculate the number of segments
    num_segments = int(total_duration // segment_duration)
    print(num_segments)
    predictions = []
    
    for i in range(num_segments):
        start_sample = int(i * segment_duration * sr)
        end_sample = int(start_sample + segment_duration * sr)
        
        # Extract the segment
        segment = audio[start_sample:end_sample]
        
        # Extract features
        psd = extract_psd_features(segment, sr)
        psd = psd.reshape(1, -1)  # Reshape for prediction
        
        # Predict
        prediction = model.predict(psd)
        predictions.append(prediction[0])
    
    return predictions

def aggregate_predictions(predictions, threshold=0.2):
    # Calculate the fraction of segments that were predicted to contain a drone
    fraction = np.mean(predictions)
    print(fraction)
    # Determine if a drone is present based on the threshold
    return "Drone Present" if fraction >= threshold else "No Drone"

# Example usage
test_file_path = '/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/output_audio.wav'
segment_predictions = predict_audio_segments(test_file_path, clf)  # Use the trained model 'clf'
final_decision = aggregate_predictions(segment_predictions)
print(final_decision)
