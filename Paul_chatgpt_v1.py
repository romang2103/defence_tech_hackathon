import librosa
import numpy as np
import os

test_path = '/Users/paulpaul/Desktop/Dronesounds'
print("Directories and files in test path:", os.listdir(test_path))

def load_and_analyze(file_path):
    # Load an audio file as a floating point time series.
    audio, sr = librosa.load(file_path)
    # Compute the spectral centroid.
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    # Compute the spectral bandwidth.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    # Fourier Transform of the audio
    fft = np.fft.fft(audio)
    # Power Spectral Density (magnitude squared of the FFT)
    psd = np.abs(fft) ** 2
    # Extract frequency bins
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    return np.mean(spectral_centroids), np.mean(spectral_bandwidth), psd, freqs, sr

def compare_sounds(reference_features, test_feature):
    # Calculate Euclidean distance between spectral features
    spectral_distances = [np.linalg.norm(np.array(ref[:2]) - np.array(test_feature[:2])) for ref in reference_features]
    min_spectral_distance = min(spectral_distances)
    closest_spectral_index = spectral_distances.index(min_spectral_distance)

    # Calculate similarity based on Power Spectral Density
    psd_distances = []
    for ref in reference_features:
        # Interpolate test PSD to match frequency bins of the reference
        interp_psd = np.interp(ref[3], test_feature[3], test_feature[2])
        # Calculate the correlation between the reference PSD and interpolated test PSD
        correlation = np.corrcoef(ref[2], interp_psd)[0, 1]
        psd_distances.append(correlation)

    max_correlation = max(psd_distances)
    closest_psd_index = psd_distances.index(max_correlation)
    
    return (closest_spectral_index, min_spectral_distance), (closest_psd_index, max_correlation)

# Specify the folder path containing the sound files
folder_path = "/Users/paulpaul/Desktop/Dronesounds"
sound_files = [f for f in os.listdir(folder_path) if f.endswith('.m4a')]

# Analyze each sound file in the folder
reference_features = []
for sound_file in sound_files:      
    feature_path = os.path.join(folder_path, sound_file)
    features = load_and_analyze(feature_path)
    reference_features.append(features)

# Path to the test sound file
test_sound_path = '/Users/paulpaul/Desktop/236978615-drone-flying-overhead-uav-asce.m4a'
test_features = load_and_analyze(test_sound_path)

# Compare the test sound with the reference sounds
spectral_comparison, psd_comparison = compare_sounds(reference_features, test_features)
print(f"The closest sound by spectral features is at index {spectral_comparison[0]} with a distance of {spectral_comparison[1]}")
print(f"The closest sound by PSD correlation is at index {psd_comparison[0]} with a correlation of {psd_comparison[1]}")
