import librosa
import numpy as np
import os
from collections import Counter

def normalize_features(features):
    # Normalize spectral features using z-score normalization
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features

def determine_audio_properties(folder_path, sound_files):
    lengths = []
    sample_rates = Counter()

    for sound_file in sound_files:
        file_path = os.path.join(folder_path, sound_file)
        audio, sr = librosa.load(file_path, sr=None)  # Load at native sampling rate
        lengths.append(len(audio))
        sample_rates[sr] += 1

    max_length = max(lengths)
    most_common_sr = sample_rates.most_common(1)[0][0]  # Get the most common sample rate

    return max_length, most_common_sr

def normalize_psd(psd):
    # Normalize PSD to have a unit sum, avoiding division by zero
    return psd / np.sum(psd) if np.sum(psd) != 0 else psd

def create_binary_mask(psd, threshold_ratio=0.1):
    # Create a binary mask where elements are 1 if above threshold_ratio percentile, else 0
    threshold = np.percentile(psd, 100 * (1 - threshold_ratio))
    return np.where(psd > threshold, 1, 0)

def load_and_analyze(file_path, target_length, target_sr):
    # Load an audio file, resampling to the target sample rate
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Pad the audio if it is shorter than the target length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    # Compute the spectral centroid and bandwidth
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    
    # Perform FFT and compute PSD
    n_fft = len(audio)  # Using the length of the padded audio
    fft = np.fft.fft(audio, n=n_fft)
    psd = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(n_fft, 1 / sr)
    
    return np.mean(spectral_centroids), np.mean(spectral_bandwidth), psd, freqs, sr

def normalize_psd(psd):
    # Normalize PSD to have a unit sum, avoiding division by zero
    return psd / np.sum(psd) if np.sum(psd) != 0 else psd

def create_binary_mask(psd, threshold_ratio=0.1):
    # Create a binary mask where elements are 1 if above threshold_ratio percentile, else 0
    threshold = np.percentile(psd, 100 * (1 - threshold_ratio))
    return np.where(psd > threshold, 1, 0)

def compare_sounds(reference_features, test_feature):
    # Normalize spectral data
    all_features = np.array([test_feature[:2]] + [ref[:2] for ref in reference_features])
    normalized_features = normalize_features(all_features)

    test_normalized = normalized_features[0]
    references_normalized = normalized_features[1:]

    # Calculate Euclidean distance between spectral features
    spectral_distances = [np.linalg.norm(ref - test_normalized) for ref in references_normalized]
    min_spectral_distance = min(spectral_distances)
    closest_spectral_index = spectral_distances.index(min_spectral_distance)

    # Normalize PSD for the test feature
    normalized_test_psd = normalize_psd(test_feature[2])
    test_mask = create_binary_mask(normalized_test_psd)

    psd_distances = []
    for index, ref in enumerate(reference_features):
        normalized_ref_psd = normalize_psd(ref[2])
        ref_mask = create_binary_mask(normalized_ref_psd)
        combined_mask = test_mask & ref_mask

        # Apply mask and calculate the correlation between masked PSDs
        if np.any(combined_mask):
            masked_test_psd = normalized_test_psd[combined_mask]
            masked_ref_psd = normalized_ref_psd[combined_mask]
            correlation = np.corrcoef(masked_test_psd, masked_ref_psd)[0, 1]
        else:
            correlation = 0  # No valid data to correlate

        psd_distances.append(correlation)

    max_correlation = max(psd_distances)
    closest_psd_index = psd_distances.index(max_correlation)

    return (closest_spectral_index, min_spectral_distance), (closest_psd_index, max_correlation)

# Setup paths and process files
folder_path = "/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/Dronesounds1"
sound_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
max_length, target_sr = determine_audio_properties(folder_path, sound_files)

reference_features = [load_and_analyze(os.path.join(folder_path, f), max_length, target_sr) for f in sound_files]
test_sound_path = "/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/test2_fieldsound.wav"
test_features = load_and_analyze(test_sound_path, max_length, target_sr)

# Compare the test sound with the reference sounds
spectral_comparison, psd_comparison = compare_sounds(reference_features, test_features)
print(f"The closest sound by spectral features is at index {spectral_comparison[0]} with a distance of {spectral_comparison[1]}")
print(f"The closest sound by PSD correlation is at index {psd_comparison[0]} with a correlation of {psd_comparison[1]}")
