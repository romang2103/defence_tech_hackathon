import librosa
import numpy as np
import os
from collections import Counter

def normalize_features(features):
    print("normalzief")
    # Normalize spectral features using z-score normalization
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features

def determine_audio_properties(folder_path, sound_files):
    print("audioprop")
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


def load_and_analyze(file_path, target_length, target_sr):
    print("load")
    # Load an audio file, resampling to the target sample rate
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Pad the audio if it is shorter than the target length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    # Compute the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    # Compute the spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    
    # Perform FFT with a consistent length
    n_fft = 2 ** np.ceil(np.log2(target_length)).astype(int)  # Ensure FFT length is a power of two
    fft = np.fft.fft(audio, n=n_fft)
    psd = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(n_fft, 1 / sr)
    
    return np.mean(spectral_centroids), np.mean(spectral_bandwidth), psd, freqs, sr

def normalize_psd(psd):
    print("normalziepsd")
    # Normalize PSD to have a unit sum, avoiding division by zero
    return psd / np.sum(psd) if np.sum(psd) != 0 else psd

def create_binary_mask(psd, threshold_ratio=0.1):
    print("binarymask")
    # Create a binary mask where elements are 1 if above threshold_ratio percentile, else 0
    threshold = np.percentile(psd, 100 * (1 - threshold_ratio))
    return np.where(psd > threshold, 1, 0)

def compare_sounds(reference_features, test_feature):
    print("compare_sounds")
    # Normalize spectral data
    all_features = np.array(
        [test_feature[:2]] + [ref[:2] for ref in reference_features]
    )
    normalized_features = normalize_features(all_features)

    test_normalized = normalized_features[0]
    references_normalized = normalized_features[1:]

    # Calculate Euclidean distance between spectral features
    spectral_distances = [
        np.linalg.norm(ref - test_normalized) for ref in references_normalized
    ]
    min_spectral_distance = min(spectral_distances)
    closest_spectral_index = spectral_distances.index(min_spectral_distance)

    
        # Normalize PSD for the test feature
    normalized_test_psd = normalize_psd(test_feature[2])
    test_mask = create_binary_mask(normalized_test_psd)

    psd_distances = []
    for ref in reference_features:
        # Ensure frequency bins are aligned
        common_freqs = np.intersect1d(test_feature[3], ref[3])
        test_psd_interp = np.interp(common_freqs, test_feature[3], test_feature[2]) 
        ref_psd_interp = np.interp(common_freqs, ref[3], ref[2]) 
        
        # Compute combined mask
        test_mask = test_feature[2] > np.percentile(test_feature[2], 95)
        ref_mask = ref[2] > np.percentile(ref[2], 95)
        combined_mask = test_mask & ref_mask

        # Apply mask to interpolated PSDs
        masked_test_psd = test_psd_interp * combined_mask
        masked_ref_psd = ref_psd_interp * combined_mask
        
        # Calculate the correlation between masked PSDs
        if np.any(combined_mask):  # Check if combined_mask is not all false
            correlation = np.corrcoef(masked_test_psd, masked_ref_psd)[0, 1]
        else:
            correlation = 0  # Set correlation to 0 if there is no overlap

        psd_distances.append(correlation)

    max_correlation = max(psd_distances)
    closest_psd_index = psd_distances.index(max_correlation)

    return (closest_spectral_index, min_spectral_distance), (
        closest_psd_index,
        max_correlation,
    )


# Specify the folder path containing the sound files
folder_path = (
    "/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/Dronesounds1"
)
sound_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]


# Assuming 'folder_path' and 'sound_files' are defined
max_length, target_sr = determine_audio_properties(folder_path, sound_files)

# Analyze each sound file in the folder
reference_features = []
for sound_file in sound_files:
    feature_path = os.path.join(folder_path, sound_file)
    features = load_and_analyze(feature_path, target_sr, max_length)
    reference_features.append(features)

# Path to the test sound file
test_sound_path = (
    "/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/output_audio.wav"
)
test_features = load_and_analyze(test_sound_path, target_sr, max_length)

print(reference_features)
# Compare the test sound with the reference sounds
spectral_comparison, psd_comparison = compare_sounds(reference_features, test_features)
print(
    f"The closest sound by spectral features is at index {spectral_comparison[0]} with a distance of {spectral_comparison[1]}"
)
print(
    f"The closest sound by PSD correlation is at index {psd_comparison[0]} with a correlation of {psd_comparison[1]}"
)
