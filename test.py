import librosa
import numpy as np
import os


def normalize_features(features):
    # Normalize spectral features using z-score normalization
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features


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
    freqs = np.fft.fftfreq(len(fft), 1 / sr)
    return np.mean(spectral_centroids), np.mean(spectral_bandwidth), psd, freqs, sr


def normalize_psd(psd):
    # Normalize PSD using Min-Max scaling
    min_psd = np.min(psd)
    max_psd = np.max(psd)
    return (psd - min_psd) / (max_psd - min_psd)


def compare_sounds(reference_features, test_feature):
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

    # Normalize and compare PSD
    normalized_test_psd = normalize_psd(test_feature[2])
    psd_distances = []
    for ref in reference_features:
        normalized_ref_psd = normalize_psd(ref[2])
        # Interpolate test PSD to match frequency bins of the reference
        interp_psd = np.interp(ref[3], test_feature[3], normalized_test_psd)
        # Calculate the correlation between the reference PSD and interpolated test PSD
        correlation = np.corrcoef(normalized_ref_psd, interp_psd)[0, 1]
        psd_distances.append(correlation)

    max_correlation = max(psd_distances)
    closest_psd_index = psd_distances.index(max_correlation)

    return (closest_spectral_index, min_spectral_distance), (
        closest_psd_index,
        max_correlation,
    )


# Specify the folder path containing the sound files
folder_path = (
    "C:\\Users\\roman\\Documents\\GitHub\\defence_tech_hackathon\\Dronesounds1"
)
sound_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Analyze each sound file in the folder
reference_features = []
for sound_file in sound_files:
    feature_path = os.path.join(folder_path, sound_file)
    features = load_and_analyze(feature_path)
    reference_features.append(features)

# Path to the test sound file
test_sound_path = (
    "C:\\Users\\roman\\Documents\\GitHub\\defence_tech_hackathon\\test.wav"
)
test_features = load_and_analyze(test_sound_path)

# Compare the test sound with the reference sounds
spectral_comparison, psd_comparison = compare_sounds(reference_features, test_features)
print(
    f"The closest sound by spectral features is at index {spectral_comparison[0]} with a distance of {spectral_comparison[1]}"
)
print(
    f"The closest sound by PSD correlation is at index {psd_comparison[0]} with a correlation of {psd_comparison[1]}"
)
