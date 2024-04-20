import os
import subprocess


def convert_m4a_to_wav(source_folder, target_folder, ffmpeg_path):
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".m4a"):
                # Full path of source and target files
                source_file = os.path.join(root, file)
                target_file = os.path.join(
                    target_folder, file[:-4] + ".wav"
                )  # Change file extension

                # Command to convert m4a to wav using the local FFmpeg
                command = [
                    ffmpeg_path,
                    "-i",
                    source_file,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    target_file,
                ]

                # Execute the command
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Converted {file} to WAV and stored in {target_folder}")


# Specify the source and target folders
source_folder = "C:\\Users\\roman\\Downloads\\Dronesounds"
target_folder = "C:\\Users\\roman\\Downloads\\ConvertedSounds"
ffmpeg_path = "C:\\Users\\roman\\Downloads\\ffmpeg-7.0"  # Adjust this path to where you keep the local FFmpeg

# Run the conversion
convert_m4a_to_wav(source_folder, target_folder, ffmpeg_path)
