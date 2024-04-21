from pytube import YouTube
from moviepy.editor import *

def download_audio(url, output_path, filename):
    # Download video from YouTube
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    
    # Download the audio file
    download_path = video.download(output_path=output_path, filename='temp.mp4')

    # Load the downloaded audio file
    clip = AudioFileClip(download_path)

    # Save the audio as a WAV file
    clip.write_audiofile(os.path.join(output_path, filename + '.wav'))

    # Remove the temporary file
    os.remove(download_path)

# URL of the YouTube video
url = 'https://www.youtube.com/watch?v=QO91wfmHPMo'
output_path = '/Users/paulpaul/Documents/GitHub/defence_tech_hackathon/'
filename = 'output_audio'

download_audio(url, output_path, filename)
