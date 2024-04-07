# Run script as is. It will process the video and provide outputs. 
# This script can take a very long time depending on video length. 

video_path = '2024-04-01 18-08-37.mkv'
minimum_silence_duration = 0.3
minimum_silence_new_clip= 8
audio_rate = 22050

from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np

def audio_to_dB(audio_clip, fps=22050):
    """
    Attempts to convert an audio clip to its dB value.
    
    :param audio_clip: The audio clip from moviepy.
    :param fps: Frames per second for the audio clip analysis.
    :return: The dB value of the audio clip, averaged over all channels.
    """
    # Extract the audio frames as an array
    audio_frames = audio_clip.to_soundarray(fps=fps)
    
    # Calculate the RMS of the audio frames
    rms = np.sqrt(np.mean(np.square(audio_frames), axis=0))
    
    # Prevent log of zero
    rms[rms == 0] = 1e-10
    
    # Calculate the dB value
    dB = 20 * np.log10(rms)
    
    # Return the mean dB value of stereo channels
    return np.mean(dB)

def detect_audio_segments(video, silence_threshold=-30.0, min_silence_duration=0.2, chunk_size=0.1):
    """
    Detects audio segments in a video file based on specified silence duration and prints progress.
    Outputs segments in units of seconds since the start of the video.
    
    :param video: VideoFileClip object
    :param silence_threshold: The dB threshold considered as silence
    :param min_silence_duration: Minimum duration of silence to consider (in seconds)
    :param chunk_size: Duration of each audio chunk analyzed (in seconds)
    :return: List of tuples indicating start and end times (in seconds) of audio segments
    """
    audio = video.audio
    duration = audio.duration
    segments = []
    current_start = None
    silence_accumulator = 0  # To accumulate silence durations

    # For progress reporting
    total_chunks = np.ceil(duration / chunk_size)
    chunk_count = 0

    for start in np.arange(0, duration, chunk_size):
        end = start + chunk_size
        if end > duration:
            end = duration
        chunk = audio.subclip(start, end)
        dB = audio_to_dB(chunk, audio_rate)
        is_silent = dB < silence_threshold

        if is_silent:
            silence_accumulator += chunk_size
            if current_start is not None and silence_accumulator >= min_silence_duration:
                # End the current segment before the silence
                segments.append((current_start, start))
                current_start = None
                silence_accumulator = 0  # Reset silence accumulator
        else:
            silence_accumulator = 0  # Reset silence accumulator
            if current_start is None:
                current_start = start  # Start a new segment

        # Update and print progress
        chunk_count += 1
        progress = (chunk_count / total_chunks) * 100
        print(f"Progress: {progress:.2f}%", end='\r\n')

    if current_start is not None:
        # Add the final segment if it doesn't end with silence
        segments.append((current_start, duration))

    return segments


def combine_segments(segments, max_gap=0.5):
    """
    Combine video segments if the gap between the end of one segment and the start of the next
    is less than the specified maximum gap.
    
    :param segments: List of tuples indicating the start and end times of segments.
    :param max_gap: Maximum allowed gap (in seconds) between segments to combine them.
    :return: A new list of tuples with combined segments.
    """
    if not segments:
        return []

    # Start with the first segment
    combined_segments = [segments[0]]

    for current_start, current_end in segments[1:]:
        # Get the end time of the last segment in the combined list
        last_end = combined_segments[-1][1]

        # If the gap between the current segment and the last segment is less than max_gap, combine them
        if current_start - last_end <= max_gap:
            # Update the end time of the last segment in the combined list
            combined_segments[-1] = (combined_segments[-1][0], current_end)
        else:
            # Otherwise, add the current segment as a new entry in the combined list
            combined_segments.append((current_start, current_end))

    return combined_segments

def group_segments_by_gap(segments, long_gap_threshold=10.0):
    """
    Group segments into lists, starting a new list when a gap longer than the specified threshold is detected.
    
    :param segments: List of tuples indicating the start and end times of segments.
    :param long_gap_threshold: Gap duration (in seconds) that triggers a new list for subsequent segments.
    :return: A list of lists, where each inner list contains segments to be combined into one clip.
    """
    grouped_segments = []
    current_group = []

    for i, (current_start, current_end) in enumerate(segments):
        # If it's the first segment, start the first group
        if i == 0:
            current_group.append((current_start, current_end))
        else:
            # Calculate the gap between the current segment and the previous segment
            previous_segment = segments[i - 1]
            gap = current_start - previous_segment[1]
            
            # If the gap is longer than the threshold, start a new group
            if gap > long_gap_threshold:
                if current_group:
                    grouped_segments.append(current_group)
                    current_group = []
            # Add the current segment to the current group
            current_group.append((current_start, current_end))
    
    # Don't forget to add the last group if it's not empty
    if current_group:
        grouped_segments.append(current_group)

    return grouped_segments


def output_grouped_segments_as_videos(source_video_path, grouped_segments):
    """
    Outputs each group of segments as its own video file, combining the segments specified
    in each group into a single continuous clip.
    
    :param source_video_path: Path to the source video file.
    :param grouped_segments: List of lists, where each inner list contains segments (start and end times)
                             that should be combined into a single video clip.
    :return: None
    """
    # Load the source video
    video = VideoFileClip(source_video_path)

    for i, group in enumerate(grouped_segments):
        # Create a list of video clips for the current group
        clips = [video.subclip(start, end) for start, end in group]

        # Combine the clips into one continuous video
        combined_clip = concatenate_videoclips(clips)

        # Output the combined clip to a file
        output_filename = f"output_video_group_{i+1:03d}.mp4"
        combined_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
        
        print(f"Generated {output_filename}")


video = VideoFileClip(video_path)

audio_segments = detect_audio_segments(video, silence_threshold=-30.0, min_silence_duration=minimum_silence_duration, chunk_size=0.1)

grouped_segments = group_segments_by_gap(audio_segments, minimum_silence_new_clip)

print(f"Generating {len(grouped_segments)} video segments based on minimum_silence_new_clip of {minimum_silence_new_clip} seconds")

output_grouped_segments_as_videos(video_path, grouped_segments)
