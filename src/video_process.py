'''
Author: Hengyuan Xu

This file is used to preprocess the video data.

We cut the video into clips, and save the video clips and audio clips into the corresponding folders.
'''

import os
import cv2
import numpy as np
import subprocess
import shutil
from moviepy.editor import VideoFileClip


MAX_NUMS_FRAMES = 64 # frames the clip contains. If the clip contains more than 300 frames, we will uniformly sample MAX_FRAME_NUM frames from the clip. This is determined by the memory limit of the model. Down size it if cuda out of memory.
VIDEO_CLIP_LENGTH = 64 # seconds. Depends on MAX_NUMS_FRAMES, and the memory limit of the model. Down size it if cuda out of memory. Actually MiniCPM sample 1 frame per second, so the video clip length should be less than MAX_NUMS_FRAMES seconds.
FPS = 1 # frames per second, sample 1 frame per second from the video


OP_TIME = 1 * 60 + 45 # seconds, skip the first 1 minute and 45 seconds of the video (it is just the intro)
EP_TIME = 43 * 60 + 27 # seconds, discard all the clips after this time (they are just the credits)

VIDEO_INPUT_DIR = "s01e01.mp4"
VIDEO_OUTPUT_DIR = "data/video_clips"
AUDIO_OUTPUT_DIR = "data/audio_clips"

####### helper functions #######

def uniform_sample(l, n):
    '''
    Uniformly sample n elements from the list l.
    '''
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def clip_video(video_input_dir, video_output_dir, audio_output_dir):
    '''
    Clip the video into clips of length VIDEO_CLIP_LENGTH seconds.
    Save the video clips into video_output_dir.
    Save the audio clips into audio_output_dir.
    '''

    # create the output directories
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    # get the video duration
    video = VideoFileClip(video_input_dir)
    video_duration = video.duration

    # cut out the intro and credits
    video = video.subclip(OP_TIME, EP_TIME)

    # get the number of clips
    num_clips = int(video.duration / VIDEO_CLIP_LENGTH)

    # cut the video into clips
    for i in range(num_clips):
        start_time = i * VIDEO_CLIP_LENGTH
        end_time = (i + 1) * VIDEO_CLIP_LENGTH
        clip = video.subclip(start_time, end_time)
        # sample FPS frames per second
        # clip = clip.set_fps(FPS)
        # maybe we should save the clip with the original FPS, and adjust the FPS when we encode the video

        # save the video clip
        clip.write_videofile(os.path.join(video_output_dir, f"clip_{i}.mp4"))

        # save the audio clip
        clip.audio.write_audiofile(os.path.join(audio_output_dir, f"clip_{i}.wav"))

    # close the video
    video.close()

def main():
    clip_video(VIDEO_INPUT_DIR, VIDEO_OUTPUT_DIR, AUDIO_OUTPUT_DIR)

if __name__ == "__main__":

    main()
