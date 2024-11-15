'''
Author: Hengyuan Xu

This file contains code for the pipeline of video summarization.
'''

from .model import Doubao, Whisper, MiniCPM

import os
import torch
import time
import argparse
# use panda to maintain the tabular data
import pandas as pd
import numpy as np
# tqdm
from tqdm import tqdm
##### helper functions #####

def construct_prompt_clip(summary_video, summary_audio):
    '''
    Construct the prompt for summarizing the video clip.
    '''
    prompt = "请你执行一个内容描述任务，你将会得到一个视频片段的内容描述文本，和这段视频对应的语音字幕文本，请你结合视频描述和角色或旁白的字幕，推测一下这段视频的内容。"
    prompt += f"\n视频内容描述：{summary_video}"
    prompt += f"\n语音字幕内容：{summary_audio}"

    return prompt
def construct_prompt_context(summary_clip, summary_context):
    '''
    Construct the prompt for summarizing the video clip with context.
    '''
    prompt = "请你执行一个内容描述任务，你将会得到一个描述，他描述了一集电视剧中一个小片段的内容。你还会得到一个语境总结，他描述了这一集电视剧在这个片段之前发生的事情。请你根据这两段文本，推测并描述当前这一集电视剧的内容。"
    prompt += f"\n语境总结：{summary_context}"
    prompt += f"\n片段内容描述：{summary_clip}"

    return prompt

def parse_args():
    parser = argparse.ArgumentParser(description="A pipeline for video summarization.")
    # fps for MiniCPM
    parser.add_argument("--fps", type=int, default=1, help="The frame rate of the video, seen by MiniCPM.")
    # data
    parser.add_argument("--data_dir", type=str, default="data", help="The directory to save the data.")
    # output_dir 
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to save the output.")

    # verbose
    parser.add_argument("--verbose", action="store_true", help="Whether to print the intermediate results.")
    return parser.parse_args()

def summarize_video(bot, video_path, video_files, num_clips, output_dir, verbose=False):
    '''
    summarize the video clips
    '''
    summaries_video = []
    # for i in range(num_clips):
    for i in tqdm(range(num_clips)):
        video = os.path.join(video_path, video_files[i])
        summary = bot(video)
        summaries_video.append(summary)
        if verbose:
            print(f"==> Processed video {i+1}/{num_clips}.")
            print(f"==> Summary: \n\t{summary}")
    bot.off_load()
    return summaries_video

def summarize_audio(bot, audio_path, audio_files, num_clips, output_dir, verbose=False):
    '''
    summarize the audio clips
    '''
    summaries_audio = []
    # for i in range(num_clips):
    for i in tqdm(range(num_clips)):
        audio = os.path.join(audio_path, audio_files[i])
        summary = bot(audio)
        summaries_audio.append(summary)
        if verbose:
            print(f"==> Processed audio {i+1}/{num_clips}.")
            print(f"==> Summary: \n\t{summary}")
    bot.off_load()
    return summaries_audio

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("==> Start the video summarization pipeline.")

    print("==> Load the Vision models.")
    
    bot_vision = MiniCPM(fps=args.fps)
    # bot_hearing = Whisper()

    summarizer_clips = Doubao(
        system_prompt = "请你执行一个内容描述任务，你将会得到一个视频片段的内容描述文本，和这段视频对应的语音字幕文本，请你结合视频描述和角色或旁白的字幕，推测一下这段视频的内容。"
    )
    summarizer_context = Doubao(
        system_prompt = "请你执行一个内容总结任务，你将会得到一个描述，他描述了一集电视剧中一个小片段的内容。你还会得到一个语境总结，他描述了这一集电视剧在这个片段之前发生的事情。请你根据这两段文本，推测并描述当前这一集电视剧的内容。"
    )



    video_path = os.path.join(args.data_dir, "video_clips")
    audio_path = os.path.join(args.data_dir, "audio_clips")

    video_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
    audio_files = [f for f in os.listdir(audio_path) if f.endswith(".wav")]

    # print all files name
    print(f"==> Found {len(video_files)} video clips and {len(audio_files)} audio clips.")
    

    num_clips = len(video_files)
    assert len(audio_files) == num_clips

    # rearrange the video files by the index
    video_files = sorted(video_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    audio_files = sorted(audio_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    print(f"==> Video clips: {video_files}")
    print(f"==> Audio clips: {audio_files}")
    
    start_time = time.time()
    # 提取视频片段内容
    print(f"==> Start processing {num_clips} video clips.")
    summaries_video = summarize_video(bot_vision, video_path, video_files, num_clips, args.output_dir, args.verbose)
    print(f"==> Time elapsed: {time.time() - start_time:.2f}s")

    # 提取音频片段内容
    print("==> Load the Audio model.")
    bot_hearing = Whisper()
    print(f"==> Start processing {num_clips} audio clips.")
    summaries_audio = summarize_audio(bot_hearing, audio_path, audio_files, num_clips, args.output_dir, args.verbose)
    print(f"==> Time elapsed: {time.time() - start_time:.2f}s")

    
    # 总结视频片段
    print("==> Start generating video summaries.")
    summaries_clip = []
    # for i in range(num_clips):
    for i in tqdm(range(num_clips)):
        prompt = construct_prompt_clip(summaries_video[i], summaries_audio[i])
        summary = summarizer_clips(user_input=prompt)
        summaries_clip.append(summary)
    print(f"==> Time elapsed: {time.time() - start_time:.2f}s")

    print("==> Start generating context-aware video summaries.")
    # 总结当前视频
    summaries_context = []
    # for i in range(num_clips):
    for i in tqdm(range(num_clips)):
        if i == 0:
            prompt = construct_prompt_context(summaries_clip[i], "这是一部新的电视剧。")
        else:
            prompt = construct_prompt_context(summaries_clip[i], summaries_context[i-1])
        summary = summarizer_context(user_input=prompt)
        summaries_context.append(summary)
    print(f"==> Time elapsed: {time.time() - start_time:.2f}s")

    print("==> Save the results.")
    # 保存结果为表格
    # 列：视频片段描述，视频片段音频描述，视频片段总结，当前视频总结
    df = pd.DataFrame({
        "视频片段描述": summaries_video,
        "视频片段音频描述": summaries_audio,
        "视频片段总结": summaries_clip,
        "当前视频总结": summaries_context
    })
    df.to_csv(os.path.join(args.output_dir, "video_summaries.csv"), index=False)

    print("==> The final summarization is:")
    print(summaries_context[num_clips-1])
    print("\n\n\n==> summarization of each minute:")
    for i in range(num_clips):
        print(f"minute {i+1}: {summaries_clip[i]}\n\n\n")
if __name__ == "__main__":
    main()