'''
Author: Hengyuan Xu

This file is for the multi-level summarization. Post-process the pipeline output to generate more high-level summaries.
'''

import os
import pandas as pd
import numpy as np

import argparse

from .model import Doubao

import json

# bot = Doubao(
#     system_prompt="请你根据几段剧情描述，来描述、总结这段剧情"
# )

#### helper functions ####
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="video_summaries.csv", help="The file containing the video summaries.")
    parser.add_argument("--output_file", type=str, default="high_level_summaries.csv", help="The file to save the high-level summaries.")
    parser.add_argument("--clips_per_level", type=int, default=5, help="The number of summaries per level.")
    return parser.parse_args()

def prompt_generation(summary_clips):
    '''
    Generate the prompt for summarizing the clips.
    '''
    prompt = "以下几段按时间顺序排列的剧情描述来自于同一部电视剧的一些片段，请你根据这些片段，来描述这段故事："
    for clip in summary_clips:
        prompt += "\n" + clip
    return prompt


def main():
    args = parse_args()
    # read the video summaries
    print("===> Reading the file...")
    df = pd.read_csv(args.input_file)
    # get the summary clips
    name_col = "视频片段总结"
    # get the number of levels
    initial_summaries = df[name_col].tolist()
    print("===> Get {} initial summaries".format(len(initial_summaries)))

    multi_level_summaries = []


    # initialize Doubao
    bot = Doubao(
        system_prompt="你是一个电视剧剧情总结机器人，用户会给出几段剧情描述，这些剧情描述按照电视剧剧情发生的时间顺序排列，你需要根据这些剧情描述来描述、总结这段故事。如果有混乱的字幕或者不相关的内容，请忽略掉。"
    )

    
    multi_level_summaries.append(initial_summaries)

    level = 0
    print("===> Start summarizing the summaries...")
    while 1:
        # if we got one final summary, break
        if len(multi_level_summaries[level]) == 1:
            break
        # else, summarize the summaries, each time summarize 5 summaries
        num_in_this_level = len(multi_level_summaries[level]) // args.clips_per_level
        print("===> Summarizing level {}... with {} summaries".format(level, len(multi_level_summaries[level])))
        multi_level_summaries.append([])

        for i in range(num_in_this_level):
            summary_clips = multi_level_summaries[level][i * args.clips_per_level: (i + 1) * args.clips_per_level]
            prompt = prompt_generation(summary_clips)
            summary = bot(prompt)
            multi_level_summaries[level + 1].append(summary)
        # deal with the remaining summaries
        if len(multi_level_summaries[level]) % args.clips_per_level != 0:
            summary_clips = multi_level_summaries[level][num_in_this_level * args.clips_per_level:]
            prompt = prompt_generation(summary_clips)
            summary = bot(prompt)
            multi_level_summaries[level + 1].append(summary)
        level += 1
    bot.off_load()
    # save the multi-level summaries

    print("===> Saving the multi-level summaries...")
    with open("multi_level_summaries.json", "w", encoding="utf-8") as f:
        json.dump(multi_level_summaries, f, ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()
    