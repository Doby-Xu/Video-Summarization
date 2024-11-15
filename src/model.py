'''
Author: Hengyuan Xu

This file contains code for the MiniCPM model.
'''

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu # pip install decord

import os
from volcenginesdkarkruntime import Ark

import whisper


class MiniCPM:
    def __init__(self, fps = 1, max_num_frames = 64):
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        self.model = self.model.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

        self.MAX_NUM_FRAMES = max_num_frames
        self.fps = fps

        self.params = {}
        self.params["use_image_id"] = False
        self.params["max_slice_nums"] = 1
    
    def encode_video(self, video_path):
        '''
        Encode the video into frames
        '''
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / self.fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, self.MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames
    
    def __call__(self, video_path, question = "描述视频内容"):
        '''
            处理视频并生成对视频内容的描述。

            参数:
            video_path (str): 视频文件的路径。
            question (str): 要求描述视频内容的问题，默认为 "描述视频内容"。

            返回:
            answer (str): 对视频内容的描述。
        '''
        frames = self.encode_video(video_path)
        msgs = [
            {'role': 'user', 'content': frames + [question]}, 
        ]
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **self.params
        )
        return answer
    
    def off_load(self):
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
    
    # destroy
    def __del__(self):
        self.off_load()



class Doubao:
    def __init__(self, system_prompt):
        self.client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        self.system_prompt = system_prompt
    def get_response(self, user_input, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.client.chat.completions.create(
            model="ep-20241115095025-cw5x8",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return completion.choices[0].message.content
    def __call__(self, user_input, system_prompt=None):
        return self.get_response(user_input, system_prompt)
    
    def off_load(self):
        pass

    
class Whisper:
    def __init__(self, model_name="turbo"):
        self.model = whisper.load_model(model_name)
    def cat_text(self, segments):
        text = ""
        for segment in segments:
            text += segment["text"]
            text += "\n"
        return text
    def __call__(self, audio_path):
        result = self.model.transcribe(audio_path, initial_prompt="中文：", word_timestamps=True)
        return self.cat_text(result["segments"])
    def off_load(self):
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
    def __del__(self):
        self.off_load()
        
    