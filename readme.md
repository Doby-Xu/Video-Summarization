# 视频总结

事实上，这只是“计算机视觉”研究生课程中一个简单的课程项目

## 项目要求

todo

## 项目实现

todo

## 使用

考虑到本项目中使用了本地运行的面壁小钢炮视觉大模型，和本地运行的OpenAI Whisper语音转文本模型，请确保本地至少有20G的显存，C盘或/home至少有35G空间

并且，考虑到语言模型调用了豆包大模型，请根据[火山引擎文档](https://www.volcengine.com/docs/82379/1302008)，将自己的API加入环境变量，将模型接入点（形似 `"ep-xxxxxxxxx"`）插入到`src/model.py`中的`Doubao`中

1. 使用`video_process.py`处理视频。在处理之前，请自行定位视频的开头标题和结尾演职员表，并填入`video_process.py`中的`OP_TIME`和`ED_TIME`处，以便删除
```bash 
python -m src.video_process
```
2. 运行pipeline
```bash
python -m src.pipeline --output_dir output
```

