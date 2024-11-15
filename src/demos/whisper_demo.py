import whisper

def cat_text(segments):
    text = ""
    for segment in segments:
        text += segment["text"]
        text += "\n"
    return text
model = whisper.load_model("turbo")
result = model.transcribe("data/audio_clips/clip_0.wav", initial_prompt="中文：", word_timestamps=True)
# check out what is in the result
# print(result.keys())
# print(result["segments"])
# print(result["text"])
# for segment in result["segments"]:
#     print(segment["text"], segment["start"], ' to ', segment["end"])
print(cat_text(result["segments"]))



result = model.transcribe("data/audio_clips/clip_1.wav", initial_prompt="中文：", word_timestamps=True)
# print(result["text"])
print(cat_text(result["segments"]))