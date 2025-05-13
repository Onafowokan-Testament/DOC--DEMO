from youtube_transcript_api import YouTubeTranscriptApi

yt_api = YouTubeTranscriptApi()
subtitle = yt_api.fetch("HZAIHTU7XgE")

print(subtitle)
