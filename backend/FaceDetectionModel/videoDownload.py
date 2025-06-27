
url = 'https://www.youtube.com/watch?v=9EqrUK7ghho'  # Replace with your video URL

import yt_dlp

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    'merge_output_format': 'mp4',  # force output format to mp4
    'outtmpl': 'downloaded_video.mp4',
    'keepvideo': True  # optional: keeps original downloaded parts
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
