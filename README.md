# Music Video Maker

Created by @studiomav with the help of Cursor AI


## Requirements

- Python 3.8 or higher
- FFmpeg installed to your system PATH
- Windows 10 or higher (probably, feel free to test on 7)

## Installation

1. Install FFmpeg:
   - Download FFmpeg from https://ffmpeg.org/download.html
   - Add FFmpeg to your system PATH
     
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

1. Run the program:
   ```bash
   python music_video_maker.py
   ```

## Notes

- Video processing time depends on the length of the music file and the number/length of video clips
- For best results, use video clips that are at least as long as the beat interval, else your clips will loop
- When a clip is longer than necessary for a beat interval, a random segment from within the clip is used

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
