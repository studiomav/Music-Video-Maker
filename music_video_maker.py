import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QCheckBox, QComboBox, QProgressBar, QMessageBox,
                            QSlider, QDialog, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QPalette, QColor
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtCore import QUrl
import moviepy.editor as mp
import librosa
import librosa.display
import numpy as np
import cv2
from pathlib import Path
import random
from datetime import datetime

# Create default directories
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Create directories if they don't exist
os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

class VideoProcessor(QThread):
    progress_updated = Signal(int, str)
    encoding_progress = Signal(int)  # New signal for encoding percentage
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, music_path, input_dir, output_dir, mute_clips, shuffle_clips, 
                 aspect_ratio, resolution, beat_sensitivity):
        super().__init__()
        self.music_path = music_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mute_clips = mute_clips
        self.shuffle_clips = shuffle_clips
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
        self.beat_sensitivity = beat_sensitivity
        print(f"VideoProcessor initialized with beat sensitivity: {self.beat_sensitivity}")

    def _progress_callback(self, t):
        if hasattr(self, 'total_duration') and self.total_duration > 0:
            percentage = int((t / self.total_duration) * 100)
            self.encoding_progress.emit(percentage)

    def _apply_transition(self, clip, duration=0.5):
        """Apply a random transition effect to the clip."""
        if not self.use_transitions:
            return clip

        transition_type = random.choice(['fade', 'painting'])
        
        if transition_type == 'fade':
            return clip.fx(fadein, duration).fx(fadeout, duration)
        else:  # painting
            angle = random.choice([-45, 45])
            return clip.fx(painting, 0.1, 0.1)

    def run(self):
        try:
            self.progress_updated.emit(0, "Validating inputs...")
            if not self._validate_inputs():
                return
            video_files = [f for f in os.listdir(self.input_dir) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if len(video_files) < 2:
                self.error.emit("At least 2 video files are required")
                return
            self.progress_updated.emit(10, "Analyzing music for beats...")
            
            # Load audio file
            y, sr = librosa.load(self.music_path)
            
            # Get onset strength with sensitivity adjustment
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Scale onset envelope based on sensitivity
            sensitivity_scale = self.beat_sensitivity / 100.0
            onset_env = onset_env * sensitivity_scale
            
            print(f"Using beat sensitivity: {self.beat_sensitivity} (scale: {sensitivity_scale})")
            
            # Get tempo and beat frames using dynamic programming
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                tightness=100,  # Keep tightness constant, we're using sensitivity_scale instead
                trim=True,
                start_bpm=librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            )
            
            print(f"Detected tempo: {tempo} BPM")
            # Convert frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Get onset times for additional beat points with sensitivity adjustment
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                wait=0.1 / sensitivity_scale,  # Adjust wait time based on sensitivity
                pre_avg=0.1,
                post_avg=0.1,
                pre_max=0.1,
                post_max=0.1
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            print(f"Initial beat count: {len(beat_times)}")
            print(f"Initial onset count: {len(onset_times)}")
            
            # Combine beat times and onset times, removing duplicates
            all_times = np.sort(np.unique(np.concatenate([beat_times, onset_times])))
            
            # Remove times that are too close together (adjust minimum interval based on sensitivity)
            min_interval = 0.1 / sensitivity_scale
            filtered_times = [all_times[0]]
            for t in all_times[1:]:
                if t - filtered_times[-1] >= min_interval:
                    filtered_times.append(t)
            
            beat_times = np.array(filtered_times)
            print(f"Final beat count after filtering: {len(beat_times)}")
            
            audio_duration = librosa.get_duration(y=y, sr=sr)
            self.progress_updated.emit(30, "Preparing video clips...")
            clips = []
            for video_file in video_files:
                clip = mp.VideoFileClip(os.path.join(self.input_dir, video_file))
                clips.append(clip)
            if self.shuffle_clips:
                random.shuffle(clips)
            self.progress_updated.emit(50, "Creating final video...")
            final_clips = []
            clip_index = 0
            beat_times = np.append(beat_times, audio_duration)
            for i in range(len(beat_times) - 1):
                duration = beat_times[i + 1] - beat_times[i]
                if duration <= 0:
                    print(f"Skipping segment {i}: non-positive duration ({duration}s)")
                    continue
                current_clip = clips[clip_index % len(clips)]
                # Only use loop if the original is at least half the interval
                if current_clip.duration < duration:
                    if current_clip.duration >= duration / 2:
                        segment = current_clip.loop(duration=duration)
                    else:
                        print(f"Warning: Clip {clip_index % len(clips)} is much shorter ({current_clip.duration:.2f}s) than interval ({duration:.2f}s). Using as-is.")
                        segment = current_clip.subclip(0, current_clip.duration)
                else:
                    max_start = current_clip.duration - duration
                    if max_start > 0:
                        start_time = random.uniform(0, max_start)
                        segment = current_clip.subclip(start_time, start_time + duration)
                    else:
                        segment = current_clip.subclip(0, duration)
                segment = self._resize_clip(segment)
                if self.mute_clips:
                    segment = segment.without_audio()
                # Clamp fps if it's unreasonably high
                if hasattr(segment, 'fps') and segment.fps and segment.fps > 60:
                    print(f'Clamping fps from {segment.fps} to 30 for segment {i}')
                    segment = segment.set_fps(30)
                final_clips.append(segment)
                clip_index += 1
            final_video = mp.concatenate_videoclips(final_clips, method="compose")
            audio = mp.AudioFileClip(self.music_path)
            if not self.mute_clips:
                clips_audio = final_video.audio
                if clips_audio is not None:
                    final_audio = mp.CompositeAudioClip([clips_audio, audio])
                    final_video = final_video.set_audio(final_audio)
                else:
                    final_video = final_video.set_audio(audio)
            else:
                final_video = final_video.set_audio(audio)
            if audio.duration > final_video.duration:
                audio = audio.subclip(0, final_video.duration)
            elif audio.duration < final_video.duration:
                final_video = final_video.subclip(0, audio.duration)
            music_name = os.path.splitext(os.path.basename(self.music_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"music_video_{music_name}_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            self.total_duration = final_video.duration
            self.progress_updated.emit(80, "Encoding video...")
            final_video.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                threads=4,
                preset='medium'
            )
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))

    def _validate_inputs(self):
        if not os.path.isfile(self.music_path):
            self.error.emit("Invalid music file")
            return False
        if not os.path.isdir(self.input_dir):
            self.error.emit("Invalid input directory")
            return False
        if not os.path.isdir(self.output_dir):
            self.error.emit("Invalid output directory")
            return False
        return True

    def _resize_clip(self, clip):
        target_width, target_height = self._get_target_dimensions()
        
        # Calculate the scaling factors for both dimensions
        width_scale = target_width / clip.w
        height_scale = target_height / clip.h
        
        # Use the smaller scale to ensure the video fits within the target dimensions
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(clip.w * scale)
        new_height = int(clip.h * scale)
        
        # Resize the clip
        resized_clip = clip.resize((new_width, new_height))
        
        # Create a black background clip of the target size
        background = mp.ColorClip(size=(target_width, target_height), 
                                color=(0, 0, 0),
                                duration=clip.duration)
        
        # Calculate position to center the resized clip
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Composite the resized clip onto the black background
        final_clip = mp.CompositeVideoClip([
            background,
            resized_clip.set_position((x_offset, y_offset))
        ])
        
        return final_clip

    def _get_target_dimensions(self):
        width, height = self._get_resolution_dimensions()
        if self.aspect_ratio == "16:9":
            return width, height
        elif self.aspect_ratio == "1:1":
            size = min(width, height)
            return size, size
        else:  # 9:16
            return height, width

    def _get_resolution_dimensions(self):
        if self.resolution == "1080p":
            return 1920, 1080
        elif self.resolution == "720p":
            return 1280, 720
        else:  # 480p
            return 854, 480

class HelpWindow(QDialog):
    def __init__(self, parent=None, is_dark_mode=True):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.setFixedSize(400, 300)
        # Set window flags to allow closing but prevent maximizing
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setLayout(QVBoxLayout(main_widget))
        self.layout().setContentsMargins(20, 20, 20, 20)
        
        # Create scroll area for the help text
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create container widget for the help text
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add help text
        help_text = QLabel(
            "Created by @studiomav with the help of Cursor AI\n\n"
            "• Select a music file, input folder of video clips, and output folder to create a music video\n"
            "• Beat Sensitivity controls how many beats are detected. Lower values mean more beats, higher values mean more detailed beat detection.\n\n"
            "Tips:\n"
            "• Use shorter video clips and shorter music for quicker results. You can test out a snippet of a song to adjust your beat sensitivity and then re-run with the full song. If the program seems frozen, it's just taking a while to encode. I'll try to get the progress bar working with video encoding progress in a future update.\n"
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.PlainText)
        container_layout.addWidget(help_text)
        
        # Set the container as the scroll area's widget
        scroll_area.setWidget(container)
        
        # Add scroll area to main layout
        self.layout().addWidget(scroll_area)
        
        # Apply theme
        self._apply_theme(is_dark_mode)
    
    def _apply_theme(self, is_dark_mode):
        if is_dark_mode:
            self.setStyleSheet("""
                QDialog {
                    background-color: rgb(53, 53, 53);
                    color: white;
                }
                QLabel {
                    color: white;
                    background-color: rgb(53, 53, 53);
                }
                QScrollArea {
                    border: none;
                    background-color: rgb(53, 53, 53);
                }
                QScrollArea > QWidget > QWidget {
                    background-color: rgb(53, 53, 53);
                }
                QScrollBar:vertical {
                    border: none;
                    background: rgb(75, 75, 75);
                    width: 10px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: rgb(100, 100, 100);
                    min-height: 20px;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: rgb(240, 240, 240);
                    color: black;
                }
                QLabel {
                    color: black;
                    background-color: rgb(240, 240, 240);
                }
                QScrollArea {
                    border: none;
                    background-color: rgb(240, 240, 240);
                }
                QScrollArea > QWidget > QWidget {
                    background-color: rgb(240, 240, 240);
                }
                QScrollBar:vertical {
                    border: none;
                    background: rgb(230, 230, 230);
                    width: 10px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: rgb(200, 200, 200);
                    min-height: 20px;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Video Maker")
        self.setFixedSize(300, 400)  # Fixed window size
        
        # Set default directories
        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR
        
        # Initialize sound effect
        self.completion_sound = QSoundEffect()
        sound_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complete.wav")
        if os.path.exists(sound_file):
            self.completion_sound.setSource(QUrl.fromLocalFile(sound_file))
            self.completion_sound.setVolume(0.5)  # Set volume to 50%
            self.completion_sound.setLoopCount(1)  # Play only once
        else:
            print(f"Warning: Sound file not found at {sound_file}")
            self.sound_alert_checkbox.setEnabled(False)
            self.sound_alert_checkbox.setChecked(False)
        
        # Initialize UI
        self._init_ui()
        self._setup_theme()
        
        # Initialize processor
        self.processor = None
        self.help_window = None

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(5)  # Reduce spacing between elements
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Music input
        music_layout = QHBoxLayout()
        music_layout.setSpacing(5)
        self.music_label = QLabel("No music file selected")
        self.music_label.setWordWrap(True)  # Allow text to wrap
        music_btn = QPushButton("Select Music")
        music_btn.clicked.connect(self._select_music)
        music_layout.addWidget(self.music_label)
        music_layout.addWidget(music_btn)
        layout.addLayout(music_layout)

        # Input directory
        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)
        self.input_label = QLabel(f"Input: {os.path.basename(self.input_dir)}")
        self.input_label.setWordWrap(True)  # Allow text to wrap
        input_btn = QPushButton("Change")
        input_btn.clicked.connect(self._select_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_btn)
        layout.addLayout(input_layout)

        # Output directory input
        output_layout = QHBoxLayout()
        output_layout.setSpacing(5)
        self.output_label = QLabel(f"Output: {os.path.basename(self.output_dir)}")
        self.output_label.setWordWrap(True)  # Allow text to wrap
        output_btn = QPushButton("Change")
        output_btn.clicked.connect(self._select_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)

        # Options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(5)
        
        # Checkboxes in their own layout
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(5)
        self.mute_checkbox = QCheckBox("Mute Clips")
        self.mute_checkbox.setChecked(True)
        self.shuffle_checkbox = QCheckBox("Shuffle Clips")
        self.shuffle_checkbox.setChecked(True)
        self.sound_alert_checkbox = QCheckBox("Sound Alert")
        self.sound_alert_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.mute_checkbox)
        checkbox_layout.addWidget(self.shuffle_checkbox)
        checkbox_layout.addWidget(self.sound_alert_checkbox)
        options_layout.addLayout(checkbox_layout)
        
        # Beat sensitivity slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.setSpacing(5)
        self.sensitivity_label = QLabel("Beat Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(50)  # Reduced max value since default is lower
        self.sensitivity_slider.setValue(15)  # New default value
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(5)  # Adjusted tick interval
        self.sensitivity_value_label = QLabel("15")  # Updated default label
        self.sensitivity_slider.valueChanged.connect(self._update_sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_value_label)
        options_layout.addLayout(sensitivity_layout)
        
        # Video settings in their own layout
        video_settings_layout = QVBoxLayout()
        video_settings_layout.setSpacing(5)
        
        # Aspect ratio
        aspect_layout = QHBoxLayout()
        aspect_layout.setSpacing(5)
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["16:9", "1:1", "9:16"])
        aspect_layout.addWidget(QLabel("Aspect:"))
        aspect_layout.addWidget(self.aspect_combo)
        video_settings_layout.addLayout(aspect_layout)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.setSpacing(5)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1080p", "720p", "480p"])
        resolution_layout.addWidget(QLabel("Res:"))
        resolution_layout.addWidget(self.resolution_combo)
        video_settings_layout.addLayout(resolution_layout)
        
        options_layout.addLayout(video_settings_layout)
        layout.addLayout(options_layout)

        # Progress bar and status
        status_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        status_layout.addWidget(self.progress_label, 1)
        self.encoding_percent_label = QLabel("")
        self.encoding_percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.encoding_percent_label.setMinimumWidth(60)
        status_layout.addWidget(self.encoding_percent_label, 0, Qt.AlignRight)
        layout.addLayout(status_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #666;
                border-radius: 3px;
                background-color: #2d2d2d;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 1px;
            }
        """)
        layout.addWidget(self.progress_bar, 1)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._start_processing)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        # Bottom buttons layout
        bottom_layout = QHBoxLayout()
        
        # Help button
        self.help_button = QPushButton("❓")  # Question mark emoji
        self.help_button.setFixedSize(36, 30)  # Same size as theme button
        self.help_button.setStyleSheet("""
            QPushButton {
                padding: 5px;
                font-size: 16px;
            }
        """)
        self.help_button.clicked.connect(self._show_help)
        bottom_layout.addWidget(self.help_button)
        
        # Add stretch to push theme button to the right
        bottom_layout.addStretch()
        
        # Theme toggle
        self.theme_button = QPushButton("☀️")
        self.theme_button.setFixedSize(36, 30)
        self.theme_button.setStyleSheet("""
            QPushButton {
                padding: 5px;
                font-size: 16px;
            }
        """)
        self.theme_button.clicked.connect(self._toggle_theme)
        bottom_layout.addWidget(self.theme_button)
        
        layout.addLayout(bottom_layout)

    def _setup_theme(self):
        self.is_dark_mode = True
        self._apply_theme()

    def _apply_theme(self):
        palette = QPalette()
        if self.is_dark_mode:
            # Dark theme colors
            window_color = QColor(53, 53, 53)
            dark_accent = QColor(75, 75, 75)
            text_color = Qt.white
            
            palette.setColor(QPalette.Window, window_color)
            palette.setColor(QPalette.WindowText, text_color)
            palette.setColor(QPalette.Base, dark_accent)  # Dropdown and input background
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, text_color)
            palette.setColor(QPalette.ToolTipText, text_color)
            palette.setColor(QPalette.Text, text_color)  # Text in dropdowns
            palette.setColor(QPalette.Button, dark_accent)  # Button background
            palette.setColor(QPalette.ButtonText, text_color)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
        else:
            # Light theme colors
            window_color = QColor(240, 240, 240)
            text_color = Qt.black
            
            palette.setColor(QPalette.Window, window_color)
            palette.setColor(QPalette.WindowText, text_color)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(233, 233, 233))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, text_color)
            palette.setColor(QPalette.Text, text_color)
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, text_color)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(0, 0, 255))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, Qt.white)

        self.setPalette(palette)
        
        # Apply theme to dropdowns
        dropdown_style = """
            QComboBox {
                background-color: %s;
                color: %s;
                border: 1px solid %s;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 5px;
            }
        """ % (
            "rgb(75, 75, 75)" if self.is_dark_mode else "white",
            "white" if self.is_dark_mode else "black",
            "rgb(100, 100, 100)" if self.is_dark_mode else "rgb(200, 200, 200)"
        )
        
        self.aspect_combo.setStyleSheet(dropdown_style)
        self.resolution_combo.setStyleSheet(dropdown_style)
        
        # Apply theme to buttons
        button_style = """
            QPushButton {
                background-color: %s;
                color: %s;
                border: 1px solid %s;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: %s;
            }
            QPushButton:disabled {
                background-color: %s;
                color: %s;
            }
        """ % (
            "rgb(75, 75, 75)" if self.is_dark_mode else "white",
            "white" if self.is_dark_mode else "black",
            "rgb(100, 100, 100)" if self.is_dark_mode else "rgb(200, 200, 200)",
            "rgb(85, 85, 85)" if self.is_dark_mode else "rgb(245, 245, 245)",
            "rgb(60, 60, 60)" if self.is_dark_mode else "rgb(230, 230, 230)",
            "rgb(150, 150, 150)" if self.is_dark_mode else "rgb(150, 150, 150)"
        )
        
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)

    def _toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self._apply_theme()
        # Update theme button emoji
        self.theme_button.setText("☀️" if self.is_dark_mode else "🌙")
        # Update help window theme if it exists
        if self.help_window:
            self.help_window._apply_theme(self.is_dark_mode)

    def _select_music(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Music File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.aac)"
        )
        if file_path:
            self.music_path = file_path
            self.music_label.setText(os.path.basename(file_path))
            self._check_run_button()

    def _select_input(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            self.input_dir
        )
        if dir_path:
            self.input_dir = dir_path
            self.input_label.setText(f"Input: {os.path.basename(dir_path)}")
            self._check_run_button()

    def _select_output(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"Output: {os.path.basename(dir_path)}")
            self._check_run_button()

    def _check_run_button(self):
        self.run_button.setEnabled(
            hasattr(self, 'music_path') and 
            os.path.isdir(self.input_dir) and
            os.path.isdir(self.output_dir)
        )

    def _update_sensitivity_label(self, value):
        self.sensitivity_value_label.setText(str(value))

    def _start_processing(self):
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.encoding_percent_label.setText("")
        sensitivity = self.sensitivity_slider.value()
        print(f"Starting processing with sensitivity: {sensitivity}")
        self.processor = VideoProcessor(
            self.music_path,
            self.input_dir,
            self.output_dir,
            self.mute_checkbox.isChecked(),
            self.shuffle_checkbox.isChecked(),
            self.aspect_combo.currentText(),
            self.resolution_combo.currentText(),
            sensitivity
        )
        self.processor.progress_updated.connect(self._update_progress)
        self.processor.encoding_progress.connect(self._update_encoding_percent)
        self.processor.finished.connect(self._processing_finished)
        self.processor.error.connect(self._processing_error)
        self.processor.start()

    def _update_progress(self, value, message):
        if message == "Encoding video...":
            # Don't update the main progress bar during encoding
            # The encoding progress will be shown separately
            self.progress_label.setText(message)
        else:
            self.progress_bar.setValue(value)
            self.progress_label.setText(message)

    def _update_encoding_percent(self, percent):
        self.encoding_percent_label.setText(f"{percent}%")
        # Update the main progress bar to show encoding progress
        # Start from 80 and go up to 95 (leaving 5% for final processing)
        encoding_progress = 80 + (percent * 15 // 100)
        self.progress_bar.setValue(encoding_progress)

    def _processing_finished(self, output_path):
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Video saved to: {output_path}")
        self.encoding_percent_label.setText("")
        self.run_button.setEnabled(True)
        
        # Play sound if enabled and source is set
        if self.sound_alert_checkbox.isChecked() and self.completion_sound.source().isValid():
            self.completion_sound.play()
        
        # Reset progress bar after a short delay
        QTimer.singleShot(2000, lambda: self._reset_progress())

    def _reset_progress(self):
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self.encoding_percent_label.setText("")

    def _processing_error(self, error_message):
        self.progress_label.setText("Error occurred")
        self.encoding_percent_label.setText("")
        self.run_button.setEnabled(True)
        QMessageBox.critical(
            self,
            "Error",
            error_message
        )

    def _show_help(self):
        if not self.help_window:
            self.help_window = HelpWindow(self, self.is_dark_mode)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 