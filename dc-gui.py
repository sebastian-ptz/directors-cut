import sys
import cv2
from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QAction, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import *

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_saliency_map = None
        self.total_frames = 0
        self.actual_fps = 0.0
        self.volume_before_mute = 75

        self.setWindowTitle("360° Directors Cut Viewer")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.resize(800, 700)

        # Menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Video...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Mediaplayer
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)

        # Timeline
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.sliderMoved.connect(self.set_position)
        self.timeline_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.timecode_label = QLabel("00:00")
        self.timecode_label.setMinimumWidth(45)
        self.timecode_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        timeline_area = QHBoxLayout()

        timeline_area.addWidget(self.timeline_slider, 1)
        timeline_area.addWidget(self.timecode_label)
        timeline_widget = QWidget()
        timeline_widget.setLayout(timeline_area)
        timeline_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Controls
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.saliency_predictors = QComboBox()
        self.saliency_predictors.addItems(["one", "two"])
        self.saliency_predictors.currentTextChanged.connect(self.update_saliency_display)

        self.framecode_label = QLabel("frame: - / -")
        self.framecode_label.setMinimumWidth(100)
        self.framecode_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.video_filename = QLabel("No video loaded")
        self.video_filename.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.video_filename.setAlignment(Qt.AlignCenter)

        self.volume_button = QPushButton()
        self.volume_button.setIconSize(QSize(16,16))
        self.volume_button.setFixedSize(QSize(24,24))
        self.volume_button.setStyleSheet("QPushButton { border: none; background-color: transparent; }")
        self.volume_button.clicked.connect(self.toggle_mute)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setMaximumWidth(150)
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.set_playback_rate)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.saliency_predictors)
        controls_layout.addWidget(self.framecode_label)
        controls_layout.addWidget(self.video_filename, 1)
        controls_layout.addWidget(self.volume_button)
        controls_layout.addWidget(self.volume_slider)
        controls_layout.addWidget(self.speed_combo)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Top area
        top_area = QVBoxLayout()
        top_area.addWidget(self.video_widget, 2)
        top_area.addWidget(controls_widget)
        top_area.addWidget(timeline_widget)
        top_area_widget = QWidget()
        top_area_widget.setLayout(top_area)
        top_area_widget.setMinimumHeight(350)

        # Saliency map label
        self.saliency_label = QLabel("Saliency Map will Be Displayed Here")
        self.saliency_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.saliency_label.setAlignment(Qt.AlignCenter)
        self.saliency_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")

        # Splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(top_area_widget)
        self.main_splitter.addWidget(self.saliency_label)
        self.main_splitter.setSizes([500, 120])
        main_layout.addWidget(self.main_splitter, 1)

        # Signals
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)
        self.media_player.playbackStateChanged.connect(self.update_play_button_icon)
        self.media_player.positionChanged.connect(self.draw_saliency_progress_marker)
        self.media_player.durationChanged.connect(self.draw_saliency_progress_marker)
        self.audio_output.mutedChanged.connect(self.update_volume_icon)
        self.audio_output.volumeChanged.connect(self.handle_volume_changed_externally)

        # Initial state
        self.audio_output.setVolume(self.volume_before_mute / 100.0)
        self.update_saliency_display()
        self.timecode_label.setText("00:00")
        self.framecode_label.setText("frame: - / -")
        self.update_volume_icon()

    # --- File Loading ---
    def open_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.load_video(selected_files[0])

    def load_video(self, file_path: str):
        self.total_frames = 0
        self.actual_fps = 0.0
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
        except Exception as e:
            print(f"OpenCV: Exception during video processing: {e}")

        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        base_name = QUrl.fromLocalFile(file_path).fileName()
        self.video_filename.setText(f"{base_name}")
        self.timecode_label.setText("00:00")
        self.framecode_label.setText("frame: 0 / 0")
        self.update_saliency_display()

    # --- Playback Controls ---
    def play_video(self):
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def set_playback_rate(self, rate_text: str):
        try:
            rate = float(rate_text[:-1])
            self.media_player.setPlaybackRate(rate)
        except ValueError:
            print(f"Invalid playback rate format: {rate_text}")

    # --- Timeline/Slider ---
    def set_position(self, position: int):
        self.media_player.setPosition(position)

    def update_slider_position(self, position: int):
        self.timeline_slider.setValue(position)
        self.timecode_label.setText(self.format_time(position))
        duration = self.media_player.duration()
        self.framecode_label.setText(self.format_framecode(position, duration))

    def update_slider_range(self, duration: int):
        self.timeline_slider.setRange(0, duration)
        position = self.media_player.position()
        self.framecode_label.setText(self.format_framecode(position, duration))

    def format_time(self, ms: int) -> str:
        ms = max(ms, 0)
        s = round(ms / 1000)
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"

    def format_framecode(self, ms: int, total_ms: int) -> str:
        ms = max(ms, 0)
        current_frame_str = "-"
        total_frames_str = str(self.total_frames) if self.total_frames > 0 else "-"
        if self.actual_fps > 0:
            current_seconds = ms / 1000.0
            current_frame = int(current_seconds * self.actual_fps)
            current_frame_str = str(current_frame)
        if self.total_frames == 0:
            total_frames_str = "-"
        return f"frame: {current_frame_str} / {total_frames_str}"

    # --- Volume Controls ---
    def set_volume(self, value: int):
        if self.audio_output:
            new_volume_float = value / 100.0
            self.audio_output.setVolume(new_volume_float)
            if value == 0:
                self.audio_output.setMuted(True)
            elif self.audio_output.isMuted() and value > 0:
                self.audio_output.setMuted(False)

    def toggle_mute(self):
        if not self.audio_output:
            return
        if self.audio_output.isMuted():
            self.audio_output.setMuted(False)
            if self.audio_output.volume() == 0:
                self.audio_output.setVolume(self.volume_before_mute / 100.0)
        else:
            current_volume_percent = self.audio_output.volume() * 100
            if current_volume_percent > 0:
                self.volume_before_mute = current_volume_percent
            self.audio_output.setMuted(True)

    def update_volume_icon(self):
        if not self.audio_output:
            return
        if self.audio_output.isMuted() or self.audio_output.volume() == 0:
            self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolumeMuted))
        else:
            self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))

    def handle_volume_changed_externally(self, volume_float: float):
        slider_value = int(round(volume_float * 100))
        if self.volume_slider.value() != slider_value:
            self.volume_slider.setValue(slider_value)
        self.update_volume_icon()

    # --- UI State Updates ---
    def update_play_button_icon(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    # --- Saliency Display ---
    def update_saliency_display(self, algorithm_name: str = None):
        if algorithm_name is None:
            algorithm_name = self.saliency_predictors.currentText()
        image_filename = f"saliency_map_{algorithm_name}.png"
        loaded_pixmap = QPixmap(image_filename)
        self.current_saliency_map = loaded_pixmap if not loaded_pixmap.isNull() else None
        self.draw_saliency_progress_marker()

    def draw_saliency_progress_marker(self):
        if self.current_saliency_map is None or self.current_saliency_map.isNull():
            self.saliency_label.setText("Select a saliency map type.")
            if self.media_player.source().isEmpty():
                self.saliency_label.setText("Open a video and select a saliency map type.")
            self.saliency_label.setPixmap(QPixmap())
            return

        pixmap = self.current_saliency_map.copy()
        if not self.media_player.source().isEmpty() and self.media_player.duration() > 0:
            position = self.media_player.position()
            duration = self.media_player.duration()
            if duration > 0:
                progress_ratio = position / duration
                marker_x = int(progress_ratio * pixmap.width())
                painter = QPainter(pixmap)
                pen = QPen(QColor("yellow"), 1)
                painter.setPen(pen)
                painter.drawLine(marker_x, 0, marker_x, pixmap.height())
                painter.end()

        self.saliency_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
