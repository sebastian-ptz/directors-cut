import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading

from PySide6.QtCore import Qt, QUrl, QSize, QThread, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QAction, QColor, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QStyle,
    QSlider, QLabel, QPushButton, QComboBox, QSizePolicy, QSplitter, QFileDialog,
    QProgressBar, QMessageBox, QTextEdit, QTabWidget
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import analysis classes from main script.py
@dataclass
class VideoMetadata:
    """Container for video metadata"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: int

@dataclass
class DirectorsCutFrame:
    """Represents a single frame in the directors cut"""
    frame_index: int
    azimuth_pixel: float
    elevation_pixel: float
    
    def to_spherical(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to spherical coordinates"""
        azimuth = (360.0 / image_width) * self.azimuth_pixel
        elevation = (180.0 / image_height) * self.elevation_pixel
        return azimuth, elevation
    
    def to_cartesian(self, image_width: int, image_height: int) -> Tuple[float, float, float]:
        """Convert to Cartesian coordinates for viewport calculations"""
        azimuth, elevation = self.to_spherical(image_width, image_height)
        theta = np.deg2rad(azimuth)
        phi = np.deg2rad(elevation)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

@dataclass
class AttentionScore:
    """Represents the attention score for a frame"""
    frame_index: int
    overlap_percentage: float
    color_rgb: Tuple[int, int, int]
    
    @property
    def attention_level(self) -> str:
        """Classify attention level based on overlap percentage"""
        if self.overlap_percentage >= 0.5:
            return "high"
        elif self.overlap_percentage >= 0.3:
            return "medium"
        else:
            return "low"

class AnalysisWorker(QThread):
    """Worker thread for running analysis without blocking GUI"""
    progress_updated = Signal(int, str)  # progress percentage, status message
    analysis_completed = Signal(list)  # List[AttentionScore]
    error_occurred = Signal(str)
    
    def __init__(self, video_path: str, dc_file_path: str, frames_dir: str, saliency_maps_dir: str):
        super().__init__()
        self.video_path = video_path
        self.dc_file_path = dc_file_path
        self.frames_dir = frames_dir
        self.saliency_maps_dir = saliency_maps_dir
        self._should_stop = False
    
    def stop(self):
        self._should_stop = True
    
    def run(self):
        try:
            from DCV_main import DirectorsCutApplication
            
            app = DirectorsCutApplication()
            
            # Simulate progress for demonstration
            self.progress_updated.emit(10, "Loading video metadata...")
            if self._should_stop: return

            
            results = app.run_analysis(
                video_path=self.video_path,
                dc_file_path=self.dc_file_path,
                video_frames_dir=self.frames_dir,
                saliency_maps_dir=self.saliency_maps_dir,
                output_dir="results"
            )
            
            if self._should_stop: return
            
            self.progress_updated.emit(100, "Analysis complete!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.error_occurred.emit(str(e))

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_saliency_map = None
        self.total_frames = 0
        self.actual_fps = 0.0
        self.volume_before_mute = 75
        self.attention_scores: List[AttentionScore] = []
        self.directors_cut_data: List[DirectorsCutFrame] = []
        self.analysis_worker = None
        self.is_dragging_slider = False # To prevent constant updates while dragging

        # Overlay states
        self.show_directors_cut = False
        self.show_saliency_overlay = False
        self.show_highlight_brush = False
        self.current_view_mode = "video-player" # "video-player", "directors-map", "saliency-map"

        self.setWindowTitle("360° Directors Cut Analysis")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.resize(1200, 800)

        # Create main tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- SETUP MEDIA PLAYER FIRST ---
        self.setup_media_player()

        # Create tabs
        self.create_player_tab()
        self.create_analysis_tab()
        self.create_results_tab()

        # --- INITIAL STATE ---
        self.audio_output.setVolume(self.volume_before_mute / 100.0)
        self.update_saliency_display()
        self.timecode_label.setText("00:00")
        self.framecode_label.setText("frame: - / -")
        self.update_volume_icon()
        self.switch_view("video-player") # Default to video player view
        
        # --- BUTTON FUNCTIONALITY ---
        self.open_file_action.triggered.connect(self.open_file)
        self.directors_map_action.triggered.connect(lambda: self.switch_view("directors-map"))
        self.saliency_map_action.triggered.connect(lambda: self.switch_view("saliency-map"))
        self.directors_cut_action.triggered.connect(self.toggle_directors_cut_overlay)
        self.saliency_overlay_action.triggered.connect(self.toggle_saliency_overlay)
        self.highlight_brush_action.triggered.connect(self.toggle_highlight_brush)

        # Try to load default video
        if os.path.exists("videos/mono_jaunt.mp4"):
            self.load_video("videos/mono_jaunt.mp4")
    
    def setup_media_player(self):
        """Initialize media player with proper audio configuration"""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        
        # Fix for audio channel layout error - set a proper audio format
        try:
            # Set default audio format to avoid channel layout issues
            self.media_player.setAudioOutput(self.audio_output)
            # Ensure audio output is properly configured
            self.audio_output.setVolume(0.75)  # Set reasonable default volume
        except Exception as e:
            logger.warning(f"Audio setup warning: {e}")
            # Continue without audio if there are issues
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            # Stop the worker thread
            self.analysis_worker.stop()
            self.analysis_worker.wait()  # Wait for thread to finish
            logger.info("Analysis worker thread stopped")
        
        # Cleanup media player
        try:
            self.media_player.stop()
        except Exception as e:
            logger.warning(f"Media player cleanup warning: {e}")
        
        # Accept the close event
        event.accept()

    def create_player_tab(self):
        """Create the main video player tab"""
        player_widget = QWidget()
        player_layout = QVBoxLayout(player_widget)

        # --- TOOLBAR ---
        self.toolbar = QToolBar("View Options")
        self.toolbar.setMovable(True)
        actual_icon_size = 20
        icon_size_metric = self.style().pixelMetric(QStyle.PixelMetric.PM_ToolBarIconSize)
        if icon_size_metric > 0:
            scaled_size = int(icon_size_metric * 0.8)
            if scaled_size > 0:
                actual_icon_size = scaled_size
        toolbar_icon_qsize = QSize(actual_icon_size, actual_icon_size)
        self.toolbar.setIconSize(toolbar_icon_qsize)
        player_layout.addWidget(self.toolbar)

        # Create actions with fallback icons if custom icons don't exist
        self.open_file_action = QAction(self.get_icon("./icons/file-open.svg", QStyle.StandardPixmap.SP_DirOpenIcon), "Open Video File", self)
        self.toolbar.addAction(self.open_file_action)
        self.toolbar.addSeparator()

        self.directors_map_action = QAction(self.get_icon("./icons/directors-map.svg", QStyle.StandardPixmap.SP_ComputerIcon), "Directors Map", self)
        self.directors_map_action.setCheckable(True)
        self.saliency_map_action = QAction(self.get_icon("./icons/saliency-map.svg", QStyle.StandardPixmap.SP_ComputerIcon), "Saliency Map", self)
        self.saliency_map_action.setCheckable(True)
        self.toolbar.addAction(self.directors_map_action)
        self.toolbar.addAction(self.saliency_map_action)
        self.toolbar.addSeparator()

        self.directors_cut_action = QAction(self.get_icon("./icons/directors-cut.svg", QStyle.StandardPixmap.SP_DialogApplyButton), "Directors Cut", self)
        self.directors_cut_action.setCheckable(True)
        self.saliency_overlay_action = QAction(self.get_icon("./icons/saliency-overlay.svg", QStyle.StandardPixmap.SP_DialogApplyButton), "Saliency Overlay", self)
        self.saliency_overlay_action.setCheckable(True)
        self.toolbar.addAction(self.directors_cut_action)
        self.toolbar.addAction(self.saliency_overlay_action)
        self.toolbar.addSeparator()

        self.highlight_brush_action = QAction(self.get_icon("./icons/paint-brush.svg", QStyle.StandardPixmap.SP_DialogApplyButton), "Highlight Brush", self)
        self.highlight_brush_action.setCheckable(True)
        self.toolbar.addAction(self.highlight_brush_action)

        # --- VIEWS ---
        self.directors_map_label = QLabel("Attention Map will be displayed here after analysis")
        self.directors_map_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.directors_map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.directors_map_label.setStyleSheet("border: 1px solid black; background-color: #fff0f0;")
        self.directors_map_label.setMinimumSize(QSize(1, 80))
        self.directors_map_label.setScaledContents(True)

        self.saliency_map_view_label = QLabel("Saliency Map View Area")
        self.saliency_map_view_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.saliency_map_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.saliency_map_view_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.saliency_map_view_label.setScaledContents(True)
        self.saliency_map_view_label.hide()

        # --- VIDEO PLAYER ---
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.paintEvent = self.paint_video_frame # Override paint event for overlays

        # --- CONTROLS ---
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.sliderPressed.connect(self.on_slider_dragged)
        self.timeline_slider.sliderReleased.connect(self.on_slider_dropped)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.timeline_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.timecode_label = QLabel("00:00")
        self.timecode_label.setMinimumWidth(45)
        self.timecode_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        timeline_area = QHBoxLayout()
        timeline_area.addWidget(self.timeline_slider, 1)
        timeline_area.addWidget(self.timecode_label)
        timeline_widget = QWidget()
        timeline_widget.setLayout(timeline_area)
        timeline_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)
        
        self.saliency_predictors = QComboBox()
        self.saliency_predictors.addItems(["FastSal"]) # These correspond to saliency algorithms
        self.saliency_predictors.currentTextChanged.connect(self.update_saliency_display) # This should trigger a display update
        
        self.framecode_label = QLabel("frame: - / -")
        self.framecode_label.setMinimumWidth(100)
        self.framecode_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        self.video_filename = QLabel("No video loaded")
        self.video_filename.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.video_filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.volume_button = QPushButton()
        self.volume_button.setIconSize(QSize(16,16))
        self.volume_button.setFixedSize(QSize(24,24))
        self.volume_button.setStyleSheet("QPushButton { border: none; background-color: transparent; }")
        self.volume_button.clicked.connect(self.toggle_mute)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
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
        controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # --- LAYOUT ---
        top_area = QVBoxLayout()
        top_area.addWidget(self.video_widget, 2)
        top_area.addWidget(controls_widget)
        top_area.addWidget(timeline_widget)
        top_area_widget = QWidget()
        top_area_widget.setLayout(top_area)
        top_area_widget.setMinimumHeight(350)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(top_area_widget)
        self.main_splitter.addWidget(self.directors_map_label)
        self.main_splitter.addWidget(self.saliency_map_view_label)
        self.saliency_map_view_label.setVisible(False)
        self.main_splitter.setSizes([500, 120]) # Initial sizes for splitter
        player_layout.addWidget(self.main_splitter, 1)

        # --- SIGNALS ---
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)
        self.media_player.playbackStateChanged.connect(self.update_play_button_icon)
        self.media_player.positionChanged.connect(self.draw_saliency_progress_marker)
        self.media_player.durationChanged.connect(self.draw_saliency_progress_marker)
        self.audio_output.mutedChanged.connect(self.update_volume_icon)
        self.audio_output.volumeChanged.connect(self.handle_volume_changed_externally)

        self.tab_widget.addTab(player_widget, "Video Player")

    def get_icon(self, icon_path: str, fallback_icon: QStyle.StandardPixmap) -> QIcon:
        """Get icon from file path or use fallback system icon"""
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        else:
            return self.style().standardIcon(fallback_icon)

    def create_analysis_tab(self):
        """Create the analysis configuration tab"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)

        # File selection section
        file_section = QVBoxLayout()
        # Video file
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video File:"))
        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        video_layout.addWidget(self.video_path_label, 1)
        video_browse_btn = QPushButton("Browse...")
        video_browse_btn.clicked.connect(self.browse_video_file)
        video_layout.addWidget(video_browse_btn)
        file_section.addLayout(video_layout)

        # Directors Cut file
        dc_layout = QHBoxLayout()
        dc_layout.addWidget(QLabel("Directors Cut File:"))
        self.dc_path_label = QLabel("No DC file selected")
        self.dc_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        dc_layout.addWidget(self.dc_path_label, 1)
        dc_browse_btn = QPushButton("Browse...")
        dc_browse_btn.clicked.connect(self.browse_dc_file)
        dc_layout.addWidget(dc_browse_btn)
        file_section.addLayout(dc_layout)

        # Frames directory
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Video Frames Dir:"))
        self.frames_path_label = QLabel("No frames directory selected")
        self.frames_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        frames_layout.addWidget(self.frames_path_label, 1)
        frames_browse_btn = QPushButton("Browse...")
        frames_browse_btn.clicked.connect(self.browse_frames_dir)
        frames_layout.addWidget(frames_browse_btn)
        file_section.addLayout(frames_layout)

        # Saliency map directory
        saliency_layout = QHBoxLayout()
        saliency_layout.addWidget(QLabel("Saliency Maps Dir:"))
        self.saliency_maps_path_label = QLabel("No saliency maps directory selected")
        self.saliency_maps_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        saliency_layout.addWidget(self.saliency_maps_path_label, 1)
        saliency_browse_btn = QPushButton("Browse...")
        saliency_browse_btn.clicked.connect(self.browse_salmaps_dir)
        saliency_layout.addWidget(saliency_browse_btn)
        file_section.addLayout(saliency_layout)

        analysis_layout.addLayout(file_section)

        # Analysis controls
        controls_section = QVBoxLayout()
        
        self.start_analysis_button = QPushButton("Start Analysis")
        self.start_analysis_button.clicked.connect(self.start_analysis)
        controls_section.addWidget(self.start_analysis_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_section.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to analyze...")
        controls_section.addWidget(self.status_label)
        
        analysis_layout.addLayout(controls_section)
        analysis_layout.addStretch(1) # Push content to top

        self.tab_widget.addTab(analysis_widget, "Analysis")

    def create_results_tab(self):
        """Create the results display tab"""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch(1)

        results_layout.addLayout(export_layout)
        self.tab_widget.addTab(results_widget, "Results")

    def open_file(self):
        """Open file dialog to select a video file."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.load_video(selected_files[0])
                # Also update analysis tab video path
                self.video_path_label.setText(selected_files[0])

    def browse_video_file(self):
        """Browse for video file in analysis tab."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.video_path_label.setText(selected_files[0])

    def browse_dc_file(self):
        """Browse for Directors Cut file in analysis tab."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.dc_path_label.setText(selected_files[0])

    def browse_frames_dir(self):
        """Browse for video frames directory in analysis tab."""
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.FileMode.Directory)
        dir_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if dir_dialog.exec():
            selected_dir = dir_dialog.selectedUrls()
            if selected_dir:
                self.frames_path_label.setText(selected_dir[0].toLocalFile())
    
    def browse_salmaps_dir(self):
        """Browse for saliency maps directory in analysis tab."""
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.FileMode.Directory)
        dir_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if dir_dialog.exec():
            selected_dir = dir_dialog.selectedUrls()
            if selected_dir:
                self.saliency_maps_path_label.setText(selected_dir[0].toLocalFile())

    def load_video(self, file_path: str):
        """Load video into the media player and extract metadata."""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", f"Video file not found: {file_path}")
            self.video_filename.setText("No video loaded")
            self.media_player.setSource(QUrl()) # Clear current source
            return

        try:
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.video_filename.setText(os.path.basename(file_path))
            
            # Brief play and pause to initialize media player
            self.media_player.play()
            self.media_player.pause()

            # Use OpenCV to get actual frame count and FPS
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                self.framecode_label.setText(f"frame: 0 / {self.total_frames}")
                logger.info(f"Loaded video: {file_path}, Frames: {self.total_frames}, FPS: {self.actual_fps}")
            else:
                self.total_frames = 0
                self.actual_fps = 0.0
                self.framecode_label.setText("frame: - / -")
                logger.warning(f"Could not open video with OpenCV: {file_path}")

            self.timeline_slider.setRange(0, 0) # Reset slider
            self.media_player.setPosition(0) # Reset playback position
            self.update_play_button_icon(self.media_player.playbackState())
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            QMessageBox.critical(self, "Video Load Error", f"Could not load video:\n{str(e)}")

    def play_video(self):
        """Toggle video playback (play/pause)."""
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.media_player.pause()
            else:
                self.media_player.play()
        except Exception as e:
            logger.error(f"Error controlling playback: {e}")

    def update_slider_position(self, position):
        """Update timeline slider position and timecode label."""
        if not self.is_dragging_slider:
            self.timeline_slider.setValue(position)
            self.update_timecode(position)
            self.update_framecode(position)
            self.video_widget.update() # Trigger repaint for overlays

    def update_slider_range(self, duration):
        """Update timeline slider range based on video duration."""
        self.timeline_slider.setRange(0, duration)
        self.update_timecode(self.media_player.position())

    def update_timecode(self, ms):
        """Update the displayed timecode."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds %= 60
        self.timecode_label.setText(f"{minutes:02d}:{seconds:02d}")

    def update_framecode(self, ms):
        """Update the displayed framecode."""
        if self.actual_fps > 0 and self.total_frames > 0:
            frame_index = int((ms / 1000.0) * self.actual_fps)
            self.framecode_label.setText(f"frame: {frame_index} / {self.total_frames}")
        else:
            self.framecode_label.setText("frame: - / -")

    def update_play_button_icon(self, state):
        """Update play/pause button icon based on playback state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def on_slider_dragged(self):
        """Handle slider pressed event."""
        self.is_dragging_slider = True

    def on_slider_dropped(self):
        """Handle slider released event."""
        self.is_dragging_slider = False
        self.media_player.setPosition(self.timeline_slider.value())

    def on_slider_moved(self, position):
        """Handle slider moved event (for smooth scrubbing)."""
        self.update_timecode(position)
        self.update_framecode(position)
        self.video_widget.update() # Trigger repaint for overlays
        
    def toggle_mute(self):
        """Toggle audio mute state."""
        try:
            self.audio_output.setMuted(not self.audio_output.isMuted())
            self.update_volume_icon()
        except Exception as e:
            logger.warning(f"Audio mute error: {e}")

    def set_volume(self, volume):
        """Set audio volume."""
        try:
            self.audio_output.setVolume(volume / 100.0)
            self.volume_before_mute = volume
            self.update_volume_icon()
        except Exception as e:
            logger.warning(f"Audio volume error: {e}")

    def update_volume_icon(self):
        """Update volume button icon based on mute state and volume level."""
        try:
            if self.audio_output.isMuted() or self.audio_output.volume() == 0:
                self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolumeMuted))
            elif self.audio_output.volume() > 0.5:
                self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
            else:
                self.volume_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
            self.volume_slider.setValue(int(self.audio_output.volume() * 100))
        except Exception as e:
            logger.warning(f"Volume icon update error: {e}")

    def handle_volume_changed_externally(self, volume):
        """Handle volume changes not initiated by the slider."""
        # This is mainly to keep the slider in sync if volume is changed programmatically
        try:
            self.volume_slider.setValue(int(volume * 100))
            self.update_volume_icon()
        except Exception as e:
            logger.warning(f"External volume change error: {e}")

    def set_playback_rate(self, rate_str):
        """Set video playback rate."""
        try:
            rate = float(rate_str.replace('x', ''))
            self.media_player.setPlaybackRate(rate)
        except Exception as e:
            logger.error(f"Playback rate error: {e}")

    def switch_view(self, view_mode: str):
        """Switch between different display views."""
        self.current_view_mode = view_mode

        #self.video_widget.setVisible(False)
        self.directors_map_label.setVisible(False)
        self.saliency_map_view_label.setVisible(False)

        self.directors_map_action.setChecked(False)
        self.saliency_map_action.setChecked(False)

        if view_mode == "video-player":
            self.video_widget.setVisible(True)
        elif view_mode == "directors-map":
            self.directors_map_label.setVisible(True)
            self.directors_map_action.setChecked(True)
        elif view_mode == "saliency-map":
            self.saliency_map_view_label.setVisible(True)
            self.saliency_map_action.setChecked(True)
        
        self.update_saliency_display() # Update display based on new view

    def update_saliency_display(self):
        """Update the saliency map or directors map display."""
        if self.current_view_mode == "saliency-map":
            algorithm_name = self.saliency_predictors.currentText()
            image_filename = f"results/saliency_map_{algorithm_name}.png" # Standard name from DCmain
            if Path(image_filename).exists():
                loaded_pixmap = QPixmap(image_filename)
                if not loaded_pixmap.isNull():
                    self.current_saliency_map = loaded_pixmap
                    self.saliency_map_view_label.setPixmap(
                        self.current_saliency_map.scaled(self.saliency_map_view_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )
                    self.saliency_map_view_label.setText("")
                else:
                    self.current_saliency_map = None
                    self.saliency_map_view_label.setText(f"Failed to load saliency map: {algorithm_name}")
            else:
                self.current_saliency_map = None
                self.saliency_map_view_label.setText(f"Saliency map not found for {algorithm_name}. Run analysis first.")
        elif self.current_view_mode == "directors-map":
            video_title = self.video_path_label.text().split('/')[-1].rsplit('.', 1)[0] if self.video_path_label.text() != "No video selected" else "unknown"
            attention_map_path = f"results/{video_title}_attention_map.png"
            if Path(attention_map_path).exists():
                loaded_pixmap = QPixmap(attention_map_path)
                if not loaded_pixmap.isNull():
                    self.directors_map_label.setPixmap(
                        loaded_pixmap.scaled(self.directors_map_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    )
                    self.directors_map_label.setScaledContents(True)
                    self.directors_map_label.setText("")
                else:
                    self.directors_map_label.setText("Failed to load Directors Map.")
            else:
                self.directors_map_label.setText("Directors Map not found. Run analysis first.")
        self.draw_saliency_progress_marker() # Update marker always

    def draw_saliency_progress_marker(self):
        """Draw a progress marker on the currently displayed saliency/directors map."""
        target_label = None
        pixmap_to_draw = QPixmap()

        if self.current_view_mode == "saliency-map" and self.current_saliency_map:
            target_label = self.saliency_map_view_label
            pixmap_to_draw = self.current_saliency_map.copy()
        elif self.current_view_mode == "directors-map":
            target_label = self.directors_map_label
            video_title = self.video_path_label.text().split('/')[-1].rsplit('.', 1)[0] if self.video_path_label.text() != "No video selected" else "unknown"
            attention_map_path = f"results/{video_title}_attention_map.png"
            if Path(attention_map_path).exists():
                pixmap_to_draw = QPixmap(attention_map_path)
            
        if target_label and not pixmap_to_draw.isNull():
            if not self.media_player.source().isEmpty() and self.media_player.duration() > 0:
                position = self.media_player.position()
                duration = self.media_player.duration()
                if duration > 0:
                    progress_ratio = position / duration
                    marker_x = int(progress_ratio * pixmap_to_draw.width())
                    painter = QPainter(pixmap_to_draw)
                    pen = QPen(QColor("black"), 10) # Thicker yellow line
                    painter.setPen(pen)
                    painter.drawLine(marker_x, 0, marker_x, pixmap_to_draw.height())
                    painter.end()
            
            target_label.setPixmap(
                pixmap_to_draw.scaled(target_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            )

    def paint_video_frame(self, event):
        """Custom paint event for video_widget to draw overlays."""
        try:
            # Call original paint event first
            QVideoWidget.paintEvent(self.video_widget, event)

            painter = QPainter(self.video_widget)
            
            # Get current frame index
            current_ms = self.media_player.position()
            if self.actual_fps > 0:
                current_frame_index = int((current_ms / 1000.0) * self.actual_fps)
            else:
                current_frame_index = -1

            video_rect = self.video_widget.rect()
            image_width = video_rect.width()
            image_height = video_rect.height()

            if self.show_directors_cut and self.directors_cut_data:
                self.draw_directors_cut_overlay(painter, current_frame_index, image_width, image_height)

            if self.show_saliency_overlay and self.attention_scores:
                self.draw_saliency_attention_overlay(painter, current_frame_index, image_width, image_height)
            
            # You could implement highlight brush drawing here if needed

            painter.end()
        except Exception as e:
            logger.warning(f"Paint event error: {e}")

    def draw_directors_cut_overlay(self, painter: QPainter, current_frame_index: int, image_width: int, image_height: int):
        """Draw the Directors Cut (gaze) overlay on the video frame."""
        try:
            for dc_frame in self.directors_cut_data:
                if dc_frame.frame_index == current_frame_index:
                    # Convert normalized pixel coordinates to display pixel coordinates
                    # Assuming dc_file_path has normalized pixel coordinates (0-width, 0-height)
                    # You might need to adjust this if the coordinates are spherical or percentage based
                    azimuth_display_pixel = (dc_frame.azimuth_pixel / 360.0) * image_width
                    elevation_display_pixel = (dc_frame.elevation_pixel / 180.0) * image_height
                    
                    # Draw a circle at the Directors Cut point
                    pen = QPen(QColor("blue"), 5) # Blue circle for Directors Cut
                    painter.setPen(pen)
                    # Draw relative to the video_widget's top-left corner
                    painter.drawEllipse(int(azimuth_display_pixel) - 10, int(elevation_display_pixel) - 10, 20, 20)
                    break
        except Exception as e:
            logger.warning(f"Directors cut overlay error: {e}")

    def draw_saliency_attention_overlay(self, painter: QPainter, current_frame_index: int, image_width: int, image_height: int):
        """Draw the saliency attention overlay on the video frame."""
        try:
            for score in self.attention_scores:
                if score.frame_index == current_frame_index:
                    # Use the color from the attention score
                    color = QColor(*score.color_rgb)
                    
                    # For now, just drawing a status indicator based on attention level
                    painter.setPen(QPen(color, 5))
                    font = painter.font()
                    font.setPointSize(20)
                    painter.setFont(font)
                    
                    # Position text in the bottom left, slightly above the video controls
                    text_rect = painter.fontMetrics().boundingRect(f"Attention: {score.attention_level.upper()}")
                    text_x = 20
                    text_y = image_height - text_rect.height() - 20 # 20 pixels from bottom
                    painter.drawText(text_x, text_y, f"Attention: {score.attention_level.upper()}")
                    break
        except Exception as e:
            logger.warning(f"Saliency overlay error: {e}")

    def toggle_directors_cut_overlay(self):
        """Toggle the visibility of the Directors Cut overlay."""
        self.show_directors_cut = self.directors_cut_action.isChecked()
        self.video_widget.update() # Trigger repaint

    def toggle_saliency_overlay(self):
        """Toggle the visibility of the Saliency Overlay."""
        self.show_saliency_overlay = self.saliency_overlay_action.isChecked()
        self.video_widget.update() # Trigger repaint

    def toggle_highlight_brush(self):
        """Toggle the highlight brush functionality."""
        self.show_highlight_brush = self.highlight_brush_action.isChecked()
        # Additional logic for brush interaction (e.g., mouse events) would go here
        self.video_widget.update() # Trigger repaint

    # --- Analysis Tab Functions ---
    def start_analysis(self):
        video_path = self.video_path_label.text()
        dc_file_path = self.dc_path_label.text()
        frames_dir = self.frames_path_label.text()
        salmap_dir = self.saliency_maps_path_label.text()

        if not all([video_path != "No video selected", dc_file_path != "No DC file selected", frames_dir != "No frames directory selected"]):
            QMessageBox.warning(self, "Missing Information", "Please select all required files and directories for analysis.")
            return

        if not os.path.exists(video_path):
            QMessageBox.warning(self, "File Not Found", f"Video file not found: {video_path}")
            return
        if not os.path.exists(dc_file_path):
            QMessageBox.warning(self, "File Not Found", f"Directors Cut file not found: {dc_file_path}")
            return
        
        # Create frames directory if it doesn't exist. The main script expects this.
        try:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)
            if not os.path.isdir(frames_dir):
                QMessageBox.warning(self, "Directory Error", f"Could not create or access video frames directory: {frames_dir}")
                return
        except Exception as e:
            QMessageBox.warning(self, "Directory Error", f"Error creating frames directory: {str(e)}")
            return
            
        # Create saliency map directory if it doesn't exist. The main script expects this.
        try:
            Path(salmap_dir).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(salmap_dir):
                QMessageBox.warning(self, "Directory Error", f"Saliency maps directory does not exist: {salmap_dir}")
                return
        except Exception as e:
            QMessageBox.warning(self, "Directory Error", f"Error creating saliency maps directory: {str(e)}")
            return
        
        # Ensure output directory exists
        try:
            Path("results").mkdir(exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Directory Error", f"Error creating results directory: {str(e)}")
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Analysis started...")
        self.start_analysis_button.setEnabled(False)
        self.tab_widget.setCurrentIndex(1) # Switch to Analysis tab

        self.analysis_worker = AnalysisWorker(video_path, dc_file_path, frames_dir, salmap_dir)
        self.analysis_worker.progress_updated.connect(self.on_analysis_progress)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def on_analysis_progress(self, percentage: int, message: str):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def on_analysis_completed(self, attention_scores: List[AttentionScore]):
        self.attention_scores = attention_scores
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis complete!")
        self.start_analysis_button.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", "Directors Cut analysis finished successfully!")
        self.display_results()
        self.tab_widget.setCurrentIndex(2) # Switch to Results tab
        self.update_saliency_display() # Refresh maps if they depend on analysis results
        self.video_widget.update() # Trigger repaint for overlays

        # Also load directors cut data for overlay drawing if not already loaded
        dc_file_path = self.dc_path_label.text()
        if Path(dc_file_path).exists():
            try:
                data = np.loadtxt(dc_file_path)
                self.directors_cut_data = [
                    DirectorsCutFrame(
                        frame_index=int(row[0]),
                        azimuth_pixel=row[1],
                        elevation_pixel=row[2]
                    ) for row in data
                ]
                logger.info(f"Directors Cut data re-loaded for overlay: {len(self.directors_cut_data)} frames")
            except Exception as e:
                logger.error(f"Error re-loading Directors Cut file for overlay: {e}")

    def on_analysis_error(self, error_message: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed!")
        self.start_analysis_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{error_message}")

    def display_results(self):
        """Display the analysis results in the results tab."""
        if not self.attention_scores:
            self.results_text.setText("No analysis results to display.")
            return

        results_str = "Directors Cut Analysis Results:\n\n"
        for score in self.attention_scores:
            results_str += (
                f"Frame {score.frame_index}: "
                f"Overlap = {score.overlap_percentage:.2f} ({score.attention_level.upper()})\n"
            )
        
        results_str += "\nSummary:\n"
        high_attention_frames = sum(1 for s in self.attention_scores if s.attention_level == "high")
        medium_attention_frames = sum(1 for s in self.attention_scores if s.attention_level == "medium")
        low_attention_frames = sum(1 for s in self.attention_scores if s.attention_level == "low")
        
        total_frames_analyzed = len(self.attention_scores)
        if total_frames_analyzed > 0:
            results_str += f"High Attention Frames: {high_attention_frames} ({(high_attention_frames/total_frames_analyzed)*100:.1f}%)\n"
            results_str += f"Medium Attention Frames: {medium_attention_frames} ({(medium_attention_frames/total_frames_analyzed)*100:.1f}%)\n"
            results_str += f"Low Attention Frames: {low_attention_frames} ({(low_attention_frames/total_frames_analyzed)*100:.1f}%)\n"
        
        self.results_text.setText(results_str)

    def export_results(self):
        """Export analysis results to a file."""
        if not self.attention_scores:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return

        try:
            file_dialog = QFileDialog(self)
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilter("JSON Files (*.json);;Text Files (*.txt)")
            file_dialog.setDefaultSuffix("json")
            if file_dialog.exec():
                file_path = file_dialog.selectedFiles()[0]
                if file_path.endswith(".json"):
                    # Export as JSON
                    results_data = {
                        "analysis_results": [
                            {
                                "frame_index": score.frame_index,
                                "overlap_percentage": score.overlap_percentage,
                                "attention_level": score.attention_level,
                                "color_rgb": score.color_rgb
                            }
                            for score in self.attention_scores
                        ]
                    }
                    
                    with open(file_path, 'w') as f:
                        json.dump(results_data, f, indent=2)
                else:
                    # Export as text
                    with open(file_path, 'w') as f:
                        f.write(self.results_text.toPlainText())
                
                QMessageBox.information(self, "Export Complete", 
                                       f"Results exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export results:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())