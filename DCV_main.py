"""
Directors Cut Video Analysis Application
Structured architecture for analyzing 360° video attention and saliency overlap
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from FastVRViewportExtractor import VRViewportExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

# FastSal imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import model.fastSal as fastsal
from dataset.utils import read_vgg_img
from utils import load_weight
from generate_img import post_process_png

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    azimuth_pixel: float    # TODO: consider changing name to match cartesian coordinates
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

class FastSalDataset(Dataset):
    """Dataset class for FastSal prediction on individual images"""
    
    def __init__(self, image_paths: List[str], output_paths: List[str]):
        self.image_paths = image_paths
        self.output_paths = output_paths
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        output_path = self.output_paths[idx]
        vgg_img, original_size = read_vgg_img(img_path, (192, 256))
        return vgg_img, original_size, output_path
        
    def __len__(self):
        return len(self.image_paths)

class SaliencyGenerator(ABC):
    """Abstract base class for saliency map generators"""
    
    @abstractmethod
    def generate_saliency_map(self, frame_path: str, output_path: str) -> bool:
        """Generate saliency map for a given frame"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the saliency generator is available"""
        pass

class FastSalGenerator(SaliencyGenerator):
    """FastSal saliency map generator implementation"""
    
    def __init__(self, model_type: str = 'A', finetune_dataset: str = 'salicon', gpu: bool = True):
        self.model_type = model_type
        self.finetune_dataset = finetune_dataset
        self.gpu = gpu and torch.cuda.is_available()
        self.model = None
        self.batch_size = 4  # Adjust based on GPU memory
        self._check_fastsal_availability()
    
    def _check_fastsal_availability(self):
        """Check if FastSal is available and load model"""
        try:
            # Load FastSal model
            self.model = fastsal.fastsal(pretrain_mode=False, model_type=self.model_type)
            
            # Load weights
            weights_path = f'weights/{self.finetune_dataset}_{self.model_type}.pth'
            if not os.path.exists(weights_path):
                logger.error(f"FastSal weights not found: {weights_path}")
                self.model = None
                return
                
            state_dict, opt_state = load_weight(weights_path, remove_decoder=False)
            self.model.load_state_dict(state_dict)
            
            if self.gpu:
                self.model.cuda()
                
            self.model.eval()  # Set to evaluation mode
            logger.info(f"FastSal model loaded successfully (Type: {self.model_type}, Dataset: {self.finetune_dataset}, GPU: {self.gpu})")
            
        except Exception as e:
            logger.error(f"FastSal not available: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def generate_saliency_map(self, frame_path: str, output_path: str) -> bool:
        """Generate saliency map using FastSal for a single frame"""
        if not self.is_available():
            logger.error("FastSal not available")
            return False
        
        try:
            # Create dataset for single image
            dataset = FastSalDataset([frame_path], [output_path])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            with torch.no_grad():
                for x, original_size_list, output_path_list in dataloader:
                    if self.gpu:
                        x = x.float().cuda()
                    
                    # Generate prediction
                    y = self.model(x)
                    y = nn.Sigmoid()(y)  # Apply sigmoid for final output
                    
                    if self.gpu:
                        y = y.detach().cpu()
                    
                    y = y.numpy()
                    
                    # Process the prediction
                    prediction = y[0, 0, :, :]  # First batch, first channel
                    original_size = original_size_list[0].numpy()
                    img_output_path = output_path_list[0]
                    
                    # Post-process and save
                    img_data = post_process_png(prediction, original_size)
                    cv2.imwrite(img_output_path, img_data)
                    
                    logger.info(f"Saliency map generated: {img_output_path}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error generating saliency map: {e}")
            return False
        
        return False
    
    def generate_batch_saliency_maps(self, frame_paths: List[str], output_paths: List[str]) -> List[bool]:
        """Generate saliency maps for multiple frames in batches"""
        if not self.is_available():
            logger.error("FastSal not available")
            return [False] * len(frame_paths)
        
        if len(frame_paths) != len(output_paths):
            logger.error("Mismatch between frame paths and output paths")
            return [False] * len(frame_paths)
        
        results = []
        
        try:
            # Create dataset
            dataset = FastSalDataset(frame_paths, output_paths)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            with torch.no_grad():
                for x, original_size_list, output_path_list in dataloader:
                    if self.gpu:
                        x = x.float().cuda()
                    
                    # Generate predictions
                    y = self.model(x)
                    y = nn.Sigmoid()(y)  # Apply sigmoid for final output
                    
                    if self.gpu:
                        y = y.detach().cpu()
                    
                    y = y.numpy()
                    
                    # Process each prediction in the batch
                    for i in range(y.shape[0]):
                        try:
                            prediction = y[i, 0, :, :]
                            original_size = original_size_list[i].numpy()
                            img_output_path = output_path_list[i]
                            
                            # Post-process and save
                            img_data = post_process_png(prediction, original_size)
                            cv2.imwrite(img_output_path, img_data)
                            
                            results.append(True)
                            logger.info(f"Saliency map generated: {img_output_path}")
                            
                        except Exception as e:
                            logger.error(f"Error processing batch item {i}: {e}")
                            results.append(False)
                            
        except Exception as e:
            logger.error(f"Error in batch saliency generation: {e}")
            # Fill remaining results with False
            results.extend([False] * (len(frame_paths) - len(results)))
        
        return results

class ViewportExtractor:
    """Handles viewport extraction using FastVRViewportExtractor"""
    
    def __init__(self, h_fov_deg: float = 90, v_fov_deg: float = 70):
        self.h_fov_deg = h_fov_deg
        self.v_fov_deg = v_fov_deg
        self.current_extractor = None
        self.current_image_path = None
        
    def extract_viewport_pixels(
        self, 
        equirectangular_image: Image.Image,
        viewport_center: Tuple[float, float],
        viewport_width_px: int = 1024,
        viewport_height_px: int = 768
    ) -> np.ndarray:
        """Extract viewport pixels using FastVRViewportExtractor"""
        
        # Validate viewport center coordinates
        if isinstance(equirectangular_image, Image.Image):
            img_width, img_height = equirectangular_image.size
        else:
            # If it's a path, load to get dimensions
            temp_img = Image.open(str(equirectangular_image))
            img_width, img_height = temp_img.size
        
        # Ensure viewport center is within image bounds
        center_x = max(0, min(img_width - 1, viewport_center[0]))
        center_y = max(0, min(img_height - 1, viewport_center[1]))
        
        logger.debug(f"Image size: {img_width}x{img_height}")
        logger.debug(f"Viewport center (original): {viewport_center}")
        logger.debug(f"Viewport center (clamped): {center_x}, {center_y}")
        
        # Save image to temp file if needed
        if isinstance(equirectangular_image, Image.Image):
            temp_path = "temp_equirect.png"
            equirectangular_image.save(temp_path)
            image_path = temp_path
        else:
            image_path = str(equirectangular_image)
        
        # Create new extractor if image changed
        if self.current_image_path != image_path:
            self.current_extractor = VRViewportExtractor(image_path)
            self.current_image_path = image_path
            
        # Prepare VR specs with validated dimensions
        vr_specs = {
            'horizontal_fov': self.h_fov_deg,
            'vertical_fov': self.v_fov_deg,
            'width': viewport_width_px,
            'height': viewport_height_px
        }
        
        center_px = int(center_x)
        center_py = int(center_y)
        
        try:
            # Extract viewport
            viewport = self.current_extractor.extract_viewport_vectorized(
                (center_px, center_py),
                vr_specs
            )
            
            # Convert to numpy array and ensure proper format
            viewport_array = np.array(viewport)
            
            logger.debug(f"Extracted viewport shape: {viewport_array.shape}")
            logger.debug(f"Viewport value range: {viewport_array.min()} - {viewport_array.max()}")
            
            return viewport_array
            
        except Exception as e:
            logger.error(f"Error in viewport extraction: {e}")
            # Return a zero array with expected dimensions
            return np.zeros((viewport_height_px, viewport_width_px), dtype=np.float32)
    
    def _normalize_vector(self, v):
        """Normalize a vector"""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

class FrameExtractor:
    """Extracts frames from video files for analysis"""
    
    def __init__(self, output_base_dir: str = "VideoData"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
    def extract_frames(self, video_path: str) -> Tuple[str, bool]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple[str, bool]: (output_directory_path, success)
        """
        if not Path(video_path).exists():
            logger.error(f"Input video file not found: {video_path}")
            return "", False

        try:
            # Get video information
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Could not open video file for frame extraction")
                return "", False
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output directory for this video
            video_name = Path(video_path).stem
            output_video_dir = self.output_base_dir / f"{video_name}"
            output_video_dir.mkdir(exist_ok=True)
            
            logger.info(f"Extracting frames to: {output_video_dir}")
            logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
            
            # Extract frames
            frame_count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_filename = output_video_dir / f"frame_{frame_count:05d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Extracted {frame_count}/{total_frames} frames...")
            
            cap.release()
            logger.info(f"Successfully extracted {frame_count} frames")
            
            return str(output_video_dir), True
            
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
            return "", False
    
    def check_existing_frames(self, video_path: str) -> Tuple[str, bool]:
        """
        Check if frames for this video already exist.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple[str, bool]: (frames_directory_path, exists)
        """
        try:
            video_name = Path(video_path).stem
            
            # Check for existing frame directories
            for dir_path in self.output_base_dir.glob(f"{video_name}_*"):
                if dir_path.is_dir():
                    # Verify frames exist
                    frame_files = list(dir_path.glob("frame_*.jpg"))
                    if frame_files:
                        logger.info(f"Found existing frames in: {dir_path}")
                        return str(dir_path), True
            
            return "", False
            
        except Exception as e:
            logger.error(f"Error checking existing frames: {e}")
            return "", False

class OverlapCalculator:
    """Calculates overlap between viewport and saliency maps"""
    
    def __init__(self, viewport_extractor: ViewportExtractor):
        self.viewport_extractor = viewport_extractor
    
    def calculate_saliency_overlap(
        self,
        saliency_map: Image.Image,
        viewport_center: Tuple[float, float]
    ) -> float:
        """Calculate overlap percentage between viewport and saliency map"""
        
        # Convert PIL Image to numpy array for consistent processing
        saliency_array = np.array(saliency_map)
        
        # Handle different image formats
        if len(saliency_array.shape) == 3:
            # If RGB/RGBA, convert to grayscale
            if saliency_array.shape[2] == 3:
                saliency_array = np.mean(saliency_array, axis=2)
            elif saliency_array.shape[2] == 4:
                saliency_array = np.mean(saliency_array[:, :, :3], axis=2)
        
        # Normalize saliency values to [0, 1] range
        if saliency_array.max() > 1.0:
            saliency_array = saliency_array / 255.0
        
        # Extract viewport from saliency map
        try:
            viewport_pixels = self.viewport_extractor.extract_viewport_pixels(
                saliency_map, viewport_center
            )
            
            # Ensure viewport_pixels is a numpy array
            if not isinstance(viewport_pixels, np.ndarray):
                viewport_pixels = np.array(viewport_pixels)
            
            # Handle different viewport formats
            if len(viewport_pixels.shape) == 3:
                # If RGB/RGBA, convert to grayscale
                if viewport_pixels.shape[2] == 3:
                    viewport_pixels = np.mean(viewport_pixels, axis=2)
                elif viewport_pixels.shape[2] == 4:
                    viewport_pixels = np.mean(viewport_pixels[:, :, :3], axis=2)
            
            # Normalize viewport values to [0, 1] range
            if viewport_pixels.max() > 1.0:
                viewport_pixels = viewport_pixels / 255.0
            
        except Exception as e:
            logger.error(f"Error extracting viewport: {e}")
            return 0.0
        
        # Calculate saliency values
        viewport_saliency_sum = np.sum(viewport_pixels)
        total_saliency_sum = np.sum(saliency_array)
        
        # Debug logging
        logger.debug(f"Viewport saliency sum: {viewport_saliency_sum}")
        logger.debug(f"Total saliency sum: {total_saliency_sum}")
        logger.debug(f"Viewport shape: {viewport_pixels.shape}")
        logger.debug(f"Saliency map shape: {saliency_array.shape}")
        
        # Calculate overlap percentage
        if total_saliency_sum > 0:
            overlap_percentage = viewport_saliency_sum / total_saliency_sum
        else:
            overlap_percentage = 0.0
        
        # Ensure the result is within valid bounds
        overlap_percentage = max(0.0, min(1.0, overlap_percentage))
        
        return overlap_percentage

class AttentionAnalyzer:
    """Analyzes attention based on overlap scores"""
    
    @staticmethod
    def calculate_attention_score(overlap_percentage: float) -> AttentionScore:
        """Convert overlap percentage to attention score with color coding"""
        
        color_rgb = AttentionAnalyzer.get_dcs_color(overlap_percentage)
        
        return AttentionScore(
            frame_index=-1,  # Will be set by caller
            overlap_percentage=overlap_percentage,
            color_rgb=color_rgb
        )
    
    @staticmethod
    def generate_attention_map(
        attention_scores: List[AttentionScore],
        output_path: str,
        map_height: int = 100
    ):
        """Generate visual attention map"""
        if not attention_scores:
            return
        
        num_frames = len(attention_scores)
        map_array = np.zeros((map_height, num_frames, 3), dtype=np.uint8)
        
        for i, score in enumerate(attention_scores):
            map_array[:, i] = score.color_rgb
        
        map_image = Image.fromarray(map_array)
        map_image.save(output_path)
        logger.info(f"Attention map saved to: {output_path}")

    @staticmethod
    def get_dcs_color(overlap_percentage: float) -> Tuple[int, int, int]:
        """
        Calculates the RGB color tuple based on the overlap percentage for the Director's Cut Similarity Map.
        The color coding transitions from blue-green (low overlap) to red (high overlap).

        Args:
            overlap_percentage (float): The calculated overlap percentage, ranging from 0.0 to 1.0.

        Returns:
            Tuple[int, int, int]: An RGB color tuple (0-255, 0-255, 0-255).
        """
        R, G, B = 0, 0, 0

        if overlap_percentage >= 0.5:
            # From green (0.5) to red (1.0)
            R = int(255 * (2 * (overlap_percentage - 0.5)))
            G = int(255 * (2 * (1 - overlap_percentage)))
            B = 0
        else:
            # From blue (0.0) to green (0.5)
            R = 0
            G = int(255 * (2 * overlap_percentage))
            B = int(255 * (2 * (0.5 - overlap_percentage)))
        
        # Ensure values are within 0-255 range
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))

        return (R, G, B)

class DirectorsCutProcessor:
    """Main processor for Directors Cut analysis"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.saliency_generator = FastSalGenerator()
        self.viewport_extractor = ViewportExtractor()
        self.overlap_calculator = OverlapCalculator(self.viewport_extractor)
        self.attention_analyzer = AttentionAnalyzer()
        self.frame_extractor = FrameExtractor(output_base_dir="VideoData")
        
        self.video_metadata: Optional[VideoMetadata] = None
        self.directors_cut_data: List[DirectorsCutFrame] = []
    
    def load_video(self, video_path: str) -> bool:
        """Load video and extract metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0
            
            self.video_metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_ms=duration_ms
            )
            
            cap.release()
            logger.info(f"Video loaded: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False
    
    def load_directors_cut(self, dc_file_path: str) -> bool:
        """Load Directors Cut data from file"""
        try:
            data = np.loadtxt(dc_file_path)
            self.directors_cut_data = [
                DirectorsCutFrame(
                    frame_index=int(row[0]),
                    azimuth_pixel=row[1],
                    elevation_pixel=row[2]
                )
                for row in data
            ]
            
            logger.info(f"Directors Cut data loaded: {len(self.directors_cut_data)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Directors Cut file: {e}")
            return False
    
    def check_saliency_maps(self, saliency_dir: str) -> Dict[int, str]:
        """Check for existing saliency maps"""
        saliency_path = Path(saliency_dir)
        existing_maps = {}
        
        if saliency_path.exists():
            # Matches png, jpg, jpeg files
            for file_path in saliency_path.glob("*.[pj][np][gj]*"):  # This line was Generated by Claude Sonnet 3.5
                try:
                    # Assuming naming convention: out_frame_XXXXX.ext
                    filename = file_path.stem
                    if filename.startswith('out_frame_'):
                        frame_num = int(filename.split('_')[-1])
                        existing_maps[frame_num] = str(file_path)
                except ValueError:
                    continue
        
        logger.info(f"Found {len(existing_maps)} existing saliency maps")
        return existing_maps
    
    def check_saliency_maps_complete(self, saliency_dir: Path) -> Tuple[bool, List[int]]:
        """
        Check if all required frames from directors_cut_data have saliency maps.
        
        Args:
            saliency_dir: Directory containing saliency maps
            
        Returns:
            Tuple[bool, List[int]]: (is_complete, missing_frame_numbers)
        """
        if not self.directors_cut_data:
            logger.error("No directors cut data loaded")
            return False, []
            
        missing_frames = []
        saliency_dir = Path(saliency_dir)
        
        for dc_frame in self.directors_cut_data:
            frame_idx = dc_frame.frame_index
            frame_exists = False
            
            # Check for file with any common image extension
            for ext in ['.jpg', '.jpeg', '.png']:
                if (saliency_dir / f"out_frame_{frame_idx:05d}{ext}").exists():
                    frame_exists = True
                    break
                    
            if not frame_exists:
                missing_frames.append(frame_idx)
        
        is_complete = len(missing_frames) == 0
        if is_complete:
            logger.info("All required saliency maps present")
        else:
            logger.warning(f"Missing {len(missing_frames)} saliency maps")
            
        return is_complete, sorted(missing_frames)

    def generate_missing_saliency_maps(
        self, 
        video_path: str,
        video_frames_dir: str, 
        saliency_output_dir: str,
        existing_maps: Dict[int, str]
    ) -> Dict[int, str]:
        """Generate missing saliency maps using FastSal with batch processing"""
        saliency_output_path = Path(saliency_output_dir)
        saliency_output_path.mkdir(exist_ok=True)
        
        # Check which frames need to be processed
        is_complete, missing_frames = self.check_saliency_maps_complete(saliency_output_path)
        if is_complete:
            logger.info("All saliency maps present, skipping generation")
            return existing_maps
        
        all_maps = existing_maps.copy()
        frames_dir = Path(video_frames_dir)
        
        if not frames_dir.exists():
            logger.warning(f"Video frames directory not found: {frames_dir}")
            frames_path, success = self.frame_extractor.extract_frames(video_path)
            if not success:
                logger.error("Failed to extract video frames")
                return all_maps
            frames_dir = Path(frames_path)
        
        # Check if frames directory is empty
        if not any(frames_dir.iterdir()):
            logger.info("Frames directory is empty, extracting frames...")
            frames_path, success = self.frame_extractor.extract_frames(video_path)
            if not success:
                logger.error("Failed to extract video frames")
                return all_maps
            frames_dir = Path(frames_path)
        elif not any(frames_dir.glob("frame_*.jpg")):
            logger.error("No frame files found in frames directory")
            return all_maps
        
        # Prepare batch processing data
        frame_paths = []
        output_paths = []
        frame_indices = []
        
        for frame_idx in missing_frames:
            # Check if the frame file exists
            frame_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = frames_dir / f"frame_{frame_idx:05d}{ext}"
                if candidate.exists():
                    frame_file = candidate
                    break
                    
            if frame_file is None:
                logger.warning(f"Source frame {frame_idx} not found")
                continue
            
            output_file = saliency_output_path / f"out_frame_{frame_idx:05d}.png"
            frame_paths.append(str(frame_file))
            output_paths.append(str(output_file))
            frame_indices.append(frame_idx)
        
        if not frame_paths:
            logger.warning("No valid frames found for saliency generation")
            return all_maps
        
        logger.info(f"Generating saliency maps for {len(frame_paths)} frames using FastSal batch processing...")
        
        # Process in batches using FastSal batch generation
        batch_size = 8  # Adjust based on GPU memory
        for i in range(0, len(frame_paths), batch_size):
            batch_frame_paths = frame_paths[i:i+batch_size]
            batch_output_paths = output_paths[i:i+batch_size]
            batch_frame_indices = frame_indices[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(frame_paths) + batch_size - 1)//batch_size}")
            
            # Generate saliency maps for this batch
            results = self.saliency_generator.generate_batch_saliency_maps(
                batch_frame_paths, batch_output_paths
            )
            
            # Update the maps dictionary for successful generations
            for j, (success, frame_idx, output_path) in enumerate(zip(results, batch_frame_indices, batch_output_paths)):
                if success:
                    all_maps[frame_idx] = output_path
                    logger.info(f"Generated saliency map for frame {frame_idx}")
                else:
                    logger.error(f"Failed to generate saliency map for frame {frame_idx}")
        
        # Final verification
        is_complete, still_missing = self.check_saliency_maps_complete(saliency_output_path)
        if not is_complete:
            logger.warning(f"Some frames still missing after generation: {still_missing}")
        else:
            logger.info("All required saliency maps generated successfully")
        
        return all_maps
    
    def process_video_analysis(
        self,
        video_path: str,
        dc_file_path: str,
        video_frames_dir: str,
        saliency_dir: str = "saliency_maps"
    ) -> List[AttentionScore]:
        """Main processing pipeline"""
        
        # Step 1: Load video and directors cut data
        if not self.load_video(video_path):
            return []
        
        if not self.load_directors_cut(dc_file_path):
            return []
        
        # Step 2: Check for existing saliency maps
        existing_maps = self.check_saliency_maps(saliency_dir)
        
        # Step 3: Generate missing saliency maps
        all_saliency_maps = self.generate_missing_saliency_maps(
             video_path, video_frames_dir, saliency_dir, existing_maps
        )
        
        # Step 4: Calculate overlaps and attention scores
        attention_scores = []
        
        

        # Prepare frame data for processing
        frame_data = [
            (dc_frame, all_saliency_maps.get(dc_frame.frame_index))
            for dc_frame in self.directors_cut_data
            if dc_frame.frame_index in all_saliency_maps
        ]
        
        # Process frames in parallel using ThreadPoolExecutor
        attention_scores = []
        batch_size = 100  # Adjust based on available memory
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(0, len(frame_data), batch_size):
                batch = frame_data[i:i + batch_size]
                futures = [executor.submit(self.process_frame, data) for data in batch]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        attention_scores.append(result)
                        logger.info(
                            f"Frame {result.frame_index}: {result.overlap_percentage:.2%} overlap, "
                            f"{result.attention_level} attention"
                        )
        
        # Step 5: Generate attention map
        video_title = Path(video_path).stem
        if attention_scores:
            output_map_path = self.output_dir / f"{video_title}_attention_map.png"
            self.attention_analyzer.generate_attention_map(
                attention_scores, str(output_map_path)
            )
            
            # Save detailed results
            self._save_results(attention_scores)
        
        return attention_scores
    
    # Function to process each frame in parallel
    def process_frame(self, frame_data):
            dc_frame, saliency_map_path = frame_data
            try:
                # Load saliency map
                saliency_map = Image.open(saliency_map_path)
                
                # Calculate overlap
                overlap_percentage = self.overlap_calculator.calculate_saliency_overlap(
                    saliency_map, (dc_frame.azimuth_pixel, dc_frame.elevation_pixel)
                )
                
                # Create attention score
                attention_score = self.attention_analyzer.calculate_attention_score(
                    overlap_percentage
                )
                attention_score.frame_index = dc_frame.frame_index
                
                return attention_score
                
            except Exception as e:
                logger.error(f"Error processing frame {dc_frame.frame_index}: {e}")
            return None
    
    def _save_results(self, attention_scores: List[AttentionScore]):
        """Save detailed analysis results"""
        results = {
            "metadata": {
                "total_frames_analyzed": len(attention_scores),
                "video_metadata": {
                    "width": self.video_metadata.width,
                    "height": self.video_metadata.height,
                    "fps": self.video_metadata.fps,
                    "total_frames": self.video_metadata.total_frames
                } if self.video_metadata else None
            },
            "attention_scores": [
                {
                    "frame_index": score.frame_index,
                    "overlap_percentage": score.overlap_percentage,
                    "attention_level": score.attention_level,
                    "color_rgb": score.color_rgb
                }
                for score in attention_scores
            ],
            "statistics": self._calculate_statistics(attention_scores)
        }
        
        results_file = self.output_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
    
    def _calculate_statistics(self, attention_scores: List[AttentionScore]) -> Dict[str, Any]:
        """Calculate analysis statistics"""
        if not attention_scores:
            return {}
        
        overlaps = [score.overlap_percentage for score in attention_scores]
        
        attention_counts = {"high": 0, "medium": 0, "low": 0}
        for score in attention_scores:
            attention_counts[score.attention_level] += 1
        
        return {
            "overlap_mean": np.mean(overlaps),
            "overlap_std": np.std(overlaps),
            "overlap_min": np.min(overlaps),
            "overlap_max": np.max(overlaps),
            "attention_distribution": attention_counts,
            "high_attention_percentage": (attention_counts["high"] / len(attention_scores)) * 100
        }

# Main application class that integrates with GUI
class DirectorsCutApplication:
    """Main application class"""
    
    def __init__(self):
        self.processor = DirectorsCutProcessor()
    
    def run_analysis(
        self,
        video_path: str,
        dc_file_path: str,
        video_frames_dir: str,
        saliency_maps_dir: str,
        output_dir: str = "results"
    ) -> List[AttentionScore]:
        """Run the complete analysis pipeline"""
        
        self.processor.output_dir = Path(output_dir)
        self.processor.output_dir.mkdir(exist_ok=True)
        
        logger.info("Starting Directors Cut analysis...")
        
        attention_scores = self.processor.process_video_analysis(
            video_path=video_path,
            dc_file_path=dc_file_path,
            video_frames_dir=video_frames_dir,
            saliency_dir=saliency_maps_dir
        )
        
        logger.info(f"Analysis complete. Processed {len(attention_scores)} frames.")
        return attention_scores

def debug_overlap_calculation(
    saliency_map_path: str,
    viewport_center: Tuple[float, float],
    viewport_extractor: ViewportExtractor
) -> Dict[str, Any]:
    """
    Debug function to analyze overlap calculation step by step
    """
    try:
        # Load saliency map
        saliency_map = Image.open(saliency_map_path)
        saliency_array = np.array(saliency_map)
        
        print(f"Saliency map path: {saliency_map_path}")
        print(f"Saliency map size: {saliency_map.size}")
        print(f"Saliency array shape: {saliency_array.shape}")
        print(f"Saliency array dtype: {saliency_array.dtype}")
        print(f"Saliency value range: {saliency_array.min()} - {saliency_array.max()}")
        
        # Handle different image formats
        if len(saliency_array.shape) == 3:
            if saliency_array.shape[2] == 3:
                saliency_array = np.mean(saliency_array, axis=2)
            elif saliency_array.shape[2] == 4:
                saliency_array = np.mean(saliency_array[:, :, :3], axis=2)
        
        # Normalize if needed
        if saliency_array.max() > 1.0:
            saliency_array = saliency_array / 255.0
        
        print(f"Processed saliency array shape: {saliency_array.shape}")
        print(f"Processed saliency value range: {saliency_array.min()} - {saliency_array.max()}")
        
        # Extract viewport
        viewport_pixels = viewport_extractor.extract_viewport_pixels(
            saliency_map, viewport_center
        )
        
        print(f"Viewport center: {viewport_center}")
        print(f"Viewport pixels shape: {viewport_pixels.shape}")
        print(f"Viewport pixels dtype: {viewport_pixels.dtype}")
        print(f"Viewport value range: {viewport_pixels.min()} - {viewport_pixels.max()}")
        
        # Handle viewport format
        if len(viewport_pixels.shape) == 3:
            if viewport_pixels.shape[2] == 3:
                viewport_pixels = np.mean(viewport_pixels, axis=2)
            elif viewport_pixels.shape[2] == 4:
                viewport_pixels = np.mean(viewport_pixels[:, :, :3], axis=2)
        
        # Normalize if needed
        if viewport_pixels.max() > 1.0:
            viewport_pixels = viewport_pixels / 255.0
        
        # Calculate sums
        viewport_saliency_sum = np.sum(viewport_pixels)
        total_saliency_sum = np.sum(saliency_array)
        
        print(f"Viewport saliency sum: {viewport_saliency_sum}")
        print(f"Total saliency sum: {total_saliency_sum}")
        
        # Calculate overlap
        if total_saliency_sum > 0:
            overlap_percentage = viewport_saliency_sum / total_saliency_sum
        else:
            overlap_percentage = 0.0
        
        print(f"Raw overlap percentage: {overlap_percentage}")
        print(f"Clamped overlap percentage: {max(0.0, min(1.0, overlap_percentage))}")
        
        return {
            "saliency_shape": saliency_array.shape,
            "saliency_range": (saliency_array.min(), saliency_array.max()),
            "viewport_shape": viewport_pixels.shape,
            "viewport_range": (viewport_pixels.min(), viewport_pixels.max()),
            "viewport_sum": viewport_saliency_sum,
            "total_sum": total_saliency_sum,
            "raw_overlap": overlap_percentage,
            "final_overlap": max(0.0, min(1.0, overlap_percentage))
        }
        
    except Exception as e:
        print(f"Error in debug function: {e}")
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Example of how to use the application
    app = DirectorsCutApplication()
    
    # Define standart paths 
    video_path = "videos/mono_smart.mp4"
    dc_file_path = "DirectorsCut/DC-smart.txt"
    video_frames_dir = "VideoData/mono_smart"
    saliency_maps_dir = "saliency_maps"
    output_dir = "results"
    
    # Run analysis
    results = app.run_analysis(
        video_path=video_path,
        dc_file_path=dc_file_path,
        video_frames_dir=video_frames_dir,
        saliency_maps_dir=saliency_maps_dir,
        output_dir=output_dir
    )
    
    print(f"Analysis completed with {len(results)} frames processed.")