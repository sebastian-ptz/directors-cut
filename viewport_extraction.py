import numpy as np
from PIL import Image, ImageDraw

# Helper function for vector normalization
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        # Return the zero vector if the norm is zero to avoid division by zero.
        # This case should ideally not happen with valid gaze directions.
        return v
    return v / norm

def to_spherical(x,y) -> Tuple[float, float]:
    """Convert pixel coordinates to spherical coordinates"""
    azimuth = (360.0 / 2880) * x
    elevation = (180.0 / 1440) * y
    return azimuth, elevation



def get_viewport_pixels(
    equirectangular_image: Image.Image,
    gaze_cartesian: tuple[float, float, float],
    h_fov_rad: float,
    v_fov_rad: float,
    viewport_width_px: int = 1024,
    viewport_height_px: int = 768
) -> Image.Image:
    """
    Extracts the pixel values within the viewport of VR glasses from an equirectangular image,
    generating a truly rectilinear (undistorted) projection, as seen through a pinhole camera.

    Args:
        equirectangular_image (PIL.Image.Image): The input equirectangular image.
                                                 It will be converted to RGB format if necessary.
        gaze_cartesian (tuple[float, float, float]): A 3D Cartesian vector (x, y, z)
                                                      representing the direction of the
                                                      user's gaze from the center of the sphere.
                                                      Its magnitude does not does not matter, only its direction.
                                                      Example: (1.0, 0.0, 0.0) for looking
                                                      along the positive X-axis.
        h_fov_rad (float): The horizontal Field of View (FOV) of the VR glasses in radians.
        v_fov_rad (float): The vertical Field of View (FOV) of the VR glasses in radians.
        viewport_width_px (int, optional): The desired pixel width of the output VR
                                           viewport image. Defaults to 1024.
        viewport_height_px (int, optional): The desired pixel height of the output VR
                                            viewport image. Defaults to 768.

    Returns:
        PIL.Image.Image: A new PIL Image object representing the view within the
                         VR glasses viewport.
    """
    # Ensure the input image is in RGB format and convert to a NumPy array for efficient pixel access
    if equirectangular_image.mode != 'RGB':
        equirectangular_image = equirectangular_image.convert('RGB')
    
    equi_pixels = np.array(equirectangular_image)
    equi_width, equi_height = equirectangular_image.size

    # 1. Define Camera Orientation (Basis Vectors) based on Gaze
    # The gaze_vector is the camera's 'forward' direction
    gaze_vector = normalize_vector(np.array(gaze_cartesian, dtype=float))

    # Define the world 'up' vector [0, 0, 1] for Z-up.
    world_up_vector = np.array([0.0, 0.0, 1.0])

    # Calculate camera's 'right' vector using cross product
    # Handle the special case looking straight up/down
    if np.isclose(np.linalg.norm(np.cross(gaze_vector, world_up_vector)), 0):
        # Gaze is straight up for cross product.
        temp_up_for_cross = np.array([0.0, 1.0, 0.0]) # Use Y-axis as temp up
        right_vector = normalize_vector(np.cross(gaze_vector, temp_up_for_cross))
    else:
        right_vector = normalize_vector(np.cross(gaze_vector, world_up_vector))
    
    camera_up_vector = normalize_vector(np.cross(right_vector, gaze_vector))

    # Create an empty NumPy array to store the pixels of the output viewport image
    viewport_pixels_np = np.zeros((viewport_height_px, viewport_width_px, 3), dtype=np.uint8)

    # Pre-calculate half FOV tangents for efficiency in ray construction
    tan_half_h_fov = np.tan(h_fov_rad / 2)
    tan_half_v_fov = np.tan(v_fov_rad / 2)

    # 2. Iterate Through Each Pixel in the Output Viewport Image
    for vy in range(viewport_height_px):
        for vx in range(viewport_width_px):
            # Calculate normalized viewport coordinates.[-1, 1] range
            vx_norm = (vx / viewport_width_px) * 2 - 1
            vy_norm = (vy / viewport_height_px) * 2 - 1

            # 3. Construct the Ray in World Space for the Current Viewport Pixel
            # The vector that points from the camera center to the pixel on the projection plane:
            # This forms the ray direction.
            # Negative `vy_norm` is used because pixel Y increases downwards, but camera_up points upwards.
            world_ray = (gaze_vector +
                         right_vector * (vx_norm * tan_half_h_fov) +
                         camera_up_vector * (-vy_norm * tan_half_v_fov))
            
            # Normalize the world_ray to get a unit direction vector
            world_ray = normalize_vector(world_ray)

            # 4. Convert World Ray to Spherical Coordinates (phi, theta)
            # Latitude (theta): angle from the XY plane towards the Z-axis
            target_theta = np.arcsin(world_ray[2]) 

            # Longitude (phi): angle from the positive X-axis in the XY plane
            target_phi = np.arctan2(world_ray[1], world_ray[0]) 

            # 5. Map Spherical Coordinates to Equirectangular Pixel Coordinates
            # src_x: Maps longitude from [-pi, pi] to pixel column [0, equi_width-1].
            src_x = ((target_phi + np.pi) / (2 * np.pi)) * equi_width
            
            # src_y: Maps latitude from [pi/2 (top) to -pi/2 (bottom)] to pixel row [0, equi_height-1].
            src_y = ((np.pi/2 - target_theta) / np.pi) * equi_height

            # --- 6. Clamp Source Coordinates to Ensure They Are Within Image Bounds ---
            src_x = np.clip(src_x, 0, equi_width - 1)
            src_y = np.clip(src_y, 0, equi_height - 1)

            # --- 7. Sample Pixel from Equirectangular Image (Nearest-Neighbor Interpolation) ---
            viewport_pixels_np[vy, vx] = equi_pixels[int(src_y), int(src_x)]
    return viewport_pixels_np


def draw_viewport_on_equirectangular(
    equirectangular_image: Image.Image,
    gaze_cartesian: tuple[float, float, float],
    h_fov_rad: float,
    v_fov_rad: float,
    viewport_width_px: int = 1024,
    viewport_height_px: int = 768,
    line_color: tuple[int, int, int] = (255, 0, 0), # Red by default
    line_thickness: int = 3,
    num_segments: int = 100 # Number of line segments per edge to approximate curve
) -> Image.Image:
    """
    Draws the rectilinear viewport boundaries onto the equirectangular image.
    The viewport borders will appear curved on the equirectangular projection,
    correctly representing the projection of a rectilinear frame onto a sphere.

    Args:
        equirectangular_image (PIL.Image.Image): The input equirectangular image.
        gaze_cartesian (tuple[float, float, float]): The 3D Cartesian gaze vector.
        h_fov_rad (float): Horizontal Field of View in radians.
        v_fov_rad (float): Vertical Field of View in radians.
        viewport_width_px (int): Width of the conceptual viewport (used for FOV calculations).
        viewport_height_px (int): Height of the conceptual viewport (used for FOV calculations).
        line_color (tuple[int, int, int]): RGB tuple for the line color. Defaults to red.
        line_thickness (int): Thickness of the drawn line. Defaults to 3.
        num_segments (int): Number of line segments to approximate each curved edge.
                            Higher numbers yield smoother curves but are slower.

    Returns:
        PIL.Image.Image: A new PIL Image object with the viewport borders drawn.
    """
    # Create a copy of the image to draw on and ensure RGB mode
    if equirectangular_image.mode != 'RGB':
        equirectangular_image = equirectangular_image.convert('RGB')
    
    drawn_image = equirectangular_image.copy()
    draw = ImageDraw.Draw(drawn_image)

    equi_width, equi_height = equirectangular_image.size

    # --- 1. Define Camera Orientation (Basis Vectors) based on Gaze ---
    gaze_vector = normalize_vector(np.array(gaze_cartesian, dtype=float))
    world_up_vector = np.array([0.0, 0.0, 1.0])

    if np.isclose(np.linalg.norm(np.cross(gaze_vector, world_up_vector)), 0):
        temp_up_for_cross = np.array([0.0, 1.0, 0.0])
        right_vector = normalize_vector(np.cross(gaze_vector, temp_up_for_cross))
    else:
        right_vector = normalize_vector(np.cross(gaze_vector, world_up_vector))
    
    camera_up_vector = normalize_vector(np.cross(right_vector, gaze_vector))

    tan_half_h_fov = np.tan(h_fov_rad / 2)
    tan_half_v_fov = np.tan(v_fov_rad / 2)

    # --- 2. Generate Normalized Viewport Coordinates for Border Points ---
    # These coordinates are then mapped to 3D rays and then to equirectangular pixels.
    border_points_normalized = []

    # Iterate through each edge to generate sample points
    # Top edge: vy_norm = -1, vx_norm from -1 to 1
    for i in range(num_segments + 1):
        vx_norm = -1 + (2 * i / num_segments)
        border_points_normalized.append((vx_norm, -1))
    
    # Right edge: vx_norm = 1, vy_norm from -1 to 1
    for i in range(num_segments + 1):
        vy_norm = -1 + (2 * i / num_segments)
        border_points_normalized.append((1, vy_norm))

    # Bottom edge: vy_norm = 1, vx_norm from 1 to -1 (to connect properly)
    for i in range(num_segments + 1):
        vx_norm = 1 - (2 * i / num_segments)
        border_points_normalized.append((vx_norm, 1))
    
    # Left edge: vx_norm = -1, vy_norm from 1 to -1 (to connect properly)
    for i in range(num_segments + 1):
        vy_norm = 1 - (2 * i / num_segments)
        border_points_normalized.append((-1, vy_norm))

    # --- 3. Convert Each Normalized Viewport Point to Equirectangular Pixel Coordinates ---
    equi_border_pixels = []
    for vx_norm, vy_norm in border_points_normalized:
        # Construct the 3D ray corresponding to this viewport point
        world_ray = (gaze_vector +
                     right_vector * (vx_norm * tan_half_h_fov) +
                     camera_up_vector * (-vy_norm * tan_half_v_fov))
        world_ray = normalize_vector(world_ray)

        # Convert the 3D ray to spherical coordinates (latitude, longitude)
        target_theta = np.arcsin(world_ray[2]) 
        target_phi = np.arctan2(world_ray[1], world_ray[0]) 

        # Map spherical coordinates to equirectangular pixel coordinates
        src_x = ((target_phi + np.pi) / (2 * np.pi)) * equi_width
        src_y = ((np.pi/2 - target_theta) / np.pi) * equi_height

        # Clamp source coordinates to ensure they are within image bounds
        src_x = np.clip(src_x, 0, equi_width - 1)
        src_y = np.clip(src_y, 0, equi_height - 1)
        
        equi_border_pixels.append((src_x, src_y))

    # --- 4. Draw Lines on the Equirectangular Image ---
    # Draw segments between consecutive points. Handle horizontal wrap-around.
    
    # Add the first point to the end to close the loop
    equi_border_pixels.append(equi_border_pixels[0])

    for i in range(len(equi_border_pixels) - 1):
        p1_x, p1_y = equi_border_pixels[i]
        p2_x, p2_y = equi_border_pixels[i+1]

        # Check for horizontal wrap-around (large horizontal jump)
        # This typically indicates crossing the seam at x=0 or x=equi_width
        if abs(p1_x - p2_x) > equi_width / 2:
            # It's a wrap-around. Draw two segments.
            # Segment 1: from p1 to the respective edge (0 or equi_width-1)
            # Segment 2: from the other edge (0 or equi_width-1) to p2
            
            # Interpolate y for the edge crossing to make the line continuous
            # This is a linear interpolation, which is fine for short segments.
            y_at_wrap = p1_y + (p2_y - p1_y) * ((equi_width / 2 - p1_x if p1_x > equi_width / 2 else -p1_x) / (p2_x - p1_x))

            if p1_x < equi_width / 2 and p2_x > equi_width / 2: # Crossing from left to right (e.g., -170 deg to +170 deg)
                draw.line([(p1_x, p1_y), (0, y_at_wrap)], fill=line_color, width=line_thickness)
                draw.line([(equi_width - 1, y_at_wrap), (p2_x, p2_y)], fill=line_color, width=line_thickness)
            elif p1_x > equi_width / 2 and p2_x < equi_width / 2: # Crossing from right to left (e.g., +170 deg to -170 deg)
                draw.line([(p1_x, p1_y), (equi_width - 1, y_at_wrap)], fill=line_color, width=line_thickness)
                draw.line([(0, y_at_wrap), (p2_x, p2_y)], fill=line_color, width=line_thickness)
            else: # Should not happen often, but a fallback if logic above fails
                 draw.line([(p1_x, p1_y), (p2_x, p2_y)], fill=line_color, width=line_thickness)

        else:
            # No wrap-around, draw a single line segment
            draw.line([(p1_x, p1_y), (p2_x, p2_y)], fill=line_color, width=line_thickness)

        #draw.circle((1063.0, 769.400), radius=line_thickness, fill=(255, 0, 0))  # Draw circle at start point
    return drawn_image

# --- Example Usage ---
if __name__ == '__main__':
    equi_image = Image.open("./VideoData/mono_smart_59/frame_04998.jpg")
    # 2. Define VR glasses parameters
    # Gaze direction: Looking along the positive X-axis (center of the image horizontally)
    # and slightly upwards (positive Z)

    # carthesian cords
    # 1062.900	769.400
    cartesianToSphereAngles(1062.900, 769.400, 0)

    gaze_dir = (np.pi, 0.0, 0.0) 

    # Field of View (FOV) in degrees, then converted to radians
    h_fov_deg = 90  # 90 degrees horizontal FOV - common for VR
    v_fov_deg = 70  # 70 degrees vertical FOV
    h_fov_rad = np.deg2rad(h_fov_deg)
    v_fov_rad = np.deg2rad(v_fov_deg)

    # Desired resolution for the viewport image
    output_width = 1024
    output_height = 768

    print(f"Extracting viewport for gaze {gaze_dir}, HFOV={h_fov_deg}°, VFOV={v_fov_deg}°...")

    # 3. Call the function to get the viewport pixels (rectilinear view)
    try:
        viewport_pixels = get_viewport_pixels(
            equirectangular_image=equi_image,
            gaze_cartesian=gaze_dir,
            h_fov_rad=h_fov_rad,
            v_fov_rad=v_fov_rad,
            viewport_width_px=output_width,
            viewport_height_px=output_height
        )
        viewport_image = Image.fromarray(viewport_pixels, 'RGB')
        output_filename_viewport = "vr_viewport_output_corrected.png"
        viewport_image.save(output_filename_viewport)
        print(f"Corrected viewport image saved successfully to {output_filename_viewport}")

        # 4. Call the new function to draw the viewport borders on the equirectangular image
        print(f"Drawing viewport borders on the equirectangular image...")
        equi_with_border_image = draw_viewport_on_equirectangular(
            equirectangular_image=equi_image,
            gaze_cartesian=gaze_dir,
            h_fov_rad=h_fov_rad,
            v_fov_rad=v_fov_rad,
            viewport_width_px=output_width, # These are used to determine the conceptual viewport aspect ratio/FOV
            viewport_height_px=output_height,
            line_color=(0, 255, 0), # Green border
            num_segments=200 # More segments for smoother curves
        )
        output_filename_border = "equirectangular_with_viewport_border.png"
        equi_with_border_image.save(output_filename_border)
        print(f"Equirectangular image with viewport border saved successfully to {output_filename_border}")

        # You can also open the images to view them immediately (requires an image viewer)
        # viewport_image.show()
        # equi_with_border_image.show()

    except Exception as e:
        print(f"An error occurred: {e}")

