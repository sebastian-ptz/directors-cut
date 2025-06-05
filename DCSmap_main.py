import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

"""
    This Script has been translated by Google Gemini from the matlab Script 'story360_DCSmap_main.m'

        Project Story360: Calculation and Visualization of the Director's Cut Similarity Map
        Original Author: Sebastian Knorr
        Type: Main program
        Version: 1.2
        Date: March 14,2018
        Copyright: Sebastian Knorr / Trinity College Dublin
    Date: April 26, 2025
"""

# Helper function to convert spherical coordinates to Cartesian coordinates.
# This function is a standard conversion used in the original MATLAB script.
def sphereTocartesian(theta, phi, r):
    """
    Converts spherical coordinates (theta, phi, r) to Cartesian coordinates (x, y, z).

    Args:
        theta (float): Azimuthal angle in radians.
        phi (float): Polar angle (elevation) in radians.
        r (float): Radius.

    Returns:
        numpy.ndarray: A 3-element array [x, y, z] representing the Cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def calculate_full_overlap_dcs(director_cut_file, scan_path_file, export_path):
    """
    Calculates and visualizes the Director's Cut Similarity Map focusing on
    the full viewport overlap between director's cut and user's scan path.

    Args:
        director_cut_file (str): Path to the director's cut data file (e.g., .txt).
                                 Expected format: [frame_index, azimuth_pixel, elevation_pixel].
        scan_path_file (str): Path to the user's scan path data file (e.g., .txt).
                              Expected format: [image_width, image_height, normalized_azimuth, normalized_elevation].
        export_path (str): Directory where the output image and statistics file will be saved.
    """
    
    # Ensure the export directory exists
    os.makedirs(export_path, exist_ok=True)

    # Load Director's Cut data
    # Assumes data is space-delimited, similar to MATLAB's dlmread.
    try:
        M = np.loadtxt(director_cut_file)
    except Exception as e:
        print(f"Error loading director's cut file '{director_cut_file}': {e}")
        return

    # Load Viewport Scan-Path data
    try:
        scanpath = np.loadtxt(scan_path_file)
    except Exception as e:
        print(f"Error loading scan path file '{scan_path_file}': {e}")
        return

    # Extract parameters from loaded data
    image_width = scanpath[0, 0]
    image_height = scanpath[0, 1]
    frames = len(M)
    
    # Define viewport dimensions (horizontal and vertical Field of View)
    vp_h = 80  # horizontal FOV of Oculus Rift CV1 in degrees
    vp_v = 90  # vertical FOV of Oculus Rift CV1 in degrees
    
    # DCS_height is used for the height of the output similarity map image.
    # It's derived from the original MATLAB code's logic.
    DCS_height = int(np.floor(vp_h * vp_v / 4)) 

    # Calculate spherical coordinates for the director's cut
    # M[:, 1] is azimuth pixel, M[:, 2] is elevation pixel
    director_azimuth = (360 / image_width * M[:, 1])
    director_elevation = (180 / image_height * M[:, 2])
    director_theta = np.deg2rad(director_azimuth) # Convert to radians for spherical math
    director_phi = np.deg2rad(director_elevation) # Convert to radians
    r = 1 # Radius for spherical to Cartesian conversion

    # Calculate spherical coordinates for the user's Scan-Path
    # scanpath[:, 2] is normalized azimuth, scanpath[:, 3] is normalized elevation
    scan_azimuth = (360 * scanpath[:, 2])
    scan_elevation = (180 * scanpath[:, 3])

    # Resample user scan path data to match the number of frames in the director's cut.
    # This is done using linear interpolation, similar to MATLAB's `resample` for this context.
    x_scanpath = np.linspace(0, frames - 1, len(scanpath))
    x_M = np.linspace(0, frames - 1, frames)
    
    # Interpolate user azimuth
    f_azimuth = interp1d(x_scanpath, scan_azimuth, kind='linear', fill_value="extrapolate")
    user_azimuth = f_azimuth(x_M)

    # Interpolate user elevation
    f_elevation = interp1d(x_scanpath, scan_elevation, kind='linear', fill_value="extrapolate")
    user_elevation = f_elevation(x_M)

    user_theta = np.deg2rad(user_azimuth) # Convert to radians
    user_phi = np.deg2rad(user_elevation) # Convert to radians

    # Calculate the difference in azimuth and elevation between user and director
    diff_azimuth = user_azimuth - director_azimuth
    diff_elevation = user_elevation - director_elevation

    # Initialize arrays for storing full overlap percentage and RGB color data
    Npercent_all = np.zeros(frames)
    RGB_all = np.zeros((3, frames))
    DCS_map_all = np.zeros((DCS_height, frames, 3)) # This will store the final image data

    # Loop through each frame to calculate overlap and color code it
    for i in range(frames):
        # Calculate the overlapping width and height.
        # `max(0, ...)` ensures that overlap is not negative.
        # `vp_h - abs(diff_azimuth[i])` calculates the overlapping width.
        overlap_azimuth = max(0, vp_h - abs(diff_azimuth[i]))
        overlap_elevation = max(0, vp_v - abs(diff_elevation[i]))
        
        # Calculate the area of overlap
        N_overlap_area = overlap_azimuth * overlap_elevation
        
        # Calculate the percentage of total viewport overlap
        Npercent_all[i] = N_overlap_area / (vp_h * vp_v)

        # Color code the overlap percentage (from blue-green to red)
        # This is a direct translation of the MATLAB color mapping logic.
        if Npercent_all[i] >= 0.5:
            RGB_all[0, i] = 255 * (2 * (Npercent_all[i] - 0.5)) # Red component
            RGB_all[1, i] = 255 * (2 * (1 - Npercent_all[i]))   # Green component
            RGB_all[2, i] = 0                                   # Blue component
        else:
            RGB_all[0, i] = 0                                   # Red component
            RGB_all[1, i] = 255 * (2 * (Npercent_all[i]))       # Green component
            RGB_all[2, i] = 255 * (2 * (0.5 - Npercent_all[i])) # Blue component
        
        # Populate the DCS map for the full viewport with the calculated RGB values
        # The entire height of DCS_map_all for the current frame 'i' will have the same color.
        DCS_map_all[:, i, 0] = RGB_all[0, i]
        DCS_map_all[:, i, 1] = RGB_all[1, i]
        DCS_map_all[:, i, 2] = RGB_all[2, i]

    # Calculate statistics for the full overlap percentage
    Npercent_stat_mean = np.mean(Npercent_all)
    Npercent_stat_std = np.std(Npercent_all)
    Npercent_stat_var = np.var(Npercent_all)

    # --- Output and Visualization ---

    # Extract base filename for naming output files
    base_filename = os.path.basename(director_cut_file)
    name_without_ext = os.path.splitext(base_filename)[0]

    # Display and save the Director's Cut Similarity Map (full viewport)
    plt.figure(figsize=(30, 5)) # Set figure size for better visualization
    
    # imshow expects image data, `astype(np.uint8)` is crucial for correct display
    plt.imshow(DCS_map_all.astype(np.uint8), aspect='auto') 
    
    plt.xlim(0, frames) # Set x-axis limits to match frames
    plt.gca().set_xticks([]) # Remove x-axis ticks
    plt.gca().set_yticks([]) # Remove y-axis ticks
    
    # Add minor grid lines and ticks for frames, mimicking MATLAB's 'XMinorGrid' and 'XMinorTick'
    plt.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5) 
    plt.tick_params(axis='x', which='minor', bottom=True) 
    
    plt.title(base_filename)
    plt.xlabel('Frames', fontsize=14)
    
    # Construct the export path for the image
    export_filename_dmf = os.path.join(export_path, name_without_ext + '_dmf.png')
    plt.savefig(export_filename_dmf, bbox_inches='tight') # Save the figure
    plt.close() # Close the plot to free memory
    print(f"Director's Cut Similarity Map (full overlap) saved to: {export_filename_dmf}")

    # Output statistics for viewport overlap to a text file
    text_filename = name_without_ext + '_Stat.txt'
    export_text_file = os.path.join(export_path, text_filename)
    with open(export_text_file, 'w') as f:
        f.write(f"{'Mean':<6}\t {'Std.':<6}\t {'Var.':<6}\n")
        f.write(f"{Npercent_stat_mean:<6.4f}\t {Npercent_stat_std:<6.4f}\t {Npercent_stat_var:<6.4f}\n")
    print(f"Overlap statistics saved to: {export_text_file}")

# --- Example Usage (Uncomment and modify with your actual file paths) ---
# To run this code, you would typically have two input data files:
# 1. A director's cut file (e.g., `directors_cut.txt`)
#    Example content (frame_index, azimuth_pixel, elevation_pixel):
#    0 960 540
#    1 970 550
#    ...
# 2. A scan path file (e.g., `scan_path.txt`)
#    Example content (image_width, image_height, normalized_azimuth, normalized_elevation):
#    1920 1080 0.5 0.5
#    1920 1080 0.51 0.52
#    ...

# For demonstration, you can create dummy files:
# np.savetxt('dummy_director_cut.txt', np.column_stack((np.arange(100), np.random.rand(100, 2) * [1920, 1080])))
# np.savetxt('dummy_scan_path.txt', np.column_stack((np.full((100, 1), 1920), np.full((100, 1), 1080), np.random.rand(100, 2))))

director_cut_file_path = '/home/anton/Code/Uni/Project 2/DirectorsCut/DC-war.txt'
scan_path_file_path = '/home/anton/Code/Uni/Project 2/SubjectiveData/participant01/participant01-war.txt'
results_export_directory = './results'

calculate_full_overlap_dcs(director_cut_file_path, scan_path_file_path, results_export_directory)

#----------------------------------------------------------------
#/home/anton/Code/Uni/Project 2/DirectorsCut/DC-war.txt
#/home/anton/Code/Uni/Project 2/SubjectiveData/participant01/participant01-war.txt