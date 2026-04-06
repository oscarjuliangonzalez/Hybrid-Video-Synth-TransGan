import cv2
from pathlib import Path

def extract_frames_from_avi(video_path, output_folder):
    """
    Extract all frames from an AVI video file and save them as PNG images.
    
    Args:
        video_path (str): Path to the .avi video file
        output_folder (str): Folder where frames will be saved
        
    Returns:
        int: Number of frames extracted
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    video_name = Path(video_path).stem  # Get filename without extension
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save the frame as PNG
        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")
    
    return frame_count

def process_avi_folder(input_folder, output_base_folder):
    """
    Process all .avi files in a folder, extracting frames from each video.
    Creates a subfolder for each video's frames.
    
    Args:
        input_folder (str): Folder containing .avi files
        output_base_folder (str): Base folder where subfolders for each video will be created
    
    Returns:
        dict: Dictionary with video names as keys and frame counts as values
    """
    Path(output_base_folder).mkdir(parents=True, exist_ok=True)
    
    # Find all .avi files in the input folder
    avi_files = sorted(Path(input_folder).glob("*.avi"))
    
    if not avi_files:
        print(f"No .avi files found in {input_folder}")
        return {}
    
    results = {}
    
    for video_file in avi_files:
        video_name = video_file.stem
        output_folder = os.path.join(output_base_folder, video_name)
        
        try:
            frame_count = extract_frames_from_avi(str(video_file), output_folder)
            results[video_name] = frame_count
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            results[video_name] = None
    
    print(f"\nProcessing complete! Summary:")
    for video_name, count in results.items():
        if count is not None:
            print(f"  {video_name}: {count} frames")
        else:
            print(f"  {video_name}: Failed")
    
    return results

def consolidate_images_from_subfolders(input_folder, output_folder, move=False):
    """
    Consolidate all images from a folder with subfolders into a single folder.
    
    This function recursively searches through all subfolders and collects all image files
    (jpg, jpeg, png, bmp, tiff, gif) into a single output folder.
    
    Args:
        input_folder (str): Base folder containing subfolders with images
        output_folder (str): Destination folder where all images will be collected
        move (bool): If True, move files (cut). If False, copy files. Default is False.
    
    Returns:
        dict: Summary with 'total_files' count and 'files_by_subfolder' breakdown
    """
    # Define supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.GIF'}
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    total_files = 0
    files_by_subfolder = {}
    
    # Recursively search for all image files
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            # Get the relative subfolder path for tracking
            relative_path = file_path.relative_to(input_path)
            subfolder = str(relative_path.parent)
            
            # Track files by subfolder
            if subfolder not in files_by_subfolder:
                files_by_subfolder[subfolder] = 0
            files_by_subfolder[subfolder] += 1
            
            # Create destination filename (preserve original name)
            output_file = os.path.join(output_folder, file_path.name)
            
            # Handle duplicate filenames by appending underscore + counter
            counter = 1
            base_name = file_path.stem
            extension = file_path.suffix
            while os.path.exists(output_file):
                output_file = os.path.join(output_folder, f"{base_name}_{counter}{extension}")
                counter += 1
            
            # Copy or move the file
            if move:
                shutil.move(str(file_path), output_file)
            else:
                shutil.copy2(str(file_path), output_file)
            
            total_files += 1
    
    # Print summary
    print(f"{'Moved' if move else 'Copied'} {total_files} image files to {output_folder}")
    if files_by_subfolder:
        print("\nFiles by subfolder:")
        for subfolder, count in sorted(files_by_subfolder.items()):
            print(f"  {subfolder}: {count} files")
    
    return {
        'total_files': total_files,
        'files_by_subfolder': files_by_subfolder,
        'move': move
    }