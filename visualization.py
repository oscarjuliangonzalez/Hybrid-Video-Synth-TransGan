import cv2
import os
from pathlib import Path
from typing import List, Union


def images_to_video(
    image_paths: List[str],
    output_video_path: str,
    fps: int = 30,
    frame_size: tuple = None,
    codec: str = 'mp4v',
    verbose: bool = True
) -> bool:
    """
    Convert a sequence of images to a video file.
    
    This function takes a list of image paths and creates a video by stacking them
    together at a specified frame rate. Images are processed in the order provided.
    
    Args:
        image_paths (List[str]): List of paths to image files to include in the video.
                                 Images should be in the order they should appear in the video.
        output_video_path (str): Path where the output video will be saved (e.g., 'output.mp4').
        fps (int): Frames per second for the output video. Default is 30.
                  Higher values make the video play faster, lower values make it slower.
        frame_size (tuple): Optional (width, height) tuple to resize all images to.
                           If None, uses the size of the first image.
        codec (str): Video codec to use. Default is 'mp4v' for MP4 format.
                    Other options: 'MJPG' for Motion JPEG, 'XVID', 'H264', etc.
        verbose (bool): If True, print progress information. Default is True.
    
    Returns:
        bool: True if video creation was successful, False otherwise.
    
    Raises:
        ValueError: If image_paths is empty or output_video_path is invalid.
        FileNotFoundError: If any of the specified image files don't exist.
    
    Examples:
        >>> images = ['frame_001.png', 'frame_002.png', 'frame_003.png']
        >>> images_to_video(images, 'output.mp4', fps=24)
        
        >>> # Create a video with custom size
        >>> images_to_video(images, 'output.mp4', fps=60, frame_size=(1280, 720))
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    
    if not output_video_path:
        raise ValueError("output_video_path cannot be empty")
    
    # Validate that all image files exist
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the first image to determine frame size if not provided
        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            raise ValueError(f"Could not read image: {image_paths[0]}")
        
        if frame_size is None:
            frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
        
        # Create video writer object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_video_path}")
        
        if verbose:
            print(f"Creating video: {output_video_path}")
            print(f"  FPS: {fps}")
            print(f"  Frame size: {frame_size[0]}x{frame_size[1]}")
            print(f"  Number of frames: {len(image_paths)}")
        
        # Write each frame to the video
        for idx, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Warning: Could not read image {idx+1}/{len(image_paths)}: {img_path}")
                continue
            
            # Resize frame if necessary
            if frame.shape[:2] != (frame_size[1], frame_size[0]):
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
            
            writer.write(frame)
            
            if verbose and (idx + 1) % max(1, len(image_paths) // 10) == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} frames")
        
        writer.release()
        
        if verbose:
            video_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
            duration_sec = len(image_paths) / fps
            print(f"Video created successfully!")
            print(f"  Output: {output_video_path}")
            print(f"  Duration: {duration_sec:.2f} seconds")
            print(f"  File size: {video_size_mb:.2f} MB")
        
        return True
    
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        # Clean up incomplete video file
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except:
                pass
        return False


def images_from_folder_to_video(
    image_folder: str,
    output_video_path: str,
    fps: int = 30,
    pattern: str = '*',
    sort_by: str = 'name',
    frame_size: tuple = None,
    codec: str = 'mp4v',
    verbose: bool = True
) -> bool:
    """
    Convert all images in a folder to a video.
    
    This is a convenience wrapper that automatically discovers image files in a folder
    and converts them to a video. Images can be sorted by name or modification time.
    
    Args:
        image_folder (str): Path to folder containing image files.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second. Default is 30.
        pattern (str): Glob pattern to filter images (e.g., '*.png', 'frame_*.jpg').
                      Default is '*' (all files).
        sort_by (str): How to sort images. Options: 'name' (default), 'mtime' (modification time).
        frame_size (tuple): Optional (width, height) to resize frames.
        codec (str): Video codec. Default is 'mp4v'.
        verbose (bool): Print progress information. Default is True.
    
    Returns:
        bool: True if successful, False otherwise.
    
    Examples:
        >>> images_from_folder_to_video('frames/', 'output.mp4', fps=24)
        >>> images_from_folder_to_video('frames/', 'output.mp4', pattern='*.png', fps=60)
    """
    if not os.path.exists(image_folder):
        raise ValueError(f"Image folder not found: {image_folder}")
    
    # Find all image files matching the pattern
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    image_paths = []
    
    for file_path in Path(image_folder).glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    
    if not image_paths:
        raise ValueError(f"No image files found in {image_folder} matching pattern '{pattern}'")
    
    # Sort images
    if sort_by == 'mtime':
        image_paths.sort(key=lambda x: os.path.getmtime(x))
    else:  # 'name' or default
        image_paths.sort()
    
    if verbose:
        print(f"Found {len(image_paths)} images in {image_folder}")
    
    return images_to_video(
        image_paths=image_paths,
        output_video_path=output_video_path,
        fps=fps,
        frame_size=frame_size,
        codec=codec,
        verbose=verbose
    )
