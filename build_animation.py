import cv2
import os

def images_to_video(image_folder, output_video_name, fps=10):  # Set fps to 10
    # Get all image files in the folder
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    output_video_path = os.path.join(image_folder, output_video_name)
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")

# Example usage
image_folder = "/root/xzcllwx_ws/GameFormer-Planner/figure"  # Replace with your folder path
output_video_name = "output_video.mp4"  # Replace with your desired output file name
fps = 10  # Frames per second (0.1s per frame)
images_to_video(image_folder, output_video_name, fps)