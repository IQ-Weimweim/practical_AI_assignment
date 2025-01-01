import cv2
from try_on_diffusion_client import TryOnDiffusionClient
from ultralytics import YOLO
import os
import shutil

yolo_model = YOLO("../yolov8n.pt")

def detect_person(image_path: str) -> int:
    """Detect the number of people in an image using YOLO."""
    results = yolo_model(image_path)
    # Filter detections for 'person' class (class index 0 in COCO)
    person_detections = [det for det in results[0].boxes.data if int(det[5]) == 0]
    return len(person_detections)


def generate_viton_image(person_image_path, cloth_image_path, api_key, base_url="https://try-on-diffusion.p.rapidapi.com/"):

  # Create the TryOnDiffusionClient object
  client = TryOnDiffusionClient(base_url=base_url, api_key=api_key)

  # Load images as NumPy arrays
  try:
    clothing_image = cv2.imread(cloth_image_path)
    avatar_image = cv2.imread(person_image_path)
  except Exception as e:
    print(f"Error loading images: {e}")
    return False

  # Call the API
  response = client.try_on_file(
      clothing_image=clothing_image,
      avatar_image=avatar_image,
      seed=-1,
  )

  # Handle the response
  if response.status_code == 200 and response.image is not None:
    # Save the resulting image
    try:
      cv2.imwrite("output.jpg", response.image)
      print("Image saved as 'output.jpg'")
      return True
    except Exception as e:
      print(f"Error saving image: {e}")
      return False
  else:
    # Handle errors
    print(f"API request failed. Status Code: {response.status_code}")
    if response.error_details:
      print(f"Error details: {response.error_details}")
    return False



def image_save_to_result(source_file, result_dir="result"):
    # Ensure the `result` directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Define the destination path
    destination_path = os.path.join(result_dir, os.path.basename(source_file))

    # Move the file
    shutil.move(source_file, destination_path)
    print(f"File save to: {destination_path}")
