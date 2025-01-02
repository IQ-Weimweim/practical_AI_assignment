import logging
import gradio as gr
import numpy as np
import cv2
import os
import base64
from ultralytics import YOLO
from try_on_diffusion_client import TryOnDiffusionClient

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(thread)-8s %(name)-16s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "examples")

API_URL = "https://try-on-diffusion.p.rapidapi.com/"
API_KEY = "ffd59fd390msh937ac7182a38043p1603ebjsn5136f77de6f3"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

client = TryOnDiffusionClient(base_url=API_URL, api_key=API_KEY)

def save_numpy_as_png(image_numpy: np.ndarray, output_path: str):
    # Ensure the NumPy array is in uint8 format
    if image_numpy.dtype != np.uint8:
        image_numpy = (image_numpy * 255).astype(np.uint8)

    cv2.imwrite(output_path, image_numpy)
    return output_path

yolo_model = YOLO("../yolov8n.pt")
def detect_person(image_path: str) -> int:
    """Detect the number of people in an image using YOLO."""
    results = yolo_model(image_path)
    # Filter detections for 'person' class (class index 0 in COCO)
    person_detections = [det for det in results[0].boxes.data if int(det[5]) == 0]
    return len(person_detections)

def get_image_base64(file_name: str) -> str:
    _, ext = os.path.splitext(file_name.lower())

    content_type = "image/jpeg"

    if ext == ".png":
        content_type = "image/png"
    elif ext == ".webp":
        content_type = "image/webp"
    elif ext == ".gif":
        content_type = "image/gif"

    with open(file_name, "rb") as f:
        return f"data:{content_type};base64," + base64.b64encode(f.read()).decode("utf-8")


def get_examples(example_dir: str) -> list[str]:
    file_list = [f for f in os.listdir(os.path.join(EXAMPLE_PATH, example_dir)) if f.endswith(".jpg")]
    file_list.sort()

    return [os.path.join(EXAMPLE_PATH, example_dir, f) for f in file_list]


def try_on(
        clothing_image: np.ndarray = None,
        avatar_image: np.ndarray = None,
        seed: int = -1,
) -> tuple:
    if clothing_image is not None and avatar_image is not None:
        image = save_numpy_as_png(avatar_image, "image.png")
        num = detect_person(image)

        if num == 1:
            result = client.try_on_file(
                clothing_image=cv2.cvtColor(clothing_image, cv2.COLOR_RGB2BGR) if clothing_image is not None else None,
                avatar_image=cv2.cvtColor(avatar_image, cv2.COLOR_RGB2BGR) if avatar_image is not None else None,
                seed=seed,
            )

            if result.status_code == 200:
                return cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
            else:
                error_message = f"<h3>Error {result.status_code}</h3>"

                if result.error_details is not None:
                    error_message += f"<p>{result.error_details}</p>"

                return None, error_message

        else:
            error_massage = f"<h3>only one person in the image</h3>"
            return None, error_massage

    else:
        error_massage = f"<h3>please put the image needed</h3>"
        return None, error_massage


with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(3600, 3600)) as app:
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <h2>Clothing</h2>
                <p>
                    Clothing may be specified with a image given. 
                    <br/><br/>
                </p>
                """
            )

            with gr.Tab("Image"):
                clothing_image = gr.Image(label="Clothing Image", sources=["clipboard"], type="numpy")

                clothing_image_examples = gr.Examples(
                    inputs=clothing_image, examples_per_page=18, examples=get_examples("clothing")
                )

        with gr.Column():
            gr.HTML(
                """
                <h2>Person</h2>
                <p>
                    Upload your image.
                     <br/><br/>
                </p>
                """
            )

            with gr.Tab("Image"):
                avatar_image = gr.Image(label="Person Image", sources=["upload"], type="numpy")


    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <h2>Generation</h2>
                """
            )

            generate_button = gr.Button(value="Generate", variant="primary")

            result_image = gr.Image(label="Result", show_share_button=False, format="jpeg")
            result_details = gr.HTML(label="Details")

    generate_button.click(
        fn=try_on,
        inputs=[
            clothing_image,
            avatar_image,
        ],
        outputs=[result_image, result_details],
        api_name=False,
    )

    app.title = "Virtual Try-On Diffusion"

if __name__ == "__main__":
    app.queue(api_open=False).launch(show_api=False, ssr_mode=False)



