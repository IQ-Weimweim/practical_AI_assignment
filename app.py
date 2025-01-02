import logging
import gradio as gr
from utils import get_examples, try_on

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(thread)-8s %(name)-16s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


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

    generate_button.click(
        fn=try_on,
        inputs=[
            clothing_image,
            avatar_image,
        ],
        outputs=[result_image],
        api_name=False,
    )

    app.title = "Cloth Virtual Try-On"

if __name__ == "__main__":
    app.queue(api_open=False).launch(show_api=False, ssr_mode=False)



