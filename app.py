from utils import generate_viton_image, detect_person, image_save_to_result

api_key = "ffd59fd390msh937ac7182a38043p1603ebjsn5136f77de6f3"

# User uploads their image
image_path = "path/to/person/image"
cloth_image_path = "path/to/cloth/image"

#detect a number of person to make sure only one person in the image
num = detect_person(image_path)
success = ""
if num == 1:
    # Generate the virtual try-on image
    success = generate_viton_image(
        person_image_path=image_path,
        cloth_image_path=cloth_image_path,
        api_key=api_key,
    )
else:
    print("the image is not valid, only one person in the image.")

if not success:
    print("VITON image generation failed.")


    #save the generated image in result directory
image_save_to_result("output.jpg")


