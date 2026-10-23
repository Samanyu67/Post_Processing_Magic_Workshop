from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageEnhance, ImageFilter

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def generate_image(prompt):
    return pipe(prompt).image[0]

def post_process(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    image.image.filter(ImageFilter.GaussianBlur(2))
    return image

def main():
    print("Post Processing Magic Workshop")
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a description for an Image: ")
        if prompt.lower() == "exit":
            print("Goodbye")
            break
        try:
            print("Generating your image")
            img = generate_image(prompt)
            img1 = post_process(img)
            img1.show()
            save1 = input("Would you like to save your image (yes/no): ")
            if save1 == "yes":
                filename = input("Enter the name of the file: ")
                img1.save(f"Image Saved as {filename}.png")
                print("Image saved successfully")
        except Exception as e:
            print("An error occured", e)

if __name__ == "__main__":
    main()
    