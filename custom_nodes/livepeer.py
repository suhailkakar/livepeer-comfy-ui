
import requests
from PIL import Image
import torch
from io import BytesIO
import numpy as np 

def fetch_and_prepare_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    img = img.convert('RGB')
    
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    if img_tensor.shape[-1] != 3:
        img_tensor = img_tensor.permute(0, 2, 3, 1)
    
    return (img_tensor,) 

class PipelineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (["text-to-image", "image-to-image"],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_pipeline"
    CATEGORY = "Livepeer Studio"

    def select_pipeline(self, pipeline):
        return (pipeline,)

class TextToImageModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "SG161222/RealVisXL_V4.0_Lightning",
                    "ByteDance/SDXL-Lightning",
                ],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_model"
    CATEGORY = "Livepeer Studio/Text to image"

    def select_model(self, model):
        return (model,)

class ImageToImageModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "timbrooks/instruct-pix2pix",
                ],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_model"
    CATEGORY = "Livepeer Studio/Image to image"

    def select_model(self, model):
        return (model,)

class PromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_prompt"
    CATEGORY = "Livepeer Studio"

    def process_prompt(self, prompt):
        return (prompt,)

class TextToImageSettingsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 20}),
                "negative_prompt": ("STRING", {"default": ""}),
                "safety_check": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("TEXT_TO_IMAGE_SETTINGS",)
    FUNCTION = "process_settings"
    CATEGORY = "Livepeer Studio/Text to image"

    def process_settings(self, width, height, guidance_scale, negative_prompt, safety_check, seed, num_inference_steps, num_images_per_prompt):
        settings = {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "safety_check": safety_check,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt
        }
        return (settings,)

class ImageToImageSettingsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 20}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0, "max": 10}),
                "negative_prompt": ("STRING", {"default": ""}),
                "safety_check": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),
                "num_inference_steps": ("INT", {"default": 100, "min": 1, "max": 200}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE_TO_IMAGE_SETTINGS",)
    FUNCTION = "process_settings"
    CATEGORY = "Livepeer Studio/Image to image"

    def process_settings(self, strength, guidance_scale, image_guidance_scale, negative_prompt, safety_check, seed, num_inference_steps, num_images_per_prompt):
        settings = {
            "strength": strength,
            "guidance_scale": guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "negative_prompt": negative_prompt,
            "safety_check": safety_check,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt
        }
        return (settings,)
class ImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "Livepeer Studio/Image to image"

    def process_image(self, image):
        return (image,)


class TextToImageRunNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("STRING", {"forceInput": True}),
                "model": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "settings": ("TEXT_TO_IMAGE_SETTINGS", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_text_to_image"
    CATEGORY = "Livepeer Studio/Text to image"

    LIVEPEER_API_KEY = ""

    def run_text_to_image(self, pipeline, model, prompt,  settings=None):
        url = "https://livepeer.studio/api/beta/generate/text-to-image"
        headers = {
            "Authorization": f"Bearer {self.LIVEPEER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
          "prompt": prompt,
          "negative_prompt": "",
          "width": 512,
          "height": 512,
          "num_images_per_prompt": 4,
          "num_inference_steps": 20
        }
       

        if settings:
            payload.update(settings)

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()

            if not data or 'images' not in data or not data['images']:
                raise ValueError("No images received in the response")

            first_image_url = data['images'][0].get('url')
            if not first_image_url:
                raise ValueError("No URL found for the first image")
            
            return  fetch_and_prepare_image(first_image_url)


        except (requests.RequestException, ValueError) as e:
            print(f"Error processing Livepeer API response: {e}")
            return (torch.zeros((1, 3, 512, 512)),)




class ImageToImageRunNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("STRING", {"forceInput": True}),
                "model": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE", {"forceInput": True}),
            },
            "optional": {
                "settings": ("IMAGE_TO_IMAGE_SETTINGS", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_image_to_image"
    CATEGORY = "Livepeer Studio/Image to image"

    def run_image_to_image(self, pipeline, model, prompt, image, settings=None):
        print(f"Running Image-to-Image with pipeline: {pipeline}, model: {model}")
        print(f"Prompt: {prompt}")
        if settings:
            print(f"Settings: {settings}")
        else:
            print("Using default settings")

        return (image,)
    

    
# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "PipelineNode": PipelineNode,
    "TextToImageModelNode": TextToImageModelNode,
    "ImageToImageModelNode": ImageToImageModelNode,
    "PromptNode": PromptNode,
    "TextToImageSettingsNode": TextToImageSettingsNode,
    "ImageToImageSettingsNode": ImageToImageSettingsNode,
    "ImageNode": ImageNode,
    "TextToImageRunNode": TextToImageRunNode,
    "ImageToImageRunNode": ImageToImageRunNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipelineNode": "Pipeline",
    "TextToImageModelNode": "Text to Image Model",
    "ImageToImageModelNode": "Image to Image Model",
    "PromptNode": "Prompt",
    "TextToImageSettingsNode": "Text to Image Settings",
    "ImageToImageSettingsNode": "Image to Image Settings",
    "ImageNode": "Image",
    "TextToImageRunNode": "Text to Image Run",
    "ImageToImageRunNode": "Image to Image Run",
}