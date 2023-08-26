import os
from modules.processing import StableDiffusionProcessingImg2Img
from scripts.faceswap import FaceSwapScript, get_models
from utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil
from logging_patch import apply_logging_patch


def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


class roop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                # Comma separated face number(s)
                "faces_index": ("STRING", {"default": "0"}),
                "reference_faces_index": ("STRING", {"default": "0"}),
                # Allow user to change the logging amount, going from minimal to verbose
                "console_logging_level": ([0, 1, 2],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    def execute(self, image, reference_image, swap_model, faces_index, reference_faces_index, console_logging_level):
        apply_logging_patch(console_logging_level)

        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(image)
        source = tensor_to_pil(reference_image)
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(
            p=p, img=source, enable=True, faces_index=faces_index,
            reference_faces_index=reference_faces_index, model=swap_model,
            face_restorer_name=None, face_restorer_visibility=None,
            upscaler_name=None, upscaler_scale=None, upscaler_visibility=None,
            swap_in_source=True, swap_in_generated=True
        )
        result = batched_pil_to_tensor(p.init_images)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "roop": roop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "roop": "roop",
}
