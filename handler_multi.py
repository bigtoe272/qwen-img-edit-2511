"""Compatibility overlay adding simplified ``images[]`` support to handler.py.

This module keeps the original worker implementation intact while extending its
simplified edit API to accept one to three image inputs:

    {"input": {"prompt": "...", "images": [image1, image2, image3], ...}}

Legacy ``image`` + ``reference_image`` payloads and raw workflow mode remain
supported.
"""

import copy
import json

import handler as base


# Extend the existing workflow with a third LoadImage slot. The built-in
# TextEncodeQwenImageEditPlus node accepts optional image1/image2/image3 inputs.
DEFAULT_WORKFLOW = copy.deepcopy(base.DEFAULT_WORKFLOW)
DEFAULT_WORKFLOW["83"]["_meta"]["title"] = "Load Reference Image 1"
DEFAULT_WORKFLOW["84"] = {
    "inputs": {"image": "source_image.png"},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Reference Image 2"},
}


def build_edit_workflow(
    prompt,
    source_image="source_image.png",
    reference_images=None,
    seed=None,
    steps=4,
    negative_prompt="",
    lora=None,
    cfg=None,
    shift=3.1,
    sampler="euler",
    scheduler="simple",
):
    """Build a Qwen edit workflow for one to three input images."""
    lora_key, use_lora, auto_cfg = base._resolve_lora_mode(steps, lora)
    effective_cfg = cfg if cfg is not None else auto_cfg

    reference_images = list(reference_images or [])
    if len(reference_images) > 2:
        raise ValueError("A maximum of 3 input images is supported")

    workflow = copy.deepcopy(DEFAULT_WORKFLOW)
    workflow["41"]["inputs"]["image"] = source_image

    # image1 is the source image. image2/image3 are optional and are only
    # connected when supplied, so unused slots are not duplicated.
    for node_id in ("170:149", "170:151"):
        node_inputs = workflow[node_id]["inputs"]
        node_inputs["image1"] = ["170:160", 0]
        node_inputs.pop("image2", None)
        node_inputs.pop("image3", None)

    if len(reference_images) >= 1:
        workflow["83"]["inputs"]["image"] = reference_images[0]
        workflow["170:149"]["inputs"]["image2"] = ["83", 0]
        workflow["170:151"]["inputs"]["image2"] = ["83", 0]

    if len(reference_images) >= 2:
        workflow["84"]["inputs"]["image"] = reference_images[1]
        workflow["170:149"]["inputs"]["image3"] = ["84", 0]
        workflow["170:151"]["inputs"]["image3"] = ["84", 0]

    workflow["170:151"]["inputs"]["prompt"] = prompt
    workflow["170:149"]["inputs"]["prompt"] = negative_prompt
    workflow["170:169"]["inputs"]["seed"] = (
        seed if seed is not None else base.random.randint(0, 2**53)
    )

    workflow["170:168"]["inputs"]["value"] = use_lora
    if use_lora and lora_key:
        workflow["170:153"]["inputs"]["lora_name"] = base.LORA_FILES[lora_key]

    workflow["170:169"]["inputs"]["steps"] = steps
    workflow["170:169"]["inputs"]["cfg"] = effective_cfg
    workflow["170:169"]["inputs"]["sampler_name"] = sampler
    workflow["170:169"]["inputs"]["scheduler"] = scheduler
    workflow["170:145"]["inputs"]["shift"] = shift

    return workflow


def _resolve_simplified_images(job_input):
    """Return (image_values, image_names, using_list_api, error_message)."""
    list_images = job_input.get("images")
    if list_images is not None:
        if not isinstance(list_images, list):
            return None, None, True, (
                "In simplified edit mode, 'images' must be a list of 1 to 3 image strings"
            )
        if not 1 <= len(list_images) <= 3:
            return None, None, True, (
                "In simplified edit mode, 'images' must contain between 1 and 3 images"
            )
        if not all(isinstance(image, str) and image for image in list_images):
            return None, None, True, (
                "Each item in simplified 'images' must be a non-empty "
                "base64/data-URI string or filename"
            )

        image_names = job_input.get("image_names")
        if image_names is not None:
            if not isinstance(image_names, list) or len(image_names) != len(list_images):
                return None, None, True, (
                    "'image_names' must be a list with the same length as 'images'"
                )
            if not all(isinstance(name, str) and name.strip() for name in image_names):
                return None, None, True, (
                    "Each item in 'image_names' must be a non-empty string"
                )
        else:
            image_names = [
                "source_image.png",
                "reference_image_1.png",
                "reference_image_2.png",
            ][: len(list_images)]

        return list_images, image_names, True, None

    # Legacy simplified API.
    source_image = job_input.get("image")
    if not source_image:
        return None, None, False, (
            "Simplified edit mode requires either 'images' (1 to 3 images) "
            "or the legacy 'image' field"
        )
    if not isinstance(source_image, str):
        return None, None, False, (
            "'image' must be a base64/data-URI string or filename"
        )

    image_values = [source_image]
    image_names = [job_input.get("image_name", "source_image.png")]

    reference_image = job_input.get("reference_image")
    if reference_image:
        if not isinstance(reference_image, str):
            return None, None, False, (
                "'reference_image' must be a base64/data-URI string or filename"
            )
        image_values.append(reference_image)
        image_names.append(
            job_input.get("reference_image_name", "reference_image_1.png")
        )

    return image_values, image_names, False, None


def validate_input(job_input):
    """Validate raw workflow mode or simplified one-to-three-image edit mode."""
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    if not isinstance(job_input, dict):
        return None, "Input must be a JSON object"

    workflow = job_input.get("workflow")
    prompt = job_input.get("prompt")

    if workflow is not None:
        # Raw workflow mode keeps the original upload-object schema.
        images_to_upload = job_input.get("images")
        if images_to_upload is not None:
            if not isinstance(images_to_upload, list) or not all(
                isinstance(image, dict)
                and "name" in image
                and "image" in image
                for image in images_to_upload
            ):
                return None, (
                    "In raw workflow mode, 'images' must be a list of objects "
                    "with 'name' and 'image' keys"
                )

    elif prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            return None, "'prompt' must be a non-empty string"

        image_values, image_names, using_list_api, image_error = (
            _resolve_simplified_images(job_input)
        )
        if image_error:
            return None, image_error

        lora_val = job_input.get("lora")
        if lora_val is not None:
            lora_val = str(lora_val).lower().strip()
            if lora_val not in ("4step", "8step", "none"):
                return None, "'lora' must be '4step', '8step', or 'none'"

        steps_val = job_input.get("steps", 4)
        if not isinstance(steps_val, int) or steps_val < 1:
            return None, "'steps' must be a positive integer"

        cfg_val = job_input.get("cfg")
        if cfg_val is not None and (
            not isinstance(cfg_val, (int, float)) or cfg_val < 0
        ):
            return None, "'cfg' must be a non-negative number"

        shift_val = job_input.get("shift")
        if shift_val is not None and (
            not isinstance(shift_val, (int, float)) or shift_val <= 0
        ):
            return None, "'shift' must be a positive number"

        sampler_val = job_input.get("sampler", "euler")
        if not isinstance(sampler_val, str) or not sampler_val.strip():
            return None, "'sampler' must be a non-empty string"

        scheduler_val = job_input.get("scheduler", "simple")
        if not isinstance(scheduler_val, str) or not scheduler_val.strip():
            return None, "'scheduler' must be a non-empty string"

        # Upload base64/data-URI entries and retain filenames as direct refs.
        images_to_upload = []
        resolved_refs = []
        for image_value, image_name in zip(image_values, image_names):
            if image_value.startswith("data:") or len(image_value) > 200:
                images_to_upload.append({"name": image_name, "image": image_value})
                resolved_refs.append(image_name)
            else:
                resolved_refs.append(image_value)

        source_ref = resolved_refs[0]
        reference_refs = resolved_refs[1:]

        # Preserve legacy single-image behaviour exactly: the old handler
        # reused the source as image2 if reference_image was omitted.
        if not using_list_api and len(resolved_refs) == 1:
            reference_refs = [source_ref]

        workflow = build_edit_workflow(
            prompt=prompt,
            source_image=source_ref,
            reference_images=reference_refs,
            seed=job_input.get("seed"),
            steps=steps_val,
            negative_prompt=job_input.get("negative_prompt", ""),
            lora=lora_val,
            cfg=cfg_val,
            shift=shift_val if shift_val is not None else 3.1,
            sampler=sampler_val,
            scheduler=scheduler_val,
        )

        if not images_to_upload:
            images_to_upload = None

    else:
        return None, "Missing 'workflow' or 'prompt' parameter"

    return {
        "workflow": workflow,
        "images": images_to_upload,
        "comfy_org_api_key": job_input.get("comfy_org_api_key"),
    }, None


# Patch the original module globals used by base.handler().
base.DEFAULT_WORKFLOW = DEFAULT_WORKFLOW
base.build_edit_workflow = build_edit_workflow
base.validate_input = validate_input


if __name__ == "__main__":
    print("worker-comfyui - Starting handler with simplified images[] support...")
    base.runpod.serverless.start({"handler": base.handler})
