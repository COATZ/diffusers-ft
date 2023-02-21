from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch
import sys

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
print(pipeline)
# for elt in pipeline:
#     print(elt)
# sys.exit()
pipeline.to("cuda")

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
print(upscaler)
upscaler.to("cuda")

# prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
prompt = "a spherical image of a forest, high resolution, unreal engine, ultra realistic  "
generator = torch.manual_seed(33)

# we stay in latent space! Let's make sure that Stable Diffusion returns the image
# in latent space
low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]

# Let's save the upscaled image under "upscaled_astronaut.png"
# upscaled_image.save("astronaut_1024.png")
upscaled_image.save("test_1024.png")

# as a comparison: Let's also save the low-res image
with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

# image.save("astronaut_512.png")
image.save("test_512.png")