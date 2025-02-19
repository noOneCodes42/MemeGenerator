from diffusers import StableDiffusionPipeline
import torch

# Load the fine-tuned model
fine_tuned_pipe = StableDiffusionPipeline.from_pretrained("fine_tuned_sd_model", torch_dtype=torch.float16).to("mps")

# Generate an image
prompt = "Generate a hilarious meme about planes that would make teenagers laugh. Use a funny image related to airplanes—maybe a weird flight situation, a confused pilot, or passengers reacting dramatically. The caption should be a witty joke or relatable struggle, like airport security fails, turbulence drama, overpriced snacks, or someone trying to open the emergency exit mid-flight. Keep it meme-worthy, using Gen Z humor, exaggeration, or internet slang to make it extra funny in U.S language knows as ENGLISH! !"
generated_image = fine_tuned_pipe(prompt, num_inference_steps=999).images[0]

# Save the generated image
generated_image.save("asian_meme5.png")

fine_tuned_pipe1 = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16).to("mps")

# Generate an image
prompt1 = "Generate a hilarious meme about planes that would make teenagers laugh. Use a funny image related to airplanes—maybe a weird flight situation, a confused pilot, or passengers reacting dramatically. The caption should be a witty joke or relatable struggle, like airport security fails, turbulence drama, overpriced snacks, or someone trying to open the emergency exit mid-flight. Keep it meme-worthy, using Gen Z humor, exaggeration, or internet slang to make it extra funny in U.S language knows as ENGLISH! !"
generated_image = fine_tuned_pipe1(prompt1, num_inference_steps=999).images[0]

# Save the generated image
generated_image.save("asian_meme6.png")