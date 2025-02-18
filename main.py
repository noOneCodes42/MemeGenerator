# prompt: Make code that can be used to fine tune with this dataset emily49/hateful_memes_test and this model sd-legacy/stable-diffusion-v1-5, with the macs gpu.
# The structure of the dataset is image: image, id:int64, img: string(Not a file path), text:String. 
# With this make the training loop and everything

# Install necessary libraries


# Import libraries
# import torch
# from diffusers import StableDiffusionPipeline
# from datasets import load_dataset
# from PIL import Image
# import os
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import DataLoader


# # Check for MPS availability and set device and dtype accordingly
# if torch.backends.mps.is_available():
#     device = "mps"
#     torch_dtype = torch.float16  # Use float16 for MPS
# else:
#     device = "cpu"
#     torch_dtype = torch.float32  # Use float32 for CPU
# print(f"Using device: {device}")

# # Load the dataset
# dataset = load_dataset("emily49/hateful_memes_test", split="train")

# # Load the pre-trained model
# model_id = "sd-legacy/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)

# # Remove safety checker (use with caution)
# if "safety_checker" in pipe.components:
#     del pipe.components["safety_checker"]

# # Define hyperparameters (adjust as needed)
# learning_rate = 5e-6   # Example learning rate
# num_epochs = 3  # Example number of epochs
# batch_size = 16  # Example batch size (adjust based on your GPU memory)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # Get all parameters of the pipeline's modules
# params = []
# for name, module in pipe.components.items():
#     if hasattr(module, "parameters"):
#         params.extend(list(module.parameters()))

# # Create the optimizer using the collected parameters
# optimizer = torch.optim.AdamW(params, lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     for i, example in enumerate(dataset):
#         # Preprocess text prompt (replace with your actual preprocessing if needed)
#         prompt = example['text']

#         # Generate an image with the Stable Diffusion model
#         image = pipe(prompt, num_inference_steps=50).images[0]

#         # Convert PIL Image to NumPy array and then to PyTorch tensor
#         generated_image_np = np.array(image)
#         generated_image_tensor = torch.tensor(generated_image_np, dtype=torch_dtype, device=device, requires_grad=True) / 255.0 # requires_grad=True added here

#         # Convert target image (PngImageFile) to NumPy array and then to tensor
#         target_image_np = np.array(example['image'])

#         # Ensure target image has 3 channels (RGB)
#         if target_image_np.ndim == 2:  # If grayscale, convert to RGB
#             target_image_np = np.stack([target_image_np] * 3, axis=-1)
#         elif target_image_np.shape[2] != 3:  # If not RGB, convert to RGB
#             target_image_np = target_image_np[:, :, :3]  # Take only the first 3 channels

#         target_image_tensor = torch.tensor(target_image_np, dtype=torch_dtype, device=device) / 255.0

#         # Reshape to (C, H, W) before resizing
#         target_image_tensor = target_image_tensor.permute(2, 0, 1).unsqueeze(0).float()

#         # Resize target image to match generated image dimensions
#         target_image_tensor = F.interpolate(target_image_tensor,
#                                            size=(generated_image_tensor.shape[0], generated_image_tensor.shape[1]),
#                                            mode='bilinear', align_corners=False).squeeze(0)
        
#         # Permute back to (H, W, C) for loss calculation
#         target_image_tensor = target_image_tensor.permute(1, 2, 0)

#         # Calculate the loss (e.g., mean squared error)
#         loss = F.mse_loss(generated_image_tensor.float(), target_image_tensor.float())

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if (i + 1) % 10 == 0:
#             print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}")

# # Save the fine-tuned model (replace with your desired saving location)
# pipe.save_pretrained("fine_tuned_sd_model")

# # Example inference after fine-tuning
# image = pipe(	
# "to see better, asians sometime switch to fullscreen veiw", num_inference_steps=100).images[0]

# # Save or display the generated image
# image.save("generated_image1.png")

# print("Finished")
# Optimized With Batch and Memory side by side
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_dataset
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import ceil

# Load dataset
dataset = load_dataset("emily49/hateful_memes_test", split="train")

# Define transformation
transform = T.Compose([T.ToTensor(), T.Resize((512, 512))])

# Dataset wrapper for DataLoader
class MemeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example['text']
        target_image = Image.fromarray(np.array(example['image'])).convert('RGB')
        if self.transform:
            target_image = self.transform(target_image)
        return prompt, target_image

# DataLoader with batching
num_epochs = 1
batch_size = 8  # Adjust based on GPU memory
dataloader = DataLoader(MemeDataset(dataset, transform), batch_size=batch_size, shuffle=True)

device = "mps"
torch_dtype = torch.float32

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype).to(device)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-6)  # Optimize only U-Net weights

# Training loop with batching
for epoch in range(num_epochs):
    for i, (prompts, target_images) in enumerate(dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            images = []
            for prompt in prompts:  # Generate one by one
                image = pipe(prompt, num_inference_steps=50).images[0]
                image = transform(image).to(device, dtype=torch_dtype)
                images.append(image)
        
        generated_images_tensor = torch.stack(images).requires_grad_()
        target_images = target_images.to(device, dtype=torch_dtype)

        loss = F.mse_loss(generated_images_tensor, target_images)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}")

pipe.save_pretrained("fine_tuned_sd_model")

# Example inference after fine-tuning
image = pipe(   
"to see better, asians sometime switch to fullscreen veiw", num_inference_steps=100).images[0]

# Save or display the generated image
image.save("generated_image1.png")

print("Finished")