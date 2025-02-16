from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, BertTokenizer, BertModel

from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, DiffusionPipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import sys

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel.from_pretrained("bert-base-uncased")



def preprocess(example):
    """
    Preprocess dataset: Resize and normalize images, tokenize captions
    """
    image = example["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    
    image = image.convert("RGBA")
    
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    example["pixel_values"] = transform(image).unsqueeze(0)


     # Tokenize text (captions)
    text = example["text"]  # Assuming you have a "text" field in the dataset
    tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")


    # Get encoder hidden states (embeddings) from the text model
    encoder_hidden_states = text_model(**tokenized_text).last_hidden_state.squeeze(0)
    example["encoder_hidden_states"] = encoder_hidden_states  # Store the embeddings

    return example



def main():
    # Load the dataset and preproccess it
    print("Load dataset")
    dataset = load_dataset("emily49/hateful_memes_test")
    print("Dataset loaded")
    dataset = dataset.map(preprocess, remove_columns=["image"])


    # Load a model, define lora_config, apply lora_config to the model
    # Def training args, and train the model
    print("Dataset mapped, loading model")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    #model_id = "runwayml/stable-diffusion-v1-5"
    #model_id = "CompVis/stable-diffusion-v1-4"
    #model_id = "stabilityai/stable-diffusion-2"
    model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    print("Model Loaded")
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules = ["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.1
    )
    print("LORA config defined")
    model = get_peft_model(model, lora_config)
    print("Model loaded with LORA config")

    traing_args = TrainingArguments(
        output_dir="./lora_model",
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="./logs"
    )
    print("Training args created, lets load the training set")

    # Create DataLoader
    train_dataloader = DataLoader(dataset["train"], batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    print("Training dataset loaded")
    # Move model to Apple Metal GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"The device is: {device}")
    model.to(device)

    # Training loop
    print("Defining optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=traing_args.learning_rate)
    model.train()
    print("Starting epochs")
    # Define total number of diffusion steps
    total_steps = 1000
    for epoch in range(traing_args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert list to tensor, cast to float32, and move to device
            pixel_values = torch.tensor(batch["pixel_values"]).float().to(device)
            optimizer.zero_grad()

            current_timestep = total_steps - step  # Example: start from `total_steps` and go down to 0
            
            #timestep = torch.tensor([current_timestep], device=device)  # Adjust according to your logic
            #timesteps = torch.tensor([current_timestep] * pixel_values.shape[0], device=device)
            timestep = torch.tensor([current_timestep] * pixel_values.shape[0], device=device)

        
            # Prepare encoder_hidden_states (e.g., from text encoder, assuming batch has this)
            
            encoder_hidden_states = torch.tensor(batch["encoder_hidden_states"]).float().to(device)
             # Forward pass with the required arguments
            outputs = model(pixel_values, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        
            #outputs = model(pixel_values)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

    # Save the fine tuned model
    print("Save the model")
    model.save_pretrained(traing_args.output_dir)
    print("Fine tuning completed!")

if __name__ == '__main__':
    main()
    print("Hello World!")