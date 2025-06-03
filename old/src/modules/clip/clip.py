from transformers import CLIPTokenizer, CLIPModel
import torch

# Load pre-trained CLIP model and tokenizer
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def get_text_embedding(texts):
    # Tokenize
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)  # shape: (batch_size, embedding_dim)

    return outputs  # already normalized

texts = ["a photo of a cat", "a picture of a dog"]
embeddings = get_text_embedding(texts)  # torch.Tensor of shape (2, 512)
print(embeddings.shape)
print(embeddings)  # Print the embeddings