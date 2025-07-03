from transformers import AutoTokenizer, AutoModel
import torch

def main():
    # Load the PubMedBERT model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    # Example
    text = "Patient diagnosed with metabolic syndrome, fatty liver, and elevated ALT levels."

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Model output without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print("PubMedBERT embedding shape:", cls_embedding.shape)

if __name__ == "__main__":
    main()
