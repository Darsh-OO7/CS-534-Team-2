from transformers import AutoTokenizer, AutoModel
import torch

def main():
    # Model name
    model_name = "emilyalsentzer/Bio_Discharge_Summary_BERT"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load '{model_name}'. Falling back to Bio_ClinicalBERT.")
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    model.eval()

    # Sample clinical note
    text = "Patient presents with elevated liver enzymes, insulin resistance, and hepatomegaly."

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model output without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  
    print("Bio+Discharge BERT embedding shape:", cls_embedding.shape)

if __name__ == "__main__":
    main()
