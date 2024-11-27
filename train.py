from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd


def load_qa_data(csv_file):
    """Load question-answer pairs from a CSV file."""
    df = pd.read_csv(csv_file)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise KeyError("CSV file must contain 'question' and 'answer' columns.")
    return [{"text": f"Question: {q} Answer: {a} <|endoftext|>"} for q, a in zip(df['question'], df['answer'])]


def fine_tune_model(model_name, train_data):
    """Fine-tune GPT-2 on the QA dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add padding token if missing
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Adjust embeddings for new tokens

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding="max_length",  # Ensure uniform sequence lengths
            return_tensors="np"  # Return NumPy arrays
        )

    # Tokenize the dataset
    tokenized_dataset = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Ensure both input_ids and labels are present
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"labels": x["input_ids"]},
        batched=True,
    )

    # Set dataset format for PyTorch
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuning complete. Model saved to './fine_tuned_model'.")



def main():
    csv_file = "qa_dataset.csv"  # Replace with your CSV file
    model_name = "gpt2"          # Base model to fine-tune

    print("Loading data...")
    qa_pairs = load_qa_data(csv_file)

    print("Converting to Dataset format...")
    train_data = Dataset.from_list(qa_pairs)

    print("Starting fine-tuning...")
    fine_tune_model(model_name, train_data)


if __name__ == "__main__":
    main()
