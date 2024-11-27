from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_answer(question, model_path="./fine_tuned_model", max_length=100):
    """Generate an answer for a given question using the fine-tuned model."""
    # Load the fine-tuned model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Encode the input prompt
    input_text = f"Question: {question} Answer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode("<|endoftext|>")[0],
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(input_text, "").strip()
    return response

def main():
    model_path = "./fine_tuned_model"  # Path to fine-tuned model
    print("Model loaded.")
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = generate_answer(question, model_path)
        print(f"AI: {answer}")

if __name__ == "__main__":
    main()
