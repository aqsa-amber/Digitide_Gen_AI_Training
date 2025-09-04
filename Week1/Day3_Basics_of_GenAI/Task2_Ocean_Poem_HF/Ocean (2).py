# ocean_poem.py
from transformers import pipeline

def main():
    # Load a text-generation pipeline with a small model
    generator = pipeline("text-generation", model="gpt2")

    prompt = "Write a small poem about the ocean"

    # Generate text
    output = generator(
        prompt,
        max_length=50,   # total tokens including prompt
        num_return_sequences=1,
        temperature=0.7  # creativity
    )

    print("=== Generated Poem ===\n")
    print(output[0]["generated_text"])

if __name__ == "__main__":
    main()
