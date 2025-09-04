from transformers import pipeline
import logging

# Suppress all red-colored warning messages from the library.
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize the text generation pipeline using a very stable model.
generator = pipeline('text-generation', model='distilgpt2')

# Define a specific prompt that guides the AI to generate a detailed paragraph.
prompt = "Explain how rainbows are formed?"
# The parameters below prevent repetitive answers and ensure a good length.
response = generator(
    prompt,
    max_length=250,
    num_return_sequences=1,
    no_repeat_ngram_size=3,
    pad_token_id=50256
)

# Extract and print the generated text cleanly.
print("--- AI Generated Answer ---")
print(response[0]['generated_text'])
print("-----------------------------")