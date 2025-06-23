import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# --- IMPORTANT: REPLACE THIS PATH WITH THE ACTUAL PATH TO YOUR FINE-TUNED MODEL ---
# During your fine-tuning process, you would have saved the model.
# Example: If you saved it to a folder named 'finetuned_gpt2_wikitext',
# then the path would be './finetuned_gpt2_wikitext'.
FINETUNED_MODEL_PATH = '/home/dharun/Desktop/wikitext/gpt2-wikitext-best-model' 
# --- END IMPORTANT ---

# Load model and tokenizer
@st.cache_resource
def load_finetuned_model():
    try:
        model = GPT2LMHeadModel.from_pretrained(FINETUNED_MODEL_PATH)
        tokenizer = GPT2Tokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        # Ensure the tokenizer has a pad_token if it doesn't by default (common for GPT-2 in generation)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        st.success(f"Successfully loaded fine-tuned model from {FINETUNED_MODEL_PATH}")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned model from {FINETUNED_MODEL_PATH}. Make sure the path is correct and the model files exist.")
        st.error(f"Error details: {e}")
        # Fallback to generic GPT2 if fine-tuned model fails to load, for demo purposes
        st.info("Attempting to load generic 'gpt2' model instead for demonstration.")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer


model, tokenizer = load_finetuned_model()

st.title("Next Word Prediction (Fine-tuned GPT-2)")
st.markdown("This app predicts the next most likely words based on your input.")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "predicted_words" not in st.session_state:
    st.session_state.predicted_words = []

# Workaround: Use temp variable for input display
# Note: Using a key makes the text_area a controlled component,
# but st.experimental_rerun() might still cause minor display quirks.
# For more robust solutions, consider a custom component or a more complex state management.
text = st.text_area(
    "Enter your sentence:",
    value=st.session_state.input_text,
    key="user_input_area", # Changed key to avoid potential conflicts if 'user_input' was used elsewhere
    height=100
)

# Predict next words
if st.button("Predict Next Words"):
    if not text.strip(): # Check if input is empty
        st.warning("Please enter some text to predict.")
    else:
        # Update session state with current text so it persists across reruns
        st.session_state.input_text = text

        # Ensure input_ids don't exceed model's max_position_embeddings (typically 1024 for gpt2)
        # We'll truncate if necessary, and warn the user.
        max_model_input_length = model.config.max_position_embeddings - 1 # Leave space for 1 generated token
        encoded_input = tokenizer.encode(st.session_state.input_text, return_tensors="pt")

        if encoded_input.shape[1] > max_model_input_length:
            st.warning(f"Input text is too long ({encoded_input.shape[1]} tokens). Truncating to {max_model_input_length} tokens.")
            input_ids = encoded_input[:, -max_model_input_length:] # Use only the last N tokens
        else:
            input_ids = encoded_input

        # Move input_ids to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_ids = input_ids.to(device)


        # Generate multiple samples
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1, # Generate only the very next token
            num_return_sequences=5,
            do_sample=True, # Enable sampling for variety
            top_k=50, # Consider top 50 most likely words
            temperature=0.95, # Controls randomness: higher = more creative, lower = more deterministic
            pad_token_id=tokenizer.eos_token_id # Use EOS token for padding
        )

        # Extract unique next-word predictions
        predictions = set()
        current_input_len_tokens = input_ids.shape[1]

        for output in outputs:
            # Decode the generated sequence (input + 1 new token)
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            # Get only the newly generated token/word
            # This logic needs to be careful as decoding might add spaces or slight variations.
            # A more robust way is to just get the very last token's ID and decode it.
            if output.shape[0] > current_input_len_tokens: # Ensure a new token was actually generated
                next_token_id = output[current_input_len_tokens].item()
                next_word = tokenizer.decode(next_token_id, skip_special_tokens=True).strip()
                if next_word:
                    predictions.add(next_word.split(' ')[0]) # Take only the first word if it decodes to multiple
            
        st.session_state.predicted_words = list(predictions)[:3] # Show top 3 unique predictions
        # Move model back to CPU if desired for resource management, or keep on GPU
        # model.to("cpu")


# Display prediction buttons
if st.session_state.predicted_words:
    st.markdown("**Click a word to append it:**")
    cols = st.columns(len(st.session_state.predicted_words))
    for i, word in enumerate(st.session_state.predicted_words):
        if cols[i].button(word):
            # Append the chosen word and update input_text in session_state
            st.session_state.input_text = text + " " + word
            st.session_state.predicted_words = [] # Clear predictions after one is chosen
            st.rerun() # Rerun to update the text_area - CORRECTED LINE HERE

# Important for state management: ensure the text_area always reflects session_state.input_text
# This prevents it from reverting when a button is clicked and then rerunning.
# We set the 'value' of the text_area, and updates through button clicks modify st.session_state.input_text
# which then forces the text_area to reflect that new value on rerun.
st.session_state.input_text = text # Ensure this is updated for the next rerun if user types manually