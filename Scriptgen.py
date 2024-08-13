import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")

# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def generate_sql(prompt, max_length=150, temperature=0.5, top_k=50, top_p=0.95):
    context_prompt = (
        f"Translate the following natural language request into a SQL query. "
        f"Return only the SQL code without any additional comments, explanations, or code:\n"
        f"Request: {prompt}\n"
        f"SQL Query:"
    )

    # Tokenize input prompt
    inputs = tokenizer(context_prompt, return_tensors="pt", padding=True, truncation=True)
    # Generate the output
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature, 
        top_k=top_k,
        top_p=top_p,
        do_sample=True,  # Enable sampling for variability
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL script from generated text
    sql_script = generated_text.split(';')[0].strip()
    
    # Ensure the SQL statement ends with a semicolon
    if not sql_script.endswith(';'):
        sql_script += ';'
    
    return sql_script

# Streamlit UI
st.title("SQL Query Generator")

# Session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input field
prompt = st.text_input("Enter your SQL prompt:")

if st.button("Generate SQL"):
    if prompt:
        sql_query = generate_sql(prompt)
        st.write(f"Generated SQL Query: {sql_query}")
        # Save to history
        st.session_state.history.append({"prompt": prompt, "sql_query": sql_query})
    else:
        st.error("Please enter a prompt.")

# Display history
if st.session_state.history:
    st.subheader("History")
    for entry in st.session_state.history:
        st.write(f"**Prompt:** {entry['prompt']}")
        st.write(f"**SQL Query:** {entry['sql_query']}")
        st.write("---")
