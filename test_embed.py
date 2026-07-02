from sentence_transformers import SentenceTransformer

# load the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# The text to be embedded
text = "How do I reset my password?"

# Generate the embedding for the text
vector = model.encode(text)

print("\n Embedding example for the text: ", text)
print("Vector size:", len(vector))
print("Vector:", vector)