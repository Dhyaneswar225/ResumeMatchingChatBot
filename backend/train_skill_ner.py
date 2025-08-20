import spacy
from spacy.training import Example
import json

# Load training data from JSON file
with open("train_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert raw_data to spaCy format
TRAIN_DATA = []
for item in raw_data:
    entities = [tuple(ent) for ent in item["entities"]]
    TRAIN_DATA.append((item["text"], {"entities": entities}))

# Create blank English model
nlp = spacy.blank("en")

# Add NER pipeline to the model
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add the new label to the NER pipe
ner.add_label("SKILL")

# Initialize the model and optimizer
optimizer = nlp.initialize()

# Prepare training examples
examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, annotations))

# Training loop
n_iter = 30
for i in range(n_iter):
    losses = {}
    nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Iteration {i + 1}, Losses: {losses}")

# Save the trained model to disk
output_dir = "./skill_ner_model"
nlp.to_disk(output_dir)
print(f"\nModel saved to {output_dir}")

# Test the trained model
print("\nTesting the trained model:")
test_text = "I have experience with Python, Java, Docker, Javascript, MongoDB and Kubernetes."
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)