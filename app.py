
# pip install tensorflow==2.15.0
# pip install torch==2.0.1
# pip install sentence_transformers==2.2.2
# pip install streamlit

# import libraries===================================
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# load save recommendation models===================================

embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

# load save prediction models============================
# Load the model
loaded_model = keras.models.load_model("models/model.h5")
# Load the configuration of the text vectorizer
with open("text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
# Create a new TextVectorization layer with the saved configuration
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)
# Load the saved weights into the new TextVectorization layer
with open("text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)

# Load the vocabulary
with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)


# custom functions====================================

def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list



#making the invert_muti_hot fucntion for its utilization in the predict_category function
import tensorflow as tf
# This code snippet is iterating through batches of the training dataset and printing the abstract text along with the corresponding labels.
# Example data (replace with your actual data)
text_batch = ["Abstract 1", "Abstract 2", "Abstract 3", "Abstract 4", "Abstract 5"]
label_batch = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])

# Create the StringLookup layer and its inverse
label_lookup = tf.keras.layers.StringLookup(vocabulary=["label1", "label2", "label3"], mask_token=None)
inverse_label_lookup = tf.keras.layers.StringLookup(vocabulary=label_lookup.get_vocabulary(), invert=True, mask_token=None)

# Define the invert_multi_hot function

def invert_multi_hot(label):
    indices = tf.where(label > 0)  # Get indices of non-zero elements
    inverted_labels = inverse_label_lookup(indices)
    return inverted_labels

# Iterate through the batch and print the abstract and corresponding labels
for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")

def predict_category(multi_hot_vector, category_names):
    label_lookup = tf.keras.layers.StringLookup(vocabulary=category_names, mask_token=None)
    inverse_label_lookup = tf.keras.layers.StringLookup(vocabulary=label_lookup.get_vocabulary(), invert=True, mask_token=None)
    
    indices = tf.where(multi_hot_vector > 0)  # Get indices of non-zero elements
    inverted_labels = inverse_label_lookup(indices)
    categories = tf.squeeze(inverted_labels, axis=-1)
    
    return categories

# Iterate through the batch and print the abstract and corresponding labels
for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")

#adjust_shape
def adjust_shape(preprocessed_abstract, target_shape):
    current_shape = preprocessed_abstract.shape[1]
    if current_shape < target_shape:
        # Pad with zeros if the current shape is smaller
        padding = np.zeros((preprocessed_abstract.shape[0], target_shape - current_shape))
        adjusted_abstract = np.hstack((preprocessed_abstract, padding))
    elif current_shape > target_shape:
        # Trim if the current shape is larger
        adjusted_abstract = preprocessed_abstract[:, :target_shape]
    else:
        adjusted_abstract = preprocessed_abstract
    return adjusted_abstract

def predict_category(abstract, model, vectorizer, label_lookup):
    # Preprocess the abstract using the loaded text vectorizer
    preprocessed_abstract = vectorizer([abstract])  # Using vectorizer as a callable to preprocess the text

    # Debugging information to check the shapes
    print(f"Shape of preprocessed abstract: {preprocessed_abstract.shape}")
    print(f"Expected input shape of the model: {model.input_shape}")

    # Adjust the shape to match the model input shape
    adjusted_abstract = adjust_shape(preprocessed_abstract, model.input_shape[1])

    # Make predictions using the loaded model
    predictions = model.predict(adjusted_abstract)

    # Convert predictions to human-readable labels
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels



#the app===============================================

# create app=========================================
st.title('Research Papers Recommendation App')
st.write("NLP and Deep Learning Base App")

input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Enter paper abstract....")
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation(input_paper)
    st.subheader("Recommended Papers")
    st.write(recommend_papers)

 
