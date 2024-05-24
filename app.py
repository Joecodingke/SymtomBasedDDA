import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained model and the vectorizer
with open('lgbm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Assuming the vectorizer was saved in the same way
with open('lgbm_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI components
st.title('Hate Speech Detection App')

# Text input
user_input = st.text_area("Enter the text you want to analyze:")

# Button to predict
if st.button('Predict'):
    # Vectorize the user input
    vect_input = vectorizer.transform([user_input])

    # Make a prediction
    prediction = model.predict(vect_input)

    # Display the result
    if prediction == 0:
        st.success("This text is NOT hate speech.")
    else:
        st.error("This text is HATE speech.")

# # Run the Streamlit app
# if __name__ == '__main__':
#     st.run()