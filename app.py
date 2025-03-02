import pickle
import streamlit as st

# Load the model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("TWITTER SENTIMENT ANALYSIS")
st.write("Please feed me your tweet")
userinput = st.text_area('Paste the tweet here', height=150)

if st.button("Predict"):
    if userinput.strip():
        # Convert text input to vectorized form
        user_input_vectorized = vectorizer.transform([userinput])

        # Predict sentiment
        prediction = model.predict(user_input_vectorized)
        prediction_prob = model.predict_proba(user_input_vectorized)  # Get probability if applicable

        # Debugging: Print outputs
        st.write(f"Raw Model Output: {prediction}")
        st.write(f"Prediction Probabilities: {prediction_prob}")

        # Display result
        if prediction[0] == 4:  # Check integer value
            st.success("This tweet is positive 😊")
        else:
            st.error("This tweet is negative 😞")
    else:
        st.warning("Please enter some text.")

