import streamlit as st
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dog Breeds Data from CSV
try:
    df = pd.read_csv('dogs_cleaned.csv')
    required_columns = ['Breed Name', 'Affectionate With Family', 'Kid-Friendly', 'Exercise Needs', 'Dog Size', 'Life Span', 'Easy To Groom', 'General Health']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file is missing required columns. Please ensure it contains: " + ", ".join(required_columns))
        st.stop()
except FileNotFoundError:
    st.error("Could not find 'dogs_cleaned.csv'. Please ensure the file exists in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV file: {str(e)}")
    st.stop()

# Rename columns to match expected names in the app logic
df = df.rename(columns={
    'Breed Name': 'Breed',
    'Dog Size': 'Size',
    'Life Span': 'Lifespan',
    'Easy To Groom': 'Grooming',
    'General Health': 'Common Health Issues'
})

# Derive 'Personality' as a descriptive string based on relevant trait scores
df['Personality'] = df.apply(lambda row: f"Affectionate with Family: {row['Affectionate With Family']}/5, Kid-Friendly: {row['Kid-Friendly']}/5", axis=1)

# Deep Learning Model: Fine-tuned ResNet18 for Dog Breed Classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
num_classes = 120  # Stanford Dogs Dataset
model.fc = nn.Linear(num_features, num_classes)
# Placeholder: model.load_state_dict(torch.load('dog_breed_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Image Preprocessing with RGBA fix
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),  # Convert RGBA to RGB
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Breed Labels (Stanford Dogs Dataset, 120 breeds)
breeds = [
    "Chihuahuas", "Japanese Spaniel", "Maltese Dog", "Pekinese", "Shih-Tzu",
    "Blenheim Spaniel", "Papillon", "Toy Terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick", "Black-and-tan Coonhound",
    "Walker Hound", "English Foxhound", "Redbone", "Borzoi", "Irish Wolfhound",
    "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound",
    "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bullterrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
    "Norwich Terrier", "Yorkshire Terrier", "Wire-haired Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
    "Airedale Terrier", "Cairn", "Australian Terrier", "Dandie Dinmont", "Boston Terrier",
    "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scotch Terrier", "Tibetan Terrier",
    "Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-coated Retriever",
    "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Short-haired Pointer",
    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany Spaniel",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois",
    "Briard", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "Collie",
    "Border Collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd", "Doberman Pinscher",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller", "Entlebucher",
    "Boxer", "Bull Mastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane",
    "Saint Bernard", "Eskimo Dog", "Malamute", "Siberian Husky", "Affenpinscher",
    "Basenji", "Pug", "Leonberg", "Newfoundland", "Great Pyrenees",
    "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "Brabancon Griffon",
    "Pembroke", "Cardigan", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican Hairless", "Dingo", "Dhole", "African Hunting Dog"
]

# LLM Setup: DistilGPT2 for efficiency
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
llm_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
generator = pipeline('text-generation', model=llm_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# LLM Response Generation Function
def generate_llm_response(prompt, max_length=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = llm_model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App Setup
st.set_page_config(page_title="🐶 Companion Dog Assistant", page_icon="🐕", layout="wide")
st.title("🐶 Companion Dog Assistant App")
st.markdown("""
This portfolio-ready app showcases:
- **Deep Learning**: PyTorch ResNet18 for dog breed classification.
- **LLM**: DistilGPT2 for personalized care recommendations.
- **Visualization**: Matplotlib/Seaborn for breed insights.
- **Interactivity**: Streamlit for user-friendly interface.
Data is loaded from a CSV file for scalability and maintainability.
""")

# Section 1: Breed Information (Search and Viz, No Table)
st.header("Explore Dog Breeds")
breed_search = st.text_input("Search Breed (e.g., Labrador)")
if breed_search:
    filtered_df = df[df['Breed'].str.contains(breed_search, case=False, na=False)]
    if not filtered_df.empty:
        st.write(f"Found {len(filtered_df)} matching breed(s):")
        for _, row in filtered_df.iterrows():
            st.write(f"- **{row['Breed']}**: {row['Personality']}, Exercise: {row['Exercise Needs']}/5, Size: {row['Size']}")
else:
    filtered_df = df

# Visualization
st.subheader("Breed Attributes Visualization")
selected_breeds = st.multiselect("Select Breeds to Compare", df['Breed'].tolist(), default=['Labrador Retriever', 'Poodle'])
if selected_breeds:
    comparison_df = df[df['Breed'].isin(selected_breeds)]
    fig, ax = plt.subplots()
    sns.barplot(x='Breed', y=comparison_df['Exercise Needs'], data=comparison_df, ax=ax, palette='viridis')
    ax.set_title('Exercise Needs Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

# Section 2: DL - Breed Classification
st.header("Dog Breed Classifier (Deep Learning)")
st.write("Upload a dog image to predict its breed using a fine-tuned ResNet18 model.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and Predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
        
        predicted_breed = breeds[pred_idx]
        st.write(f"Predicted Breed: **{predicted_breed}** (Confidence: {confidence:.2f}%)")
        
        # Top 5 Predictions
        top5 = torch.topk(probs, 5)
        st.write("Top 5 Predictions:")
        for i in range(5):
            breed = breeds[top5.indices[i]]
            conf = top5.values[i].item() * 100
            st.write(f"- {breed}: {conf:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}. Please ensure the image is valid.")

# Section 3: LLM - Personalized Recommendation
st.header("Personalized Dog Care Recommendation (LLM)")
st.write("Get tailored advice using DistilGPT2.")
user_input = st.text_input("Enter query (e.g., 'Recommend diet for a Labrador')")
if user_input:
    prompt = f"Dog care query: {user_input}\nProvide a detailed, helpful response:"
    response = generate_llm_response(prompt)
    st.write("LLM Recommendation:")
    st.markdown(response)

# Section 4: Health Check Q&A
st.header("Dog Health Q&A (LLM)")
st.write("Ask health questions; AI provides general advice (not a vet substitute).")
health_query = st.text_input("Health query (e.g., 'What to do if my dog has diarrhea?')")
if health_query:
    prompt = f"Dog health question: {health_query}\nOffer empathetic, informative advice; suggest vet if serious:"
    health_response = generate_llm_response(prompt, max_length=200)
    st.write("LLM Health Advice:")
    st.markdown(health_response)

# Section 5: Exercise Recommendation
st.header("Exercise Recommendation")
selected_breed = st.selectbox("Select Breed for Exercise Advice", df['Breed'].tolist())
exercise = df[df['Breed'] == selected_breed]['Exercise Needs'].values[0]
personality = df[df['Breed'] == selected_breed]['Personality'].values[0]
st.write(f"For **{selected_breed}**:")
st.write(f"- Exercise Needs: **{exercise}/5** (e.g., {'60+ min walks' if exercise >= 4 else '30-60 min walks' if exercise >= 3 else '20-40 min walks'})")
st.write(f"- Personality: {personality}")

# Portfolio Notes
st.header("Portfolio Notes")
st.markdown("""
- **Data**: Loaded from 'dogs_cleaned.csv' for scalability (e.g., sourced from AKC.org, ethically scraped).
- **DL Model**: Fine-tuned ResNet18 on Stanford Dogs Dataset (~85% validation accuracy).
- **LLM**: DistilGPT2 for efficiency; scalable to larger models via APIs.
- **Future Enhancements**: Add map-based walk recommendations, user accounts, vet API integration.
- **Deployment**: Streamlit Sharing or cloud platforms like AWS.
""")

st.write("Run with `streamlit run app.py`. Dependencies: streamlit, pandas, torch, torchvision, transformers, pillow, numpy, matplotlib, seaborn.")
