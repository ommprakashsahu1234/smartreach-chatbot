#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import random
from textblob import TextBlob
import nltk
import spacy

print("✅ All AI/NLP libraries imported successfully!")

# Download required models only if not already downloaded (run once)
os.system('python -m spacy download en_core_web_sm')
nltk.download('punkt')
nltk.download('stopwords')
print("✅ NLP models for spaCy and NLTK are ready!")

# Load datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(BASE_DIR, "datasets")

academic_stress = pd.read_csv(os.path.join(datasets_path, "academic-stress-maintenance.csv"))
stress_factors = pd.read_csv(os.path.join(datasets_path, "stressfactors.csv"))
student_scores = pd.read_csv(os.path.join(datasets_path, "student_exam_scores.csv"))
student_performance = pd.read_csv(os.path.join(datasets_path, "student_performance.csv"))
performance_factors = pd.read_csv(os.path.join(datasets_path, "StudentPerformanceFactors.csv"))
social_media = pd.read_csv(os.path.join(datasets_path, "studentssocialmediaaddiction.csv"))
ai_qa = pd.read_csv(os.path.join(datasets_path, "AI.csv")).fillna("")

print("✅ All datasets loaded successfully!")

# Normalize social media platform column for reliable filtering
social_media['Most_Used_Platform'] = social_media['Most_Used_Platform'].str.strip().str.lower()

# Prepare chatbot knowledge and embeddings
chatbot_knowledge = []
for idx, row in ai_qa.iterrows():
    chatbot_knowledge.append({"question": row['Question'], "answer": row['Answer']})

questions = [item['question'] for item in chatbot_knowledge]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embed_model.encode(questions, convert_to_numpy=True)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

print(f"✅ Generated embeddings for {len(questions)} questions")
print("✅ FAISS index built for semantic search!")

# NLP helper functions
def detect_sentiment(message):
    analysis = TextBlob(message)
    polarity = analysis.sentiment.polarity  # ranges -1 to 1
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def suggest_coping_strategy(stress_index):
    if stress_index <= 2:
        return "Your stress level seems low. Keep up your good habits!"
    elif stress_index <= 4:
        return "Moderate stress detected. Try techniques like 'Analyze the situation and handle it with intellect' or seek social support."
    else:
        return "High stress detected! Consider taking breaks, talking to family/friends, or consulting a counselor."

# Main semantic search function with relaxed threshold
def get_best_answer(query, top_k=1, threshold=5.0):
    query_vec = embed_model.encode([query.lower()])
    distances, indices = index.search(np.array(query_vec), top_k)
    if distances[0][0] > threshold:
        return None
    best_idx = indices[0][0]
    return ai_qa.iloc[best_idx]['Answer']

# Chatbot main response logic
def get_chatbot_response(user_message):
    user_input_lower = user_message.lower()
    mood = detect_sentiment(user_message)

    # Stress related queries
    if "stress" in user_input_lower or "tired" in user_input_lower:
        response = suggest_coping_strategy(stress_index=3)
        return f"[Mood: {mood}] {response}"

    # Academic performance questions
    if any(kw in user_input_lower for kw in ["performance", "risk", "predict"]):
        if not student_performance.empty and 'total_score' in student_performance.columns:
            avg_score = student_performance['total_score'].mean()
            return f"Your predicted academic score is approximately {avg_score:.2f}%. Focus on consistent study and practice."
        else:
            return "I currently do not have enough data to predict your score."

    # Social media related queries
    social_media_platforms = social_media['Most_Used_Platform'].unique().tolist()
    if any(kw in user_input_lower for kw in ["social media"] + social_media_platforms):
        if social_media.empty:
            return "I cannot respond to that. I am an educational chatbot."

        harm_words = ["bad", "harm", "affect", "negative", "distraction", "risk"]
        harm_question = any(word in user_input_lower for word in harm_words)
        mentioned_platforms = [p for p in social_media_platforms if p in user_input_lower]

        if harm_question:
            if mentioned_platforms:
                platform = mentioned_platforms[0]
                platform_users = social_media[social_media['Most_Used_Platform'] == platform]
                count_users = len(platform_users)
                if count_users == 0:
                    return f"No data available for {platform.title()} users."
                avg_hours = platform_users['Avg_Daily_Usage_Hours'].mean()
                if avg_hours > 4.5:
                    return f"Yes, using {platform.title()} extensively can negatively affect your studies due to potentially high distraction."
                else:
                    return f"{platform.title()} usage is moderate and may not severely impact studies if managed well."
            else:
                top_platforms = social_media.groupby('Most_Used_Platform')['Avg_Daily_Usage_Hours'].mean().sort_values(ascending=False).head(3)
                top_str = ", ".join([p.title() for p in top_platforms.index])
                return f"The platforms most impacting study time are {top_str}. Use them moderately to avoid distractions."
        else:
            if mentioned_platforms:
                platform = mentioned_platforms[0]
                platform_users = social_media[social_media['Most_Used_Platform'] == platform]
                if not platform_users.empty:
                    avg_hours = platform_users['Avg_Daily_Usage_Hours'].mean()
                    return f"On average, students who use {platform.title()} spend {avg_hours:.2f} hours per day. Monitor usage to reduce distraction."
                else:
                    return f"No data available for {platform.title()} users."
            else:
                avg_hours = social_media['Avg_Daily_Usage_Hours'].mean()
                return f"On average, students spend {avg_hours:.2f} hours per day on social media. Monitor usage to reduce distraction."

    # Fallback semantic search
    answer = get_best_answer(user_message)
    if answer:
        return answer
    return "I cannot respond to that. I am an educational chatbot."

# Interactive loop to test chatbot
def main():
    print("🤖 Welcome to EduCRM Chatbot with Stress & Sentiment Detection! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🤖 Goodbye!")
            break
        answer = get_chatbot_response(user_input)
        print("Chatbot:", answer)

if __name__ == "__main__":
    main()
