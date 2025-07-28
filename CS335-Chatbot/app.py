from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import json
import os
import re
from datetime import datetime

app = Flask(__name__)


# User Profile Class
class UserProfile:
    def __init__(self, name=""):
        self.name = name
        self.height = None
        self.weight = None
        self.age = None
        self.fitness_goals = []
        self.preferences = {}
        self.chat_history = []
        self.last_updated = datetime.now().isoformat()

    def to_dict(self):
        return {
            'name': self.name,
            'height': self.height,
            'weight': self.weight,
            'age': self.age,
            'fitness_goals': self.fitness_goals,
            'preferences': self.preferences,
            'chat_history': self.chat_history,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data):
        profile = cls(data.get('name', ''))
        profile.height = data.get('height')
        profile.weight = data.get('weight')
        profile.age = data.get('age')
        profile.fitness_goals = data.get('fitness_goals', [])
        profile.preferences = data.get('preferences', {})
        profile.chat_history = data.get('chat_history', [])
        profile.last_updated = data.get('last_updated')
        return profile


# Intent Classifier
class IntentClassifier:
    def __init__(self):
        self.patterns = {
            'personal_info': {
                'keywords': ['my', 'i am', 'i weigh', 'my height', 'my weight', 'my age', 'i\'m'],
            },
            'goal_setting': {
                'keywords': ['want to', 'goal', 'trying to', 'hoping to', 'plan to'],
            },
            'question': {
                'keywords': ['how', 'what', 'when', 'where', 'why', 'can you', 'should i', 'is it'],
            },
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            }
        }

    def classify_intent(self, text):
        text_lower = text.lower().strip()
        for intent, patterns in self.patterns.items():
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    return intent
        return 'general'


# Personal Info Extractor
class PersonalInfoExtractor:
    def __init__(self):
        self.extractors = {
            'height': [
                r'(\d+)\s*(?:feet|ft|\')\s*(\d+)\s*(?:inches|in|\")',
                r'(\d+)\s*(?:feet|ft|\')',
                r'(\d+\.?\d*)\s*(?:cm|centimeters)',
                r'my height is (\d+\.?\d*)',
                r'i am (\d+\.?\d*)\s*(?:feet|ft|cm|tall)'
            ],
            'weight': [
                r'(\d+\.?\d*)\s*(?:lbs|pounds|lb)',
                r'(\d+\.?\d*)\s*(?:kg|kilograms)',
                r'i weigh (\d+\.?\d*)',
                r'my weight is (\d+\.?\d*)'
            ],
            'age': [
                r'i am (\d+)\s*(?:years old|years|yrs)',
                r'i\'m (\d+)\s*(?:years old|years|yrs)',
                r'my age is (\d+)',
                r'(\d+)\s*years old'
            ],
            'name': [
                r'(?:my name is|i\'m|call me)?\s*(\w+)',
            ]
        }

    def extract_info(self, text):
        extracted = {}
        text_lower = text.lower()
        for info_type, patterns in self.extractors.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    extracted[info_type] = match.group(1)
                    break
        return extracted


# Main Chatbot Logic
class EnhancedFitProChatbot:
    def __init__(self):
        self.knowledge_base = [
            "Drinking water before, during, and after workouts improves performance and recovery.",
            "Staying hydrated helps regulate body temperature during exercise.",
            "Too much caffeine can cause dehydration and mess with your sleep.",
            "Sleep is just as important as diet and exercise.",
            "Meal prepping helps with portion control and tracking your nutrition.",
            "Yoga improves balance, flexibility, and focus when practiced regularly.",
            "Push-ups work the chest, shoulders, triceps, arms, and core.",
            "Consistency is more important than intensity — small regular workouts are best.",
            "Walking or biking to class counts toward your daily activity goals.",
            "Eating protein helps muscles heal and grow after workouts.",
            "Warming up increases blood flow and prepares your body for activity.",
            "Squats target the glutes, quads, and hamstrings for strong legs.",
            "Foam rolling can reduce muscle soreness and speed up recovery.",
            "Rest days help your muscles recover and grow stronger.",
            "Cycling is a low-impact cardio workout that strengthens your legs and heart.",
            "Good form prevents injuries and gets better results.",
            "Added sugar can lead to energy crashes and weight gain.",
            "Healthy fats support brain health and hormone balance.",
            "Bananas and oranges give quick energy and important vitamins.",
            "Small habits, repeated daily, lead to big results over time.",
            "You don't have to be perfect — just keep showing up.",
            "Avocados are full of potassium and help support healthy blood pressure.",
            "Stretching after a workout can improve flexibility and reduce soreness.",
            "Salmon is a great source of protein and omega-3 fatty acids.",
            "Getting sunlight during the day helps regulate your sleep cycle.",
            "Deep breathing and mindfulness can help reduce stress before workouts.",
            "Resistance bands are an affordable way to build strength at home.",
            "High-fiber foods help with digestion and make you feel fuller longer.",
            "Your body needs rest to build strength — rest is part of training.",
            "Whole grains give lasting energy and support steady blood sugar."
        ]
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_vectors = self.embedder.encode(self.knowledge_base)
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        self.intent_classifier = IntentClassifier()
        self.info_extractor = PersonalInfoExtractor()
        self.users_file = "user_profiles.json"
        self.current_user = None
        self.users = self.load_users()

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    return {name: UserProfile.from_dict(profile_data)
                            for name, profile_data in data.items()}
            except:
                return {}
        return {}

    def save_users(self):
        try:
            data = {name: profile.to_dict() for name, profile in self.users.items()}
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving user data: {e}")

    def get_or_create_user(self, name):
        if name not in self.users:
            self.users[name] = UserProfile(name)
            self.save_users()
        return self.users[name]

    def handle_greeting(self, user_profile):
        return f"Hello {user_profile.name if user_profile else 'there'}! 👋 How can I help you with your fitness journey today?"

    def handle_question(self, question, user_profile):
        q_vector = self.embedder.encode([question])
        scores = np.inner(q_vector, self.kb_vectors)
        best_index = np.argmax(scores)
        context = self.knowledge_base[best_index]
        prompt = f"Answer based on context: {context}\nQuestion: {question}"
        response = self.qa_pipeline(prompt, max_new_tokens=80)[0]['generated_text']
        return response.strip()

    def handle_personal_info(self, text, user_profile):
        extracted = self.info_extractor.extract_info(text)
        updated_fields = []
        for field, value in extracted.items():
            if field == 'name':
                continue
            setattr(user_profile, field, value)
            updated_fields.append(f"{field}: {value}")
        if updated_fields:
            user_profile.last_updated = datetime.now().isoformat()
            self.save_users()
            return f"Got it! I've updated your {', '.join(updated_fields)}."
        return "Tell me more about your height, weight, age, or goals!"

    def handle_message(self, user_message):
        intent = self.intent_classifier.classify_intent(user_message)

        # If no user profile yet
        if not self.current_user:
            name_match = re.search(r'(?:my name is|i\'m|call me)?\s*(\w+)', user_message.lower())
            if name_match:
                name = name_match.group(1).capitalize()
                self.current_user = self.get_or_create_user(name)
                return f"Nice to meet you, {name}! How can I help you today?"
            return "Hi there! I'd love to get to know you. What's your name?"

        # Known user
        if intent == 'greeting':
            return self.handle_greeting(self.current_user)
        elif intent == 'personal_info':
            return self.handle_personal_info(user_message, self.current_user)
        elif intent == 'goal_setting':
            goals = []
            goal_patterns = ['lose weight', 'gain muscle', 'get fit', 'build strength', 'improve endurance']
            for goal in goal_patterns:
                if goal in user_message.lower():
                    goals.append(goal)

            if goals:
                self.current_user.fitness_goals.extend(goals)
                self.current_user.fitness_goals = list(set(self.current_user.fitness_goals))  # Remove duplicates
                self.save_users()
                return f"Great goals! I've noted that you want to {', '.join(self.current_user.fitness_goals)}."
            else:
                return "Tell me more about your goals, like losing weight or building strength!"
        elif intent == 'question':
            return self.handle_question(user_message, self.current_user)
        else:
            return "I can help with fitness tips, nutrition, or workouts! Ask me anything."


chatbot = EnhancedFitProChatbot()

# Flask Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    response = chatbot.handle_message(user_message)
    return jsonify({'reply': response})


if __name__ == '__main__':
    app.run(debug=True)


