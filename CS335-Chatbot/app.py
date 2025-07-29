from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re
from datetime import datetime

app = Flask(__name__)

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

class IntentClassifier:
    def __init__(self):
        self.patterns = {
            'personal_info': {
                'keywords': ['my', 'i am', 'i weigh', 'my height', 'my weight', 'my age', "i'm"],
                'regex': [
                    r'i am (\d+)',
                    r'i weigh (\d+)',
                    r'my weight is (\d+)',
                    r'my height is (\d+)',
                    r"i'm (\d+) (years old|feet|ft|inches|lbs|kg|pounds)",
                ]
            },
            'goal_setting': {
                'keywords': ['want to', 'goal', 'trying to', 'hoping to', 'plan to'],
                'regex': [
                    r'want to (lose weight|gain muscle|get fit|build strength)',
                    r'my goal is to',
                    r'trying to (lose|gain|build|improve)'
                ]
            },
            'question': {
                'keywords': ['how', 'what', 'when', 'where', 'why', 'can you', 'should i', 'is it'],
                'regex': [
                    r'^(how|what|when|where|why|can|should|is|are|do|does)',
                    r'\?$'
                ]
            },
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                'regex': [r'^(hello|hi|hey|good (morning|afternoon|evening))']
            }
        }

    def classify_intent(self, text):
        text_lower = text.lower().strip()
        for intent, patterns in self.patterns.items():
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    return intent
            
            for pattern in patterns.get('regex', []):
                if re.search(pattern, text_lower):
                    return intent
                
        return 'general'

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
                r"i'm (\d+)\s*(?:years old|years|yrs)",
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
                    if info_type == 'height' and len(match.groups()) == 2:
                        feet, inches = match.groups()
                        total_inches = int(feet) * 12 + int(inches)
                        extracted[info_type] = f"{feet}'{inches}\" ({total_inches} inches)"
                    else:
                        extracted[info_type] = match.group(1)
                    break
        return extracted

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
            "For weight loss: Combine cardio (30 mins/day) with strength training (3x/week) and maintain a calorie deficit.",
            "For muscle gain: Focus on progressive overload in strength training and consume 1g protein per pound of body weight.",
            "A good beginner workout: 3 sets of 10-12 reps for squats, push-ups, and rows, 3 times per week.",
            "Sample workout for today: Warm up for 5-10 minutes, then do 3 rounds of: 10 push-ups, 15 bodyweight squats, 20 jumping jacks.",
            "Best fruits for fitness: Bananas (potassium), berries (antioxidants), apples (fiber), oranges (vitamin C).",
            "For endurance: Try interval training - alternate between high intensity (30 sec) and rest (90 sec) for 15-20 minutes.",
            "To lose weight safely: Aim for 1-2 lbs per week through diet and exercise combined.",
            "For core strength: Planks (start with 30 sec), Russian twists, and leg raises are effective exercises.",
            "Post-workout recovery: Stretch for 5-10 minutes and consume protein within 30 minutes after training."
        ]
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_vectors = self.embedder.encode(self.knowledge_base)
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

    def find_relevant_context(self, question, user_profile=None):
        question = question.lower().strip()
        q_vector = self.embedder.encode([question])
        scores = np.inner(q_vector, self.kb_vectors)[0]
        top_indices = scores.argsort()[-3:][::-1]
        top_scores = [scores[i] for i in top_indices]
        threshold = 0.35
        filtered = [(self.knowledge_base[i], top_scores[idx]) for idx, i in enumerate(top_indices) if top_scores[idx] >= threshold]

        if filtered:
      
            previous_responses = set(user_profile.chat_history[-5:]) if user_profile else set()
            for response, score in filtered:
                if response not in previous_responses:
                    return response, score
            return filtered[0] 


        if "cardio" in question:
            return "For cardio, try running, cycling, or interval training for 20-30 minutes.", 0.0
        if "workout" in question or "exercise" in question:
            return "A good beginner workout: 3 sets of 10-12 squats, push-ups, and rows, 3 times per week.", 0.0
        if "lose weight" in question:
            return "To lose weight: Combine daily cardio with strength training 3x/week and maintain a calorie deficit.", 0.0

        return "I'm not sure, but consistency and hydration are always great for fitness!", 0.0


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
            response = f"Got it! I've updated your {', '.join(updated_fields)}."
            
            if user_profile.age and user_profile.weight:
                response += " Now you can ask me specific fitness questions!"
            else:
                if not user_profile.age:
                    response += " Could you also tell me your age?"
                if not user_profile.weight:
                    response += " Could you also tell me your weight?"
            
            return response
        
        return "Tell me more about your height, weight, age, or goals!"

    def handle_message(self, user_message):
        user_message = user_message.strip()
        if not user_message:
            return "Please type a message."

        if user_message.lower() in ['exit', 'quit', 'reset']:
            return self.handle_special_command(user_message.lower())

        if not self.current_user:
            name_match = re.search(r"my name is (\w+)|i'm (\w+)|call me (\w+)", user_message.lower())
            if name_match:
                name = next(group for group in name_match.groups() if group)
                self.current_user = self.get_or_create_user(name.capitalize())
                return f"Nice to meet you, {name.capitalize()}! Please tell me your age, height, and weight."
            return "Hello! 👋 I'm FitPro, your personal fitness assistant. What's your name?"

        intent = self.intent_classifier.classify_intent(user_message)

        if intent == 'personal_info':
            return self.handle_personal_info(user_message, self.current_user)

        if intent == 'question':
            if not (self.current_user.age and self.current_user.weight):
                return "I need to know your age and weight first to give proper advice."

            response, confidence = self.find_relevant_context(user_message, self.current_user)
         
            self.current_user.chat_history.append(response)
            self.save_users()
            return response

        return "I'm here to help with fitness advice! Could you ask a specific question or share your goals?"

    def handle_special_command(self, command):
        if command in ['exit', 'quit']:
            return "Goodbye! Come back anytime for fitness advice."
        elif command == 'reset':
            if self.current_user:
                del self.users[self.current_user.name]
                self.current_user = None
            return "Your profile has been reset. What's your name?"
        return "I didn't understand that command."

chatbot = EnhancedFitProChatbot()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': "Please type a message."})
    
    try:
        response = chatbot.handle_message(user_message)
    except Exception as e:
        print(f"Error handling message: {e}")
        response = "Sorry, I encountered an error. Please try again."
    
    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=True)





