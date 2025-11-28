import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
import os
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from datetime import datetime
from openai import OpenAI
import torch
import json
import uuid
import random
import re
from twilio.rest import Client
import requests
from datetime import datetime



@st.cache_resource
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return model, tokenizer

model, tokenizer = load_model()


# ===============================
# ⚙️ Streamlit Page Config
# ===============================
st.set_page_config(page_title="🧠 AI Mental Health Companion", layout="wide")

# ===============================
# 📁 Per-user Long-Term Memory Helpers
# ===============================
# We'll create a stable user_id per deployment (saved to disk) and per-user memory files:
# memory_<user_id>.json

USER_ID_FILE = "user_id.txt"

def get_or_create_user_id():
    # If session already has, reuse
    if "user_id" in st.session_state:
        return st.session_state.user_id

    # Try persistent file on disk (so restarting server keeps same id)
    if os.path.exists(USER_ID_FILE):
        try:
            with open(USER_ID_FILE, "r") as f:
                user_id = f.read().strip()
                if user_id:
                    st.session_state.user_id = user_id
                    return user_id
        except Exception:
            pass

    # Otherwise generate a new stable uuid and save it
    user_id = str(uuid.uuid4())
    try:
        with open(USER_ID_FILE, "w") as f:
            f.write(user_id)
    except Exception:
        # Not critical if saving fails (e.g., permission). Still keep in session.
        pass

    st.session_state.user_id = user_id
    return user_id

def memory_filename(user_id):
    return f"memory_{user_id}.json"

def load_long_term_memory(user_id):
    fname = memory_filename(user_id)
    if not os.path.exists(fname):
        return []
    try:
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_long_term_memory(user_id, memories):
    fname = memory_filename(user_id)
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

def append_memory(user_id, role, message):
    memories = load_long_term_memory(user_id)
    memories.append({
        "role": role,
        "message": message,
        "time": datetime.now().isoformat()
    })
    # Keep file trimmed to a reasonable size (e.g., last 200 entries)
    memories = memories[-200:]
    save_long_term_memory(user_id, memories)

def get_memory_context(user_id, last_n=8):
    memories = load_long_term_memory(user_id)
    # Build a readable memory block from last_n pairs (we'll use last_n entries)
    recent = memories[-last_n:]
    lines = []
    for item in recent:
        role = item.get("role", "User")
        msg = item.get("message", "")
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)

# Ensure user_id exists for this session
user_id = get_or_create_user_id()


# ===============================
# 💬 AI Response Integration (GPT + Mistral Fallback)
# ===============================
use_openai = False
client = None
mistral_tokenizer, mistral = None, None


try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    use_openai = True
    print("✅ OpenAI API key loaded successfully.")
except Exception as e:
    print("⚠️ OpenAI unavailable, using fallback model:", e)
    use_openai = False

# --- Initialize fallback model (Phi-2 or DistilGPT2) ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize fallback model (Mistral or smaller) ---
if not use_openai:
    try:
        # Try smaller open-source model first
        fallback_model = "microsoft/phi-2"
        mistral_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        mistral = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float32
        ).to(device)
        print("✅ Fallback model (Phi-2) loaded successfully.")
    except Exception as e1:
        print("⚠️ Phi-2 failed, trying DistilGPT2:", e1)
        try:
            fallback_model = "distilgpt2"
            mistral_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            mistral = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32
            ).to(device)
            print("✅ Lightweight fallback (DistilGPT2) loaded successfully.")
        except Exception as e2:
            print("❌ Both Phi-2 and DistilGPT2 failed to load:", e2)
            mistral_tokenizer, mistral = None, None

# --- Safety check ---
if mistral_tokenizer is None or mistral is None:
    print("⚠️ No model could be initialized — forcing DistilGPT2 CPU mode.")
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    mistral_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    mistral = GPT2LMHeadModel.from_pretrained(
        "distilgpt2",
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )
    print("✅ Emergency fallback: DistilGPT2 (CPU mode) loaded successfully.")


def get_ai_reply(user_msg):
    global use_openai, mistral_tokenizer, mistral, user_id

    # ---- Build memory context (last 8 messages) ----
    memory_context = get_memory_context(user_id, last_n=8)

    prompt = f"""
You are Maya, a supportive, empathetic mental health companion.

Below is the recent conversation history. Continue the conversation naturally.

Also, if the user mentions:
- movies → extract a movie name and label it as MOVIE: <name or None>
- songs/music → extract a song name and label it as SONG: <name or None>

Format your output ONLY like this:

REPLY: <your message>
MOVIE: <movie or None>
SONG: <song or None>

STRICT RULES:
- You MUST output exactly 3 fields.
- Do NOT add any extra sections (NO disclaimers, NO picture info, NO warnings, NO titles).
- The reply should be natural and emotional, but ONLY inside the REPLY field.
- MOVIE and SONG must contain only a title or "None".
- Do NOT create any new categories.
- Do NOT talk about formatting rules.

Conversation:
{memory_context}

User: {user_msg}
Maya:
"""

    # =====================================================================
    # 1️⃣ OpenAI GPT RESPONSE
    # =====================================================================
    raw_output = None
    if use_openai and client is not None:
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are Maya, a supportive mental health companion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            raw_output = completion.choices[0].message["content"].strip()
        except Exception as e:
            print("⚠️ OpenAI error, using fallback:", e)
            raw_output = None

    # =====================================================================
    # 2️⃣ FALLBACK MODEL IF OPENAI FAILED
    # =====================================================================
    if raw_output is None:
        try:
            inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral.device)

            outputs = mistral.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2
            )

            raw_output = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print("⚠️ Fallback model failed:", e)
            return "I'm here for you. Tell me what's on your mind."

    # =====================================================================
    # 3️⃣ CLEANUP & PARSE REPLY / MOVIE / SONG
    # =====================================================================
    import re

    # Old cleanup logic
    if "Maya:" in raw_output:
        raw_output = raw_output.split("Maya:")[-1]
    if "User:" in raw_output:
        raw_output = raw_output.split("User:")[0]
    raw_output = raw_output.strip()
    if not raw_output or len(raw_output.split()) < 3:
        raw_output = "I'm here for you. Tell me what's going on."

    # Parse REPLY / MOVIE / SONG
    reply_match = re.search(r"REPLY:\s*(.*?)(?=\nMOVIE:|\Z)", raw_output, re.S)
    movie_match = re.search(r"MOVIE:\s*(.*?)(?=\nSONG:|\Z)", raw_output, re.S)
    song_match = re.search(r"SONG:\s*(.*)", raw_output, re.S)


    reply = reply_match.group(1).strip() if reply_match else raw_output.strip()
    movie = movie_match.group(1).strip() if movie_match else "None"
    song = song_match.group(1).strip() if song_match else "None"

    # Normalize invalid values
    if not movie or movie.lower() in ["none", "null", "undefined"]:
        movie = None
    if not song or song.lower() in ["none", "null", "undefined"]:
        song = None

    # Limit reply length
    reply = reply[:400]

    # =====================================================================
    # 4️⃣ BUILD YOUTUBE + SPOTIFY LINKS
    # =====================================================================
    def yt_link(title):
        if not title or title.lower() == "none":
            return None
        return f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+trailer"

    def sp_link(title):
        if not title or title.lower() == "none":
            return None
        return f"https://open.spotify.com/search/{title.replace(' ', '+')}"

    movie_url = yt_link(movie)
    song_url = sp_link(song)


    # return final
    return reply, movie_url, song_url




# # ===============================
# # 🔒 Email Credentials (from secrets or fallback)
# # ===============================
# if "email" not in os.environ:
#     os.environ["email"] = "ananthakrishnang793@gmail.com"
# if "password" not in os.environ:
#     os.environ["password"] = "pznu guyb gedz ewcx"

# # ===============================
# # 📧 Email Alert Function
# # ===============================
# def send_email_alert(contact_email, user_message, risk_label):
#     sender = "maya793@gmail.com"
#     password = "pznu guyb gedz ewcx"   # Gmail App Password

#     subject = f"[ALERT] Critical Risk Detected ({risk_label})"
#     body = f"""
# Dear Contact,

# This is an automated alert from the Mental Health Chat Companion.
# A high-risk message was detected:

# "{user_message}"

# Please reach out to the individual immediately.

# — AI Companion System
# """
#     msg = MIMEText(body)
#     msg["Subject"] = subject
#     msg["From"] = sender
#     msg["To"] = contact_email

#     try:
#         # Connect & login
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#             server.login(sender, password)
#             server.send_message(msg)

#         print(f"🚨 Email alert sent successfully to {contact_email}")
#         return True

#     except smtplib.SMTPAuthenticationError as e:
#         print("❌ AUTH ERROR — Gmail blocked login!")
#         print("Details:", e)
#         return False

#     except smtplib.SMTPConnectError as e:
#         print("❌ CONNECTION ERROR — SMTP server unreachable!")
#         print("Details:", e)
#         return False

#     except Exception as e:
#         print("⚠️ UNKNOWN ERROR sending email:")
#         print(type(e), e)
#         return False


# FAST2SMS_API_KEY = "dU0kXw3sxcv1FLQbfmhPOVRJTyr2GAiWoaeHj69NzIBS7tCYgZn9kHNUL5DTxA71ZGsSQbOgudXioJwl"

# def send_sms_alert(phone, message):
#     url = "https://www.fast2sms.com/dev/bulkV2"

#     payload = {
#         "route": "q",
#         # "sender_id": "TXTIND",
#         "message": message,
#         "language": "english",
#         # "flash": 0,
#         "numbers": phone
#     }

#     headers = {
#         "authorization": FAST2SMS_API_KEY,
#         "Content-Type": "application/json"
#     }

#     response = requests.post(url, data=json.dumps(payload), headers=headers)

#     print("SMS API RESPONSE:", response.text)  # ← Helps debugging

#     return response.status_code == 200

# ===============================
# 🔹 Brevo Email Integration
# ===============================
# Use Streamlit secrets for API key & sender
def send_email_alert(contact_email, user_message, risk_label):
    BREVO_API_KEY = "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"
    BREVO_SENDER_EMAIL = "maya793@gmail.com"
    BREVO_SENDER_NAME = "AI Mental Health Companion"

    BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

    subject = f"[ALERT] Critical Risk Detected ({risk_label})"
    body = f"""
Dear Contact,

This is an automated alert from the Mental Health Chat Companion.
A high-risk message was detected:

"{user_message}"

Please reach out to the individual immediately.

— AI Companion System
"""

    payload = {
        "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
        "to": [{"email": contact_email}],
        "subject": subject,
        "htmlContent": body,
        "textContent": body
    }

    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            st.success(f"✅ Email alert sent successfully to {contact_email}")
            return True
        else:
            st.error(f"❌ Failed to send email. Status: {response.status_code}")
            st.write(response.text)
            return False
    except Exception as e:
        st.error(f"⚠️ Error sending email: {e}")
        return False



# ===============================
# ⚠️ Rule-Based Risk Check
# ===============================
def rule_based_risk_check(text):
    text = text.lower().strip()
    if any(kw in text for kw in ["hi", "hello", "hey", "good morning", "good evening", "how are you"]):
        return "LOW"
    if any(kw in text for kw in ["kill myself", "want to die", "end my life", "suicide", "die soon", "i'm going to die", "self harm", "cut myself"]):
        return "CRITICAL"
    if any(kw in text for kw in ["hopeless", "worthless", "can't go on", "overwhelmed", "panic", "breakdown", "anxiety attack", "no reason to live"]):
        return "HIGH"
    if any(kw in text for kw in ["sad", "lonely", "tired", "stressed", "anxious", "depressed", "upset", "worried"]):
        return "CONCERN"
    return "LOW"


# ===============================
# 💡 Suggestions & Recommendations
# ===============================
def suggest_activity(risk_label):
    suggestions = {
        "LOW": [
            "🌼 Write down 3 things you're grateful for today.",
            "📖 Read a page from a book you enjoy.",
            "🌤 Take a 5-minute walk and notice something beautiful.",
            "📝 Try journaling a happy memory.",
            "🎧 Listen to a favorite upbeat song for 2 minutes."
        ],
        "CONCERN": [
            "🕊 Try a 4-7-8 breathing exercise: inhale 4s, hold 7s, exhale 8s.",
            "💧 Drink a glass of water slowly and mindfully.",
            "🧘 Do 30 seconds of mindful breathing.",
            "🌱 Step outside for 2 minutes and take a deep breath.",
            "📘 Write one sentence about what you're feeling right now."
        ],
        "HIGH": [
            "💛 Talk to someone you trust. Sharing helps lighten the load.",
            "🚶‍♂️ Take a walk while focusing on your footsteps.",
            "🔄 Try grounding: name 5 things you can see, 4 you can touch, 3 you can hear.",
            "🤍 Sit somewhere quiet and breathe deeply for 60 seconds.",
            "☕ Make a warm drink and sit with it for a moment."
        ],
        "CRITICAL": [
            "🚨 Please reach out to a trusted friend or helpline immediately — you are not alone.",
            "📞 Call someone close to you right now. Please don’t stay alone with these feelings.",
            "🤝 You deserve help — contact a professional or helpline urgently.",
            "💬 Talk to a nearby friend, family member, or counselor right away.",
            "❤️ Please pause everything and contact a helpline or someone you trust immediately."
        ]
    }

    return random.choice(suggestions.get(risk_label, ["I'm here for you."]))

def get_time_based_suggestions():
    current_hour = datetime.now().hour
    if 6 <= current_hour < 18:
        return [
            "🌞 Take a short walk and enjoy sunlight.",
            "📝 Journal 3 things you're grateful for today.",
            "💧 Stay hydrated — drink a glass of water."
        ]
    else:
        return [
            "🌙 Dim the lights and relax before bed.",
            "🛌 Prepare for sleep — avoid screens for 20 minutes.",
            "🧘‍♀️ Try a 2-minute breathing or meditation exercise."
        ]

# ===============================
# 🌬 Breathing GIFs
# ===============================
def breathing_gif(exercise="Box Breathing"):
    if exercise == "Box Breathing":
        return "https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif"
    elif exercise == "4-7-8 Breathing":
        return "https://media.giphy.com/media/l4FGuhL4U2WyjdkaY/giphy.gif"
    return None

# ===============================
# 😴 Sleep & Meditation Tips
# ===============================
def get_sleep_meditation_tips():
    return [
        "🛏 Keep a regular sleep schedule.",
        "📵 Avoid screens 30 minutes before sleep.",
        "🌿 Try aromatherapy or calming music before bed.",
        "🧘‍♂️ Short meditation (5-10 mins) helps relax the mind."
    ]

# ===============================
# 🎨 Hobby-Based Suggestions
# ===============================
def hobby_based_tip(hobby):
    hobby = hobby.lower()
    if "music" in hobby:
        return "🎵 Listen to a new playlist or learn a new song."
    elif "reading" in hobby:
        return "📚 Read a short story or article."
    elif "art" in hobby:
        return "🎨 Sketch, paint, or color for 10 minutes."
    elif "gaming" in hobby:
        return "🎮 Try a relaxing or story-based game session."
    else:
        return f"🌟 Spend 15 minutes on your hobby: {hobby}"


def recommend_content(risk_label):
    recommendations = {
        "LOW": {
            "music": [
                "🎵 'Happy' – Pharrell Williams",
                "🎵 'Good Life' – OneRepublic",
                "🎵 'Best Day of My Life' – American Authors",
                "🎵 'Walking on Sunshine' – Katrina & The Waves"
            ],
            "movie": [
                "🎬 'Soul' (2020)",
                "🎬 'Paddington 2'",
                "🎬 'Luca' (2021)",
                "🎬 'The Secret Life of Walter Mitty'"
            ]
        },
        "CONCERN": {
            "music": [
                "🎵 'Let It Be' – The Beatles",
                "🎵 'Breathe Me' – Sia",
                "🎵 'Someone Like You' – Adele",
                "🎵 'Halo' – Beyoncé"
            ],
            "movie": [
                "🎬 'Inside Out'",
                "🎬 'Finding Nemo'",
                "🎬 'A Beautiful Day in the Neighborhood'",
                "🎬 'The Perks of Being a Wallflower'"
            ]
        },
        "HIGH": {
            "music": [
                "🎵 'Fix You' – Coldplay",
                "🎵 'The Night We Met' – Lord Huron",
                "🎵 'Stay' – Rihanna",
                "🎵 'Everybody Hurts' – R.E.M."
            ],
            "movie": [
                "🎬 'Good Will Hunting'",
                "🎬 'The Pursuit of Happyness'",
                "🎬 'The Theory of Everything'",
                "🎬 'Silver Linings Playbook'"
            ]
        },
        "CRITICAL": {
            "music": [
                "🎵 Soft piano music (YouTube)",
                "🎵 Ocean waves or rain sounds",
                "🎵 'Weightless' – Marconi Union (scientifically calming)"
            ],
            "movie": [
                "🆘 Avoid emotional movies — choose light nature videos instead.",
                "🆘 Avoid heavy content — watch short calming clips."
            ]
        }
    }

    rec = recommendations.get(risk_label, {"music": [], "movie": []})

    if risk_label == "CRITICAL":
        return "🚨 Avoid heavy media. Please reach out for help instead."

    # Pick a random movie and a random song
    music_title = random.choice(rec.get("music", ["None"]))
    movie_title = random.choice(rec.get("movie", ["None"]))

    # Build URLs
    def yt_link(title):
        if not title or title.lower() == "none":
            return None
        return f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+trailer"

    def sp_link(title):
        if not title or title.lower() == "none":
            return None
        return f"https://open.spotify.com/search/{title.replace(' ', '+')}"


    # return f"🎧 **Music:** {music}  \n🎬 **Movie:** {movie}"
    return {
        "music": music_title,
        "movie": movie_title,
        "music_url": sp_link(music_title),
        "movie_url": yt_link(movie_title)
    }



# ===============================
# 💾 Chat Logging
# ===============================
def log_chat(user, message, risk, bot_reply):
    log_entry = pd.DataFrame([[datetime.now(), user, message, risk, bot_reply]],
                             columns=["Time", "User", "Message", "RiskLevel", "BotReply"])
    if os.path.exists("chat_log.csv"):
        log_entry.to_csv("chat_log.csv", mode="a", header=False, index=False)
    else:
        log_entry.to_csv("chat_log.csv", index=False)


# ===============================
# 🧩 Multi-Page Navigation
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "contact"

def go_to_chat():
    if st.session_state.name and st.session_state.email and st.session_state.phone:
        st.session_state.page = "chat"
    else:
        st.warning("⚠️ Please fill all emergency contact details before continuing.")


# ===============================
# 🧠 PAGE 1 — Emergency Contact Setup
# ===============================
if st.session_state.page == "contact":
    st.title("🚨 Emergency Contact Setup")

    st.markdown("Please provide your trusted contact’s details. These will be used **only in emergencies.**")

    st.session_state.name = st.text_input("👤 Name")
    st.session_state.email = st.text_input("📧 E-mail")
    st.session_state.phone = st.text_input("📱 Phone Number")

    if st.button("✅ Save & Continue to Chat"):
        go_to_chat()


# ===============================
# 💬 PAGE 2 — Chat Companion
# ===============================
elif st.session_state.page == "chat":
    st.title("💬 Your AI Mental Health Companion")
    

    with st.sidebar:
        st.header("Emergency Contact")
        st.write(f"👤 **Name:** {st.session_state.name}")
        st.write(f"📧 **Email:** {st.session_state.email}")
        st.write(f"📱 **Phone:** {st.session_state.phone}")
        st.markdown("---")
        st.write("Session / User ID:")
        st.code(user_id)

        # --------------------------
        # 🎨 Hobby-Based Suggestions
        # --------------------------
        if "hobby" not in st.session_state:
            st.session_state.hobby = ""

        st.session_state.hobby = st.text_input("📝 What's your favorite hobby?", value=st.session_state.hobby)

        if st.session_state.hobby.strip():
            tip = hobby_based_tip(st.session_state.hobby)
            st.markdown(f"💡 Hobby Suggestion: {tip}")  

    # Load conversation from long-term memory (per-user JSON)
    memories = load_long_term_memory(user_id)

    msg = st.text_area("You:", key="user_input", height=100)
    send = st.button("Send")

    

    if send and msg.strip():
        # --- Save user message to long-term memory (replaces short-term memory) ---
        append_memory(user_id, "User", msg)

        # --- Risk detection (rule only for now) ---
        final_risk = rule_based_risk_check(msg)



        # # --- Generate AI reply (now returns 3 values) ---
        ai_reply, trailer, spotify = get_ai_reply(msg)

        # # --- Save ONLY the reply to long-term memory ---
        append_memory(user_id, "Companion", ai_reply)
        log_chat("User", msg, final_risk, ai_reply)


        # # # --- Show Recommendation Links separately ---
        rec = recommend_content(final_risk)
        st.markdown("### 🎬 / 🎧 Recommended Content")

        if isinstance(rec, str):
            rec = {"movie_url": None, "music_url": None, "movie": None, "music": None}

        if trailer:
            st.markdown(f"🎬 [Watch Trailer]({trailer})", unsafe_allow_html=True)
        elif rec.get("movie_url"):
            st.markdown(f"🎬 [Watch Trailer]({rec['movie_url']}) — {rec['movie']}", unsafe_allow_html=True)

        if spotify:
            st.markdown(f"🎵 [Listen on Spotify]({spotify})", unsafe_allow_html=True)
        elif rec.get("music_url"):
            st.markdown(f"🎵 [Listen on Spotify]({rec['music_url']}) — {rec['music']}", unsafe_allow_html=True)


        # --- Show Risk & Suggestions ---
        color_map = {"LOW": "#44bd32", "CONCERN": "#fbc531", "HIGH": "#e84118", "CRITICAL": "#c23616"}
        st.markdown(f"<div style='background:{color_map[final_risk]};padding:10px;border-radius:10px;color:white;'>🧠 Risk Level: <b>{final_risk}</b></div>", unsafe_allow_html=True)
        st.info(suggest_activity(final_risk))
        # st.success(recommend_content(final_risk))

        

        # --------------------------
        # ⏰ Time-Based Suggestions
        # --------------------------
        st.subheader("⏰ Time-based Tips")
        for tip in get_time_based_suggestions():
            st.write(f"- {tip}")

        # --------------------------
        # 🌬 Breathing GIFs
        # --------------------------
        st.subheader("🌬 Guided Breathing Exercise")
        exercise = st.selectbox("Choose a breathing exercise:", ["Box Breathing", "4-7-8 Breathing"])
        gif_url = breathing_gif(exercise)
        if gif_url:
            st.image(gif_url, width=400)

        # --------------------------
        # 😴 Sleep & Meditation Tips
        # --------------------------
        st.subheader("😴 Sleep & Meditation Tips")
        for tip in get_sleep_meditation_tips():
            st.write(f"- {tip}")
        st.markdown("[🎧 10-Minute Guided Meditation](https://www.youtube.com/watch?v=inpok4MKVLM)")

        # # --------------------------
        # # 🎨 Hobby-Based Suggestions
        # # --------------------------
        # if "hobby" not in st.session_state:
        #     st.session_state.hobby = ""

        # st.session_state.hobby = st.text_input("📝 What's your favorite hobby?", value=st.session_state.hobby)

        # if st.session_state.hobby:
        #     tip = hobby_based_tip(st.session_state.hobby)
        #     st.markdown(f"💡 Hobby Suggestion: {tip}")




        # --- Alerts ---
        if final_risk == "CRITICAL":
            st.error("🚨 Critical distress detected.")
            st.warning("📞 In India, you can contact AASRA (91-9820466726) or Helpline 9152987821.")

            # EMAIL ALERT BUTTON
            # EMAIL ALERT BUTTON (Brevo)
            if st.button("📧 Notify Emergency Contact"):
                if st.session_state.email:
                    success = send_email_alert(st.session_state.email, msg, final_risk)
                    if success:
                        st.success(f"✅ Alert sent successfully to {st.session_state.email}")
                    else:
                        st.error("❌ Failed to send alert. Check API key, sender email, or network.")
                else:
                    st.warning("⚠️ Please enter emergency contact email.")


            # SMS ALERT BUTTON
            # if st.button("📱 Send SMS Alert"):
            #     sms_message = f"URGENT: High-risk message detected:\n{msg}"
            #     success_sms = send_sms_alert(st.session_state.phone, sms_message)

            #     if success_sms:
            #         st.success("📩 SMS alert sent successfully!")
            #     else:
            #         st.error("❌ SMS sending failed! Check API key or phone number.")




    # --- Conversation Log ---
    st.markdown("---")
    st.subheader("💛 Conversation Log")
    # Re-load latest memories (reflects any new messages)
    memories = load_long_term_memory(user_id)
    # Show last 20 messages
    for entry in memories[-20:]:
        role = entry.get("role", "User")
        text = entry.get("message", "")
        time = entry.get("time", "")
        if role.lower().startswith("user"):
            st.markdown(f"<div style='text-align:right;background-color:#192a56;padding:10px;margin:5px;border-radius:10px;color:white;'>You: {text}<br><small style=\"color:#dcdde1\">{time}</small></div>", unsafe_allow_html=True)
        else:
            # Companion bubble (start)
            st.markdown(
                f"""
                <div style='text-align:left;background-color:#2f3640;padding:12px;margin:5px;border-radius:10px;color:#f5f6fa;'>
                    <b>Companion:</b><br>
                    {text}
                    <br><small style="color:#bdc3c7">{time}</small>
                </div>
                """,
                unsafe_allow_html=True
            )




    # Optional controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Conversation (JSON)"):
            fname = memory_filename(user_id)
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    data = f.read()
                st.download_button("Download memory JSON", data, file_name=fname, mime="application/json")
            else:
                st.warning("No conversation history found.")
    with col2:
        if st.button("Clear Long-Term Memory"):
            # Confirm: clear memory for this user only
            fname = memory_filename(user_id)
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                    st.success("Long-term memory cleared for this user.")
                except Exception as e:
                    st.error(f"Failed to clear memory: {e}")
            else:
                st.info("No memory file to delete.")
