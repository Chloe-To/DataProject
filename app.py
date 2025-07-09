import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datasets import Dataset
from datetime import datetime

REPO_RELEASE_ZIP_URL = "https://github.com/Chloe-To/DataProject/releases/download/v1.0.0/models.zip"
MODEL_DIR = "model"
LABELS = ['urgency', 'importance', 'tone', 'sentiment']

# --- DOWNLOAD & EXTRACT MODEL ---
def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading model from GitHub release...")
        response = requests.get(REPO_RELEASE_ZIP_URL)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("Model downloaded and extracted.")

# Load dataframe
df = pd.read_csv("/content/drive/MyDrive/Data Project/Code Trials/1000 Synthetic Data.csv")
df["text"] = df["subject"].fillna("") + " " + df["message"].fillna("")

train_df, test_df = train_test_split(df, test_size=0.2,random_state=21)

# Encode each label column
label_cols = ["urgency", "importance", "tone", "sentiment"]
label_encoders = {}
for col in label_cols:
    encoder = LabelEncoder()
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])  # Use same encoder
    label_encoders[col] = encoder

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding='max_length', max_length=256)
    for col in label_cols:
        tokens[col] = example[col]
    return tokens

train_ds = Dataset.from_pandas(train_df[["text"] + label_cols])
test_ds = Dataset.from_pandas(test_df[["text"] + label_cols])
train_ds = train_ds.map(tokenize)
test_ds = test_ds.map(tokenize)

# Train one head per label

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_model(label):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoders[label].classes_))

    args = TrainingArguments(
        output_dir=f"./models/{label}",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        #evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds.rename_column(label, "labels"),
        eval_dataset=test_ds.rename_column(label, "labels"),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(f"./models/{label}")
    tokenizer.save_pretrained(f"./models/{label}")

for label in label_cols:
    train_model(label)

# Save the trained model folder
#!cp -r ./models /content/drive/MyDrive/

def predict_labels(text):
    results = {}
    for label in label_cols:
        model = BertForSequenceClassification.from_pretrained(f"models/{label}")
        tokenizer = BertTokenizer.from_pretrained(f"models/{label}")
        model.eval()
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**tokens).logits
        pred = torch.argmax(logits, dim=1).item()
        results[label] = label_encoders[label].inverse_transform([pred])[0]
    return results

urgency_map = {"low": 0, "medium": 0.5, "high": 1.0}
importance_map = {"low": 0, "medium": 0.5, "high": 1.0}

def predict_weighted_scores(text):
    results = {}
    for label, label_map in zip(["urgency", "importance"], [urgency_map, importance_map]):
        model = BertForSequenceClassification.from_pretrained(f"models/{label}")
        tokenizer = BertTokenizer.from_pretrained(f"models/{label}")
        model.eval()
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**tokens).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        labels = label_encoders[label].classes_
        weighted_score = sum(label_map[lab] * probs[i] for i, lab in enumerate(labels))
        results[label] = round(weighted_score, 4)
    return results


t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def build_prompt(text, label_dict):
    lines = "\n".join([f"{k}: {v}" for k, v in label_dict.items()])
    return (
        f"You are a helpful assistant. Given the email below and its context, suggest what you would recommend the receiver of the email to do.\n\n"
        f"Email:\n{text}\n\n"
        f"Context:\n{lines}\n\n"
        f"Action:"
    )

def generate_action(text, labels, max_len=50):
    prompt = build_prompt(text, labels)
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = t5_model.generate(**inputs, max_length=max_len, temperature=0.8, top_p=0.9, do_sample=True)
    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def build_prompt(text, label_dict):
    lines = "\n".join([f"{k}: {v}" for k, v in label_dict.items()])
    return (
        f"""You are an assistant suggesting follow-up actions for workplace messages. Here are some examples:
1.
Message: Can you please update this by 3pm?
Action: Follow up within the hour


2.
Message: Just an FYI for next week’s meeting.
Action: No immediate action needed

Please generate an appropriate action for the following email and context:
\n\n"""
        f"Email:\n{text}\n\n"
        f"Context:\n{lines}\n\n"
        f"Action:"
    )

def generate_action(text, labels, max_len=50):
    prompt = build_prompt(text, labels)
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = t5_model.generate(**inputs, max_length=max_len, temperature=0.8, top_p=0.9, do_sample=True)
    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def build_bart_prompt(email_text, labels):
    context = ". ".join([f"The {k.lower()} is {v.lower()}" for k, v in labels.items()])
    return (
        f"You are a helpful assistant. Here's an email and its contextual tags.\n\n"
        f"Email content:\n\"{email_text}\"\n\n"
        f"Context: {context}.\n\n"
        f"What should the recipient do next? Respond in one short sentence."
    )


def generate_bart_action(email_text, labels, max_length=50):
    prompt = build_bart_prompt(email_text, labels)
    inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    output_ids = bart_model.generate(
        **inputs,
        max_length=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=2
    )

    decoded = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.strip()

model_name = "philschmid/bart-large-cnn-samsum"

bart_tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")

def build_bart_prompt(email_text, labels):
    context = ". ".join([f"The {k.lower()} is {v.lower()}" for k, v in labels.items()])
    return (
        f"You are an assistant that recommends a response action to emails.\n\n"
        f"Email: {email_text}\n\n"
        f"{context}.\n\n"
        f"Suggest an action the recipient of this email should do without recapping the contents and context of the email"
    )

def generate_bart_action(email_text, labels, max_length=50):
    prompt = build_bart_prompt(email_text, labels)
    inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    output_ids = bart_model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=2
    )

    decoded = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.strip()

#test_subject = "Reminder: Budget Review Needed"
#test_message = "Hi, please review the Q3 numbers and confirm approval before Friday."

test_subject = "Team Lunch Next Week"
test_message = "Hi Alex, Just wanted to touch base regarding team lunch next week. Let me know your thoughts. Best, Jamie"
email_text = test_subject + " " + test_message

predicted_labels = predict_labels(email_text)
recommended_action = generate_action(email_text, predicted_labels)

print("🔖 Predicted Labels:")
for k, v in predicted_labels.items():
    print(f"{k}: {v}")

print("\n🧭 Suggested Action:")
print(recommended_action)


test_subject = "Reminder: Budget Review Needed"
test_message = "Hi, please review the Q3 numbers and confirm approval before Friday."

#test_subject = "Team Lunch Next Week"
#test_message = "Hi Alex, Just wanted to touch base regarding team lunch next week. Let me know your thoughts. Best, Jamie"
email_text = test_subject + " " + test_message

predicted_labels = predict_labels(email_text)
weighted_scores = predict_weighted_scores(email_text)
recommended_action = generate_action(email_text, predicted_labels)

print("🔖 Predicted Labels:")
for k, v in predicted_labels.items():
    label = v['label'] if isinstance(v, dict) else v
    confidence = v.get('score') if isinstance(v, dict) else None
    if confidence:
        print(f"{k}: {label} (confidence: {confidence:.2f})")
    else:
        print(f"{k}: {label}")

print("\n📈 Weighted Scores:")
for k, score in weighted_scores.items():
    print(f"{k}: {score:.2f}")

print("\n🧭 Suggested Action:")
print(recommended_action)


email_text = "Hi Alex, Just wanted to touch base regarding team lunch next week. Let me know your thoughts. Best, Jamie"

predicted_labels = predict_labels(email_text)
action = generate_bart_action(email_text, predicted_labels)

# Output formatting
print("🔖 Predicted Labels:")
for k, v in predicted_labels.items():
    print(f"{k.lower()}: {v.lower()}")

print("\n🧭 Suggested Action:")
print(action.strip())


from sklearn.metrics import classification_report

def evaluate_model(label):
    model = BertForSequenceClassification.from_pretrained(f"./models/{label}")
    tokenizer = BertTokenizer.from_pretrained(f"./models/{label}")
    model.eval()

    preds, true = [], []
    for row in test_df.itertuples():
        tokens = tokenizer(row.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**tokens).logits
        pred = torch.argmax(logits).item()
        preds.append(pred)
        true.append(getattr(row, label))

    print(f"\n📊 {label} Classification Report:")
    print(classification_report(true, preds, target_names=label_encoders[label].classes_))

evaluate_model("urgency")

evaluate_model("importance")

evaluate_model("tone")

evaluate_model("sentiment")

# Count urgency values
urgency_counts = df["tone"].value_counts()

# Plot pie chart
plt.figure(figsize=(6, 6))
urgency_counts.plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Urgency Distribution")
plt.ylabel("")  # Hide y-axis label
plt.show()

#to dynamically track the metrics for plotting
precision_list=[]
accuracy_list=[]
f1_list=[]


#metric formulae:
precision = precision_score(Y_true, Y_pred)
recall = recall_score(Y_true, Y_pred)
f1 = f1_score(Y_true, Y_pred)
auc = roc_auc_score(y_true, y_pred_proba) # maybe won't need

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, precision_scores, label='Precision', marker='o')
plt.plot(time_steps, recall_scores, label='Recall', marker='o')
plt.plot(time_steps, f1_scores, label='F1 Score', marker='o')
plt.plot(time_steps, roc_auc_scores, label='ROC AUC', marker='o')

# Labels and styling
plt.title('Evaluation Metrics Over Time')
plt.xlabel('Time / Iteration / Model Version')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# IMPORTS AND SETUP

import re
import datetime
import random
import pandas as pd
import dateparser
import streamlit as st
import os
import uuid
from dateparser.search import search_dates
from pathlib import Path
import json
import plotly.express as px


#JSON TASK STORAGE

def load_tasks():
    try:
        if Path("tasks.json").exists():
            with open("tasks.json", "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
    except json.JSONDecodeError:
        st.warning("⚠️ The task file is corrupted or unreadable. Starting with an empty task list.")
        return []
    return []

def save_tasks(tasks):
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)


# WEEKDAY DICTIONARY

WEEKDAY_CORRECTIONS = {
    # Abbreviations
    "mon": "monday",
    "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "weds": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday",
    "fri": "friday", "friyay": "friday",
    "sat": "saturday",
    "sun": "sunday",

    # Common misspellings
    "wensday": "wednesday",
    "thirsday": "thursday",
    "fryday": "friday",
    "saterday": "saturday",
    "sundy": "sunday",
    "mondy": "monday",
    "tusday": "tuesday",
    "wednsday": "wednesday"
}

def correct_weekdays(text):
    for wrong, right in WEEKDAY_CORRECTIONS.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)
    return text


# DEADLINE EXTRACTION

def extract_deadline_from_message(message, reference_date):
    corrected_message = correct_weekdays(message)

    # Step 0: Handle vague phrases like "before the weekend"
    if re.search(r'\bbefore\s+the\s+weekend\b', corrected_message, re.IGNORECASE):
        days_until_friday = (4 - reference_date.weekday()) % 7
        return reference_date + datetime.timedelta(days=days_until_friday)

    # Step 1: Check for ISO format (YYYY-MM-DD)
    iso_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', corrected_message)
    if iso_match:
        try:
            iso_date = datetime.datetime.strptime(iso_match.group(1), "%Y-%m-%d")
            if iso_date > reference_date:
                return iso_date
        except:
            pass

    # Step 2: Handle "next week"
    if re.search(r'\bnext\s+week\b', corrected_message, re.IGNORECASE):
        days_until_next_monday = (7 - reference_date.weekday()) % 7 + 7
        return reference_date + datetime.timedelta(days=days_until_next_monday)

    # Step 3: Handle "next month"
    if re.search(r'\bnext\s+month\b', corrected_message, re.IGNORECASE):
        year = reference_date.year
        month = reference_date.month + 1
        if month > 12:
            month = 1
            year += 1
        return datetime.datetime(year, month, 1)

    # Step 4: Handle "next [weekday]"
    weekday_match = re.search(r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', corrected_message, re.IGNORECASE)
    if weekday_match:
        weekday_str = weekday_match.group(1).lower()
        weekday_index = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].index(weekday_str)
        days_ahead = (weekday_index - reference_date.weekday() + 7) % 7 + 7
        return reference_date + datetime.timedelta(days=days_ahead)

    # Step 5: Handle "this [weekday]"
    this_weekday_match = re.search(r'\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', corrected_message, re.IGNORECASE)
    if this_weekday_match:
        weekday_str = this_weekday_match.group(1).lower()
        weekday_index = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].index(weekday_str)
        days_ahead = (weekday_index - reference_date.weekday()) % 7
        return reference_date + datetime.timedelta(days=days_ahead)

    # Handle "this week"
    if re.search(r'\bthis\s+week\b', corrected_message, re.IGNORECASE):
        # Return the end of the current week (Sunday)
        days_until_sunday = 6 - reference_date.weekday()
        return reference_date + datetime.timedelta(days=days_until_sunday)

    # Handle "this month"
    if re.search(r'\bthis\s+month\b', corrected_message, re.IGNORECASE):
        # Return the last day of the current month
        next_month = reference_date.replace(day=28) + datetime.timedelta(days=4)  # always goes to next month
        last_day = next_month - datetime.timedelta(days=next_month.day)
        return last_day


    # Step 6: Regex-based date phrase matching
    deadline_phrases = re.findall(
        r'\b(?:by|for|on|due)?\s*('
        r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|'             # 08/07 or 08/07/2025
        r'\d{4}-\d{2}-\d{2}|'                              # 2025-07-08
        r'\d{1,2}(?:st|nd|rd|th)?\s+of\s+\w+|'             # 8th of July
        r'\d{1,2}(?:st|nd|rd|th)?\s+\w+|'                  # 8th July
        r'\w+\s+\d{1,2}(?:st|nd|rd|th)?|'                  # July 8th
        r'\w+\s+\d{1,2},?\s*\d{4}|'                        # July 8, 2025
        r'tomorrow|today|'
        r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        r')\b',
        corrected_message,
        re.IGNORECASE
    )

    for phrase in deadline_phrases:
        try:
            parsed = dateparser.parse(
                phrase,
                settings={
                    'RELATIVE_BASE': reference_date,
                    'PREFER_DATES_FROM': 'future',
                    'DATE_ORDER': 'DMY'
                }
            )
            if parsed and parsed > reference_date:
                return parsed
        except:
            pass

    # Step 7: Use search_dates as fallback
    found_dates = search_dates(
        corrected_message,
        settings={
            'RELATIVE_BASE': reference_date,
            'PREFER_DATES_FROM': 'future',
            'DATE_ORDER': 'DMY'
        }
    )

    if found_dates:
        false_positives = {"to", "on", "at", "in", "by", "are"}
        filtered_dates = [
            (text, dt) for text, dt in found_dates
            if dt > reference_date and (
                any(char.isdigit() for char in text) or text.strip().lower() not in false_positives
            )
        ]
        if filtered_dates:
            return filtered_dates[0][1]

    return None


#LLM ADDITION
# Simulated fine-tuned BERT model output
def simulate_llm_scores(message):
    urgency_score = random.uniform(0.3, 1.0)
    importance_score = random.uniform(0.3, 1.0)
    return urgency_score, importance_score

# RULE-BASED URGENCY SCORE - DEADLINE PROXIMITY
def rule_based_urgency(message_date, deadline=None):
    if deadline:
        days_diff = (deadline - message_date).days
        if days_diff < 0:
            return 0.0
        elif days_diff <= 1:
            return 1.0
        elif days_diff <= 3:
            return 0.85
        elif days_diff <= 7:
            return 0.5
        else:
            return 0.25
    return 0.0

# RULE-BASED KEYWORDS - URGENCY FLAGS
def rule_based_flags(message):
    keywords = ['urgent', 'asap', 'immediately', 'critical', 'important']
    score = 0.0
    for word in keywords:
        if re.search(rf'\b{word}\b', message, re.IGNORECASE):
            score += 0.3
    return min(score, 1.0)

# COMBINE RULE-BASED AND LLM SCORES
def combine_scores(
    urgency_rule_score,
    urgency_flag_score,
    urgency_llm_score,#llm
    importance_llm_score,#llm
    weight_rule=0.4,
    weight_llm=0.6
):
    rule_total = min(urgency_rule_score + urgency_flag_score, 1.0)
    total_weight = weight_rule + weight_llm
    normalized_rule_weight = weight_rule / total_weight
    normalized_llm_weight = weight_llm / total_weight

    combined_urgency = normalized_rule_weight * rule_total + normalized_llm_weight * urgency_llm_score
    return min(combined_urgency, 1.0), importance_llm_score


# RESPONSES - HARD CODED + LLM ACTION (e.g. This has been added to your Escalation Tab. ("LLM action"))
def generate_response(urgency, importance, escalate):
    if escalate:
        return "🚨 This message appears to be both urgent and important. Recommended action: escalate to your project lead or take immediate steps to address the issue."
    elif urgency > 0.5 or importance > 0.5:
        return "⚠️ This message has moderate urgency or importance. You may want to review it soon and follow up if needed."
    else:
        return "✅ This message does not require immediate attention. You can monitor it for now."

# ANALYSIS AND ESCALATION LOGIC
def analyze_message(message, message_date):
    deadline = extract_deadline_from_message(message, message_date)
    urgency_rule_score = rule_based_urgency(message_date, deadline)
    urgency_flag_score = rule_based_flags(message)
    urgency_llm_score, importance_llm_score = simulate_llm_scores(message)

    final_urgency, final_importance = combine_scores(
        urgency_rule_score,
        urgency_flag_score,
        urgency_llm_score,
        importance_llm_score
    )

    escalate = final_urgency > 0.7 and final_importance > 0.7
    response = generate_response(final_urgency, final_importance, escalate)

    return {
        "id": str(uuid.uuid4()),
        "user": st.session_state.username,
        "date_sent": message_date.strftime("%Y-%m-%d"),  # Convert to string for JSON
        "message": message,
        "deadline": deadline.strftime("%Y-%m-%d") if deadline else None,  # Convert to string if exists
        "project": "",
        "action": "",#llm
        "status": "Not Started",
        "escalate": escalate,
        "response": response
    }


def analyze_message(message, message_date):
    deadline = extract_deadline_from_message(message, message_date)
    urgency_rule_score = rule_based_urgency(message_date, deadline)
    urgency_flag_score = rule_based_flags(message)
    model = load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_auth_token=HF_TOKEN)
    predicted_labels = predict(message, model, tokenizer)

    urgency_label = predicted_labels.get("urgency")
    importance_label = predicted_labels.get("importance")
    tone_label = predicted_labels.get("tone")
    sentiment_label = predicted_labels.get("sentiment")


    final_urgency, final_importance = combine_scores(
        urgency_rule_score,
        urgency_flag_score,
        urgency_llm_score,
        importance_llm_score
    )

    escalate = final_urgency > 0.7 and final_importance > 0.7
    response = generate_response(final_urgency, final_importance, escalate)

    return {
        "id": str(uuid.uuid4()),
        "user": st.session_state.username,
        "date_sent": message_date.strftime("%Y-%m-%d"),
        "message": message,
        "deadline": deadline.strftime("%Y-%m-%d") if deadline else None,
        "project": "",
        "action": "",
        "status": "Not Started",
        "escalate": escalate,
        "response": response,
        "predicted_labels": {
            "urgency": urgency_label,
            "importance": importance_label,
            "tone": tone_label,
            "sentiment": sentiment_label
        }
    }

# USER DATABASE
USERS = {
    "zana": {"password": "12345678", "role": "team_member"},
    "delphine": {"password": "12345678", "role": "team_member"},
    "beatrice": {"password": "12345678", "role": "manager"}
}

# LOGIN FUNCTION
def login():
    st.title("🔐 Cognizant Message Analyzer Login")

    username = st.text_input("Username").lower()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.success(f"Welcome, {username.capitalize()}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# CHECK LOGIN STATE
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""

if not st.session_state.logged_in:
    login()
    st.stop()


#-------TAB 1---------
def urgency_calculator_tab():
    st.header("📊 Urgency Calculator")

    st.markdown(f"### 👋 Hello, {st.session_state.username.capitalize()}!")
    st.markdown("Paste your message below and we'll analyze it for urgency.")

    full_message_input = st.text_area("Paste the full message here including any deadlines:")
    message_date_input = st.date_input("Date the message was sent", datetime.date.today(), format="DD/MM/YYYY")
    project_name_input = st.text_input("Enter project name:")
    analyze_button = st.button("Analyze")

    if analyze_button and full_message_input:
        message_date = datetime.datetime.combine(message_date_input, datetime.datetime.min.time())
        result = analyze_message(full_message_input, message_date)
        result["project"] = project_name_input.strip() if project_name_input else ""
        result["user"] = st.session_state.username

        st.markdown(f"**Response:** {result['response']}")

    # Show predicted labels
        preds = result["predicted_labels"]
        st.subheader("🔍 Predicted Labels")
        st.markdown(f"- **Urgency:** `{preds['urgency']}`")
        st.markdown(f"- **Importance:** `{preds['importance']}`")
        st.markdown(f"- **Tone:** `{preds['tone']}`")
        st.markdown(f"- **Sentiment:** `{preds['sentiment']}`")

        if result["escalate"]:
            tasks = load_tasks()
            tasks.append(result)
            save_tasks(tasks)
            st.success("✅ Task added to dashboard.")



#------TAB 2-------
def dashboard_tab():
    st.header("📋 Escalated Tasks Dashboard")

    tasks = load_tasks()
    user_tasks = [task for task in tasks if task["user"] == st.session_state.username]

    if not user_tasks:
        st.info("No escalated tasks yet.")
        return

    df = pd.DataFrame(user_tasks)
    df["Date of Message"] = pd.to_datetime(df["date_sent"], errors="coerce").dt.strftime("%d/%m/%Y")
    df["Deadline"] = pd.to_datetime(df["deadline"], errors="coerce").dt.strftime("%d/%m/%Y")

    def status_emoji(status):
        return {
            "Not Started": "🔴 Not Started",
            "In Progress": "🟡 In Progress",
            "Completed": "🟢 Completed"
        }.get(status, status)

    df["Status"] = df["status"].apply(status_emoji)
    df["Project"] = df["project"]
    df["Action"] = df["action"]
    df["Message"] = df["message"]
    df["Select"] = False

    project_options_raw = df["Project"].dropna().str.title().unique().tolist()
    selected_projects = st.multiselect("Filter by project:", options=sorted(project_options_raw), key="dashboard_project_filter")
    if selected_projects:
        df = df[df["Project"].str.title().isin(selected_projects)]

    status_filter = st.selectbox("Filter by status:", options=["All", "Not Started", "In Progress", "Completed"])
    if status_filter != "All":
        df = df[df["Status"].str.contains(status_filter, case=False)]

    select_all = st.checkbox("✅ Select All")
    id_map = df["id"].tolist()
    df_editor = df[["Date of Message", "Message", "Project", "Action", "Deadline", "Status", "Select"]].copy()

    if select_all:
        df_editor["Select"] = True

    edited_df = st.data_editor(
        df_editor,
        use_container_width=True,
        column_config={
            "Status": st.column_config.SelectboxColumn("Status", options=["🔴 Not Started", "🟡 In Progress", "🟢 Completed"]),
            "Project": st.column_config.TextColumn("Project"),
            "Action": st.column_config.TextColumn("Action"),
            "Deadline": st.column_config.TextColumn("Deadline"),
            "Select": st.column_config.CheckboxColumn("Select")
        },
        disabled=["Date of Message", "Message"],
        hide_index=True,
        key="dashboard_editor"
    )

    for i in range(len(edited_df)):
        row = edited_df.iloc[i]
        if i >= len(id_map):
            continue
        task_id = id_map[i]
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = row["Status"].split(" ", 1)[-1]
                task["project"] = row["Project"]
                task["action"] = row["Action"]
                try:
                    task["deadline"] = datetime.datetime.strptime(row["Deadline"], "%d/%m/%Y").strftime("%Y-%m-%d")
                except:
                    task["deadline"] = row["Deadline"]

    save_tasks(tasks)

    if st.button("🗑️ Delete Selected"):
        selected_ids = [id_map[i] for i in range(len(edited_df)) if edited_df.iloc[i]["Select"] and i < len(id_map)]
        tasks = [task for task in tasks if task["id"] not in selected_ids]
        save_tasks(tasks)
        st.success("Selected tasks deleted.")
        st.rerun()


#-------TAB 3---------
def progress_insights_tab():
    st.header("📈 Progress Insights")

    tasks = load_tasks()
    user_tasks = [task for task in tasks if task["user"] == st.session_state.username]

    if not user_tasks:
        st.info("No tasks available.")
        return

    completed_count = sum(1 for task in user_tasks if task["status"].lower() == "completed")
    st.metric("✅ Completed Tasks", completed_count)

    leaderboard_data = {}
    for task in tasks:
        user = task.get("user", "Unknown")
        if task["status"].lower() == "completed":
            leaderboard_data[user] = leaderboard_data.get(user, 0) + 1
    leaderboard_df = pd.DataFrame(list(leaderboard_data.items()), columns=["User", "Completed Tasks"])
    leaderboard_df["User"] = leaderboard_df["User"].str.capitalize()
    leaderboard_df = leaderboard_df.sort_values(by="Completed Tasks", ascending=False).reset_index(drop=True)
    leaderboard_df.index = leaderboard_df.index + 1
    st.subheader("🏆 Leaderboard")
    st.table(leaderboard_df)

    st.subheader("📊 Task Status Distribution")
    project_options = sorted(set(task["project"].title() for task in user_tasks if task["project"]))
    selected_projects = st.multiselect("Filter by project:", options=project_options, key="progress_project_filter")

    period = st.selectbox("Filter by time period:", ["All", "This Week", "Last 2 Weeks", "This Month"], key="progress_period_filter")
    filtered_tasks = user_tasks

    if selected_projects:
        filtered_tasks = [task for task in filtered_tasks if task["project"].title() in selected_projects]

    today = datetime.date.today()
    if period == "This Week":
        start = today - datetime.timedelta(days=today.weekday())
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]
    elif period == "Last 2 Weeks":
        start = today - datetime.timedelta(days=14)
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]
    elif period == "This Month":
        start = today.replace(day=1)
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]

    if filtered_tasks:
        status_counts = pd.Series([task["status"] for task in filtered_tasks]).value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts,
            names="Status",
            values="Count",
            title="Task Status Distribution",
            hole=0.4,
            color="Status",
            color_discrete_map={
                "Not Started": "#f8d7da",
                "In Progress": "#fff3cd",
                "Completed": "#d4edda"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No tasks match the selected filters for the doughnut chart.")

#-------TAB 4---------
def team_dashboard_tab():
    st.header("👥 Team Dashboard")

    tasks = load_tasks()
    if not tasks:
        st.info("No tasks available.")
        return

    team_members = sorted(set(task["user"] for task in tasks if USERS.get(task["user"], {}).get("role") == "team_member"))
    summary = []
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())

    for member in team_members:
        member_tasks = [task for task in tasks if task["user"] == member]
        this_week = [
            task for task in member_tasks
            if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start_of_week
        ]
        not_started = sum(1 for task in this_week if task["status"].lower() == "not started")
        completed = sum(1 for task in this_week if task["status"].lower() == "completed")
        total = len(this_week)

        if total == 0:
            progress = "No Tasks This Week"
        elif not_started > total / 2:
            progress = "Falling Behind"
        elif completed / total >= 0.7:
            progress = "Efficient"
        else:
            progress = "On Track"

        recommendation = {
            "Falling Behind": "Offer support",
            "On Track": "Monitor progress",
            "Efficient": "Note good performance",
            "No Tasks This Week": "Check in for updates"
        }.get(progress, "")

        summary.append({
            "Team Member": member.capitalize(),
            "Progress": progress,
            "Recommendation": recommendation
        })

    summary_df = pd.DataFrame(summary)
    summary_df.reset_index(drop=True, inplace=True)
    summary_df.index = summary_df.index + 1

    def highlight_progress(row):
        color_map = {
            "Falling Behind": "background-color: #f8d7da",
            "On Track": "background-color: #d1ecf1",
            "Efficient": "background-color: #d4edda",
            "No Tasks This Week": "background-color: #fefefe"
        }
        return [color_map.get(row["Progress"], "")] * len(row)

    st.subheader("📋 Team Progress Overview")
    st.dataframe(summary_df.style.apply(highlight_progress, axis=1), use_container_width=True)

    st.subheader("📊 Task Status Distribution by Team Member")
    project_options = sorted(set(task["project"].title() for task in tasks if task["project"]))
    selected_projects = st.multiselect("Filter by project:", options=project_options, key="team_project_filter")

    period = st.selectbox("Filter by time period:", ["All", "This Week", "Last 2 Weeks", "This Month"], key="team_period_filter")
    filtered_tasks = [task for task in tasks if task["user"] in team_members]

    if selected_projects:
        filtered_tasks = [task for task in filtered_tasks if task["project"].title() in selected_projects]

    if period == "This Week":
        start = today - datetime.timedelta(days=today.weekday())
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]
    elif period == "Last 2 Weeks":
        start = today - datetime.timedelta(days=14)
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]
    elif period == "This Month":
        start = today.replace(day=1)
        filtered_tasks = [task for task in filtered_tasks if "date_sent" in task and datetime.datetime.strptime(task["date_sent"], "%Y-%m-%d").date() >= start]

    if filtered_tasks:
        df = pd.DataFrame(filtered_tasks)
        df["user"] = df["user"].str.capitalize()
        fig = px.histogram(
            df,
            x="user",
            color="status",
            barmode="stack",
            title="Task Status by Team Member",
            labels={"user": "Team Member", "status": "Task Status"},
            color_discrete_map={
                "Not Started": "#f8d7da",
                "In Progress": "#fff3cd",
                "Completed": "#d4edda"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No tasks match the selected filters for the chart.")


# UI
# Add a banner and title
st.markdown("""
    <div style="background-color:#003366;padding:10px;border-radius:5px">
        <h1 style="color:white;text-align:center;">📬 Cognizant Message Analyzer</h1>
    </div>
""", unsafe_allow_html=True)

# Add logout button to sidebar
with st.sidebar:
    st.markdown("### 🔐 Session")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

# UI WITH USER LOGIC
# Display tabs based on role
if st.session_state.role == "team_member":
    tab1, tab2, tab3 = st.tabs(["📊 Urgency Calculator", "📋 Dashboard", "📈 Progress Insights"])

    with tab1:
        urgency_calculator_tab()

    with tab2:
        dashboard_tab()

    with tab3:
        progress_insights_tab()

elif st.session_state.role == "manager":
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Urgency Calculator",
        "📋 Dashboard",
        "📈 Progress Insights",
        "👥 Team Dashboard"
    ])

    with tab1:
        urgency_calculator_tab()

    with tab2:
        dashboard_tab()

    with tab3:
        progress_insights_tab()

    with tab4:
        team_dashboard_tab()

