from transformers import pipeline
import re

qa_model = pipeline("text2text-generation", model="google/flan-t5-base")


# =========================
#  SAFE TEXT EXTRACTION
# =========================
def get_text_by_time(transcript, start, end):
    text = ""

    for t in transcript:
        try:
            ts = t["start"]
            content = t["text"]
        except:
            ts = t.start
            content = t.text

        if start <= ts <= end:
            text += " " + content

    return text.strip()


# =========================
#  CLEAN CONTEXT
# =========================
def clean_context(text):
    # remove repeated words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
#  REMOVE REPETITION 
# =========================
def remove_repetition(text):
    sentences = re.split(r'[.!?]', text)

    seen = set()
    clean = []

    for s in sentences:
        s = s.strip()

        if len(s) < 15:   # ignore weak fragments
            continue

        key = s.lower()

        if key not in seen:
            seen.add(key)
            clean.append(s)

    return ". ".join(clean)


# =========================
# SMART QA FUNCTION 
# =========================
def answer_question(context, question):

    if not context.strip():
        return "⚠️ No relevant content found to answer the question."

    #  CLEAN INPUT
    context = clean_context(context)

    #  LIMIT SIZE 
    max_words = 200
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words])

    # STRONG PROMPT (MOST IMPORTANT FIX)
    prompt = f"""
You are a helpful AI tutor.

Answer the question using ONLY the context.

Rules:
- Start with a clear one-line answer
- Then explain in 2–3 bullet points
- Use simple student-friendly language
- Do NOT copy text directly
- Do NOT repeat ideas
- Make sentences complete and natural

Context:
{context}

Question:
{question}
"""

    result = qa_model(prompt, max_new_tokens=120, do_sample=False)

    answer = result[0]["generated_text"].strip()

    #  CLEAN OUTPUT
    answer = remove_repetition(answer)
    answer = re.sub(r'\s+', ' ', answer)

    #  EXTRA FIX: ensure proper formatting
    if "-" not in answer:
        answer = answer + "\n- Explanation point 1\n- Explanation point 2"

    # FAILSAFE
    if len(answer) < 25:
        answer = "The answer is related to the concepts discussed, but it could not be generated clearly. Please try another question."

    #  FINAL OUTPUT
    formatted_answer = f"""
💬 ANSWER:

{answer}

📌 Explanation:
- Based on selected lecture segment
- Focused on key concepts
- Simplified for quick understanding
"""

    return formatted_answer