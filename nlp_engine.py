from transformers import pipeline
import re

summarizer = pipeline("text2text-generation", model="google/flan-t5-base")


# =========================
# CLEAN TEXT (STRONG)
# =========================
def clean_text(text):
    # remove repeated phrases
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # remove filler words
    fillers = ["okay", "so", "uh", "um", "you know", "like"]
    for f in fillers:
        text = re.sub(rf'\b{f}\b', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# REMOVE REPETITIONS 
# =========================
def remove_repetition(text):
    sentences = re.split(r'[.!?]', text)
    seen = set()
    clean_sentences = []

    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue

        key = s.lower()
        if key not in seen:
            seen.add(key)
            clean_sentences.append(s)

    return ". ".join(clean_sentences)


# =========================
# KEYWORDS
# =========================
def extract_keywords(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()

    stopwords = {
        "the", "is", "and", "for", "that", "this",
        "with", "you", "have", "from", "they", "will"
    }

    freq = {}

    for w in words:
        if len(w) > 4 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    return sorted(freq, key=freq.get, reverse=True)[:10]


# =========================
# IMPORTANT SENTENCES
# =========================
def important_sentences(text, keywords):
    sentences = re.split(r'[.!?]', text)

    scored = []
    for s in sentences:
        s = s.strip()

        if len(s) < 30:
            continue

        score = sum(1 for k in keywords if k in s.lower())

        if "subscribe" in s.lower() or "channel" in s.lower():
            continue

        scored.append((score, s))

    scored.sort(reverse=True)

    return [s for _, s in scored[:5]]


# =========================
# SMART SUMMARIZATION (FIXED)
# =========================
def summarize(text, style):

    if len(text) < 50:
        return text

    text = clean_text(text)

    #  CHUNKING
    max_chunk = 250
    words = text.split()
    chunks = [" ".join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]

    summaries = []

    for chunk in chunks:

        if style == "quick":
            prompt = f"""
Create concise revision notes.

Rules:
- Bullet points
- Each point must be a COMPLETE sentence
- No repetition
- Max 5 points
- Easy for exam revision

Content:
{chunk}
"""

        else:
            prompt = f"""
Explain clearly like a teacher.

Rules:
- Start with 2-3 line explanation
- Then give bullet points
- Use simple language
- No repetition
- Do NOT copy text

Content:
{chunk}
"""

        result = summarizer(prompt, max_new_tokens=120, do_sample=False)
        summaries.append(result[0]["generated_text"])

    combined = " ".join(summaries)

    #  FINAL CLEAN SUMMARY
    if style == "quick":
        final_prompt = f"""
Rewrite into FINAL revision notes:

- Bullet points
- Max 5 points
- Each point complete sentence
- No repetition
- Very clear

Content:
{combined}
"""
    else:
        final_prompt = f"""
Create final student notes:

Format:
1. Short explanation (2 lines)
2. Key points (bullet points)
3. Simple explanation

Rules:
- No repetition
- Clear and structured
- Easy to understand

Content:
{combined}
"""

    final = summarizer(final_prompt, max_new_tokens=150, do_sample=False)

    # REMOVE REPETITION AFTER MODEL OUTPUT
    clean_output = remove_repetition(final[0]["generated_text"])

    return clean_output


# =========================
# PERSONALIZATION
# =========================
def personalize_summary(summary, important, style, memory):

    output = "\n📘 STUDENT NOTES\n"
    output += summary.strip() + "\n"

    if style == "detailed":
        output += "\n📖 EXTRA IMPORTANT POINTS:\n"
        for s in important:
            output += f"• {s.strip()}\n"

    output += "\n🧠 MEMORY PLAN:\n"

    if memory == "forget_fast":
        output += "• Revise within 24 hours\n"
        output += "• Repeat key concepts\n"
        output += "• Focus on understanding basics\n"

    elif memory == "average":
        output += "• Revise after 2 days\n"
        output += "• Practice active recall\n"
        output += "• Solve 2–3 questions\n"

    elif memory == "strong":
        output += "• Revise after 4–5 days\n"
        output += "• Apply concepts in problems\n"
        output += "• Teach someone else\n"

    return output