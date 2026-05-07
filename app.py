from flask import Flask, render_template, request
from transcript import get_transcript, get_transcript_with_time, sample_transcript
from nlp_engine import extract_keywords, important_sentences, summarize, personalize_summary
from memory_engine import next_revision
from qa_engine import get_text_by_time, answer_question
from datetime import datetime

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    answer = ""
    next_time = ""
    memory = "average"
    style = "quick"

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        style = request.form.get("style", "quick")
        memory = request.form.get("memory", "average")

        start = request.form.get("start")
        end = request.form.get("end")
        question = request.form.get("question", "").strip()

        try:
            # =========================
            # STEP 1: GET TRANSCRIPT
            # =========================
            full_text = get_transcript(url)

            # DEMO FALLBACK (VERY IMPORTANT)
            if not full_text or len(full_text.strip()) < 20:
                summary = "⚠️ Could not extract transcript. Using demo data."
                full_text = " ".join([t["text"] for t in sample_transcript])

            working_text = full_text
            transcript_with_time = None

            # =========================
            #  STEP 2: TIME FILTERING
            # =========================
            if start and end:
                try:
                    transcript_with_time = get_transcript_with_time(url)

                    segment = get_text_by_time(
                        transcript_with_time,
                        float(start),
                        float(end)
                    )

                    #  SAFE FALLBACK
                    if segment.strip():
                        working_text = segment
                    else:
                        summary = "⚠️ No content found in this time range. Showing full content instead."
                        working_text = full_text

                except Exception:
                    summary = "⚠️ Invalid time input. Showing full content instead."
                    working_text = full_text

            # =========================
            # STEP 3: LIMIT INPUT SIZE
            # =========================
            if working_text:
                max_words = 1500
                words = working_text.split()

                if len(words) > max_words:
                    working_text = " ".join(words[:max_words])

            # =========================
            #  STEP 4: NLP PIPELINE
            # =========================
            if working_text and len(working_text.strip()) > 30:
                keywords = extract_keywords(working_text)
                important = important_sentences(working_text, keywords)

                summary_text = summarize(working_text, style)
                summary = personalize_summary(summary_text, important, style, memory)

            else:
                summary += "\n⚠️ Not enough content to summarize."

            # =========================
            #  STEP 5: MEMORY SCHEDULING
            # =========================
            next_time = next_revision(datetime.now(), 2.0, memory)

            # =========================
            #  STEP 6: QA SYSTEM
            # =========================
            if question:
                if start and end and transcript_with_time:
                    context = get_text_by_time(
                        transcript_with_time,
                        float(start),
                        float(end)
                    )
                else:
                    context = working_text

                # QA SAFETY CHECK
                if not context.strip() or len(context.split()) < 20:
                    answer = "⚠️ Not enough context to answer clearly."
                else:
                    answer = answer_question(context, question)

        except Exception as e:
            summary = f"⚠️ Error: {str(e)}"

    return render_template(
        "index.html",
        summary=summary,
        answer=answer,
        next_time=next_time,
        memory=memory,
        style=style
    )


if __name__ == "__main__":
    app.run(debug=True)