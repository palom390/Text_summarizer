from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Load model and tokenizer directly (bypasses pipeline task registry)
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def to_bullet_points(summary, n=3):
    """Split a summary string into n bullet points by sentence."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= n:
        # Pick n evenly-spaced sentences
        step = len(sentences) / n
        picked = [sentences[int(i * step)] for i in range(n)]
    else:
        # Fewer sentences than requested — use all, pad if needed
        picked = sentences
        while len(picked) < n:
            picked.append(picked[-1])  # repeat last sentence

    return "\n".join(f"• {s}" for s in picked)


if __name__ == "__main__":
    article = """
    Paste your long article here...
    """

    result = summarize_text(article)
    print("\nSummary:\n")
    print(result)

    print("\nKey Points:\n")
    print(to_bullet_points(result))
