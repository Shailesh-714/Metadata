from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_events(events):
    text = " ".join(events)
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Example usage
events = ['Person near a car', 'Person riding a bicycle', 'Car parked']
summary = summarize_events(events)
print(summary)
