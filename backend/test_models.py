from transformers import pipeline

models = [
    "vikram71198/distilroberta-base-finetuned-fake-news-detection",
    "jy46604790/Fake-News-Bert-Detect",
    "DunnBC22/roberta-base-Fake_News_Detector"
]

text = "india announced lockdown amid iran usa war"

for m in models:
    print(f"\n--- Model: {m} ---")
    try:
        classifier = pipeline("text-classification", model=m)
        res = classifier(text)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)
