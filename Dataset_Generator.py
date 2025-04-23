import pandas as pd
import random
from faker import Faker

# Initialize Faker and random
fake = Faker()
random.seed(42)

# Labels
labels = ['FAKE', 'REAL']

# Fake keywords and real keywords
fake_keywords = [
    "aliens", "hoax", "flat earth", "secret cure", "mind control",
    "miracle drug", "celebrity scandal", "government cover-up", "immortal jellyfish"
]

real_keywords = [
    "election", "policy", "research", "education", "budget report",
    "economic growth", "climate change", "public health", "technology breakthrough"
]

# Function to create a fake or real article
def generate_article(article_id, label):
    if label == "FAKE":
        keyword = random.choice(fake_keywords)
        title = f"{keyword.title()} Confirmed by Sources"
        text = f"{fake.paragraph()} This just in — shocking claims about {keyword} surface on the internet."
    else:
        keyword = random.choice(real_keywords)
        title = f"New Report on {keyword.title()}"
        text = f"{fake.paragraph()} Experts discuss the ongoing developments in {keyword}."

    return {
        "id": article_id,
        "title": title,
        "text": text,
        "label": label
    }

# Generate 500 rows (250 FAKE, 250 REAL)
data = [generate_article(f"F{str(i).zfill(3)}", "FAKE") for i in range(250)] + \
       [generate_article(f"R{str(i).zfill(3)}", "REAL") for i in range(250)]
random.shuffle(data)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("fake_news_sample.csv", index=False)

print("✅ Dataset generated and saved as fake_news_sample.csv")
