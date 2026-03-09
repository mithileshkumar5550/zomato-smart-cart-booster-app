import pyttsx3
import pandas as pd
from model_training import train_model
from inference import recommend

model = train_model()

users = pd.read_csv("users.csv")
items = pd.read_csv("items.csv")

user = users.iloc[0]

top_items = recommend(model, user, items, hour=20)

print("Top 10 Recommendations:")
print(top_items)