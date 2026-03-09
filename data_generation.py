import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def generate_data(num_users=1000, num_items=200, num_sessions=5000):
    
    users = pd.DataFrame({
        "user_id": range(num_users),
        "budget_segment": np.random.choice(["budget", "premium"], num_users),
        "veg_preference": np.random.choice([0,1], num_users)
    })

    items = pd.DataFrame({
        "item_id": range(num_items),
        "category": np.random.choice(["main", "drink", "dessert", "side"], num_items),
        "price": np.random.randint(50, 500, num_items),
        "veg_flag": np.random.choice([0,1], num_items)
    })

    sessions = []

    for _ in range(num_sessions):
        user = random.randint(0, num_users-1)
        hour = random.randint(8, 23)

        cart_size = random.randint(1,3)
        cart_items = random.sample(range(num_items), cart_size)

        for item in cart_items:
            sessions.append({
                "user_id": user,
                "item_id": item,
                "hour": hour,
                "added": 1
            })

    interactions = pd.DataFrame(sessions)

    return users, items, interactions


if __name__ == "__main__":
    users, items, interactions = generate_data()
    users.to_csv("users.csv", index=False)
    items.to_csv("items.csv", index=False)
    interactions.to_csv("interactions.csv", index=False)