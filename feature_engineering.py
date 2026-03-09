import pandas as pd

def create_features(users, items, interactions):

    data = interactions.merge(users, on="user_id")
    data = data.merge(items, on="item_id")

    data["is_peak"] = data["hour"].apply(lambda x: 1 if 12<=x<=14 or 19<=x<=22 else 0)

    data["high_price"] = data["price"].apply(lambda x: 1 if x>300 else 0)

    return data