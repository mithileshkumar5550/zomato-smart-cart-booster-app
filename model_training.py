# from sklearn.metrics import roc_auc_score
# import pandas as pd
# import numpy as np
# import random

# def generate_data(num_users=1000, num_items=200, num_sessions=5000):

#     users = pd.DataFrame({
#         "user_id": range(num_users),
#         "budget_segment": np.random.choice(["budget", "premium"], num_users),
#         "veg_preference": np.random.choice([0,1], num_users)
#     })

#     items = pd.DataFrame({
#         "item_id": range(num_items),
#         "category": np.random.choice(["main", "drink", "dessert", "side"], num_items),
#         "price": np.random.randint(50, 500, num_items),
#         "veg_flag": np.random.choice([0,1], num_items)
#     })

#     interactions = []

#     for _ in range(num_sessions):

#         user = random.randint(0, num_users-1)
#         hour = random.randint(8, 23)

#         # Positive items (user added)
#         added_items = random.sample(range(num_items), 2)

#         for item in added_items:
#             interactions.append({
#                 "user_id": user,
#                 "item_id": item,
#                 "hour": hour,
#                 "added": 1
#             })

#         # Negative items (user did NOT add)
#         not_added_items = random.sample(range(num_items), 2)

#         for item in not_added_items:
#             interactions.append({
#                 "user_id": user,
#                 "item_id": item,
#                 "hour": hour,
#                 "added": 0
#             })

#     interactions = pd.DataFrame(interactions)

#     return users, items, interactions


# if __name__ == "__main__":
#     users, items, interactions = generate_data()
#     users.to_csv("users.csv", index=False)
#     items.to_csv("items.csv", index=False)
#     interactions.to_csv("interactions.csv", index=False)



# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# import numpy as np
# import random
# def precision_at_k(y_true, y_scores, k=10):

#     # Combine true labels and predicted scores
#     data = list(zip(y_true, y_scores))

#     # Sort by predicted score descending
#     data.sort(key=lambda x: x[1], reverse=True)

#     # Take top K
#     top_k = data[:k]

#     # Count relevant items
#     relevant = sum([1 for x in top_k if x[0] == 1])

#     return relevant / k



# # -----------------------------
# # Step 1: Generate Data
# # -----------------------------
# def generate_data(num_users=1000, num_items=200, num_sessions=5000):

#     users = pd.DataFrame({
#         "user_id": range(num_users),
#         "budget_segment": np.random.choice(["budget", "premium"], num_users),
#         "veg_preference": np.random.choice([0,1], num_users)
#     })

#     items = pd.DataFrame({
#         "item_id": range(num_items),
#         "category": np.random.choice(["main", "drink", "dessert", "side"], num_items),
#         "price": np.random.randint(50, 500, num_items),
#         "veg_flag": np.random.choice([0,1], num_items)
#     })

#     interactions = []

#     for _ in range(num_sessions):

#         user = random.randint(0, num_users-1)
#         hour = random.randint(8, 23)

#         # Positive samples
#         added_items = random.sample(range(num_items), 2)
#         for item in added_items:
#             interactions.append({
#                 "user_id": user,
#                 "item_id": item,
#                 "hour": hour,
#                 "added": 1
#             })

#         # Negative samples
#         not_added_items = random.sample(range(num_items), 2)
#         for item in not_added_items:
#             interactions.append({
#                 "user_id": user,
#                 "item_id": item,
#                 "hour": hour,
#                 "added": 0
#             })

#     interactions = pd.DataFrame(interactions)

#     return users, items, interactions


# # -----------------------------
# # Step 2: Prepare Dataset
# # -----------------------------
# users, items, interactions = generate_data()

# # Merge all data
# data = interactions.merge(users, on="user_id")
# data = data.merge(items, on="item_id")

# # One-hot encode categorical columns
# data = pd.get_dummies(data, columns=["budget_segment", "category"], drop_first=True)

# # Features and Target
# X = data.drop(columns=["added"])
# y = data["added"]

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -----------------------------
# # Step 3: Train Model
# # -----------------------------
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # -----------------------------
# # Step 4: Evaluate (ROC-AUC)
# # -----------------------------
# y_pred_prob = model.predict_proba(X_test)[:, 1]
# auc_score = roc_auc_score(y_test, y_pred_prob)

# print("ROC-AUC Score:", auc_score)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from feature_engineering import create_features


# 🔹 Precision@K function (outside train_model but below imports)
def precision_at_k(y_true, y_scores, k=10):

    data = list(zip(y_true, y_scores))
    data.sort(key=lambda x: x[1], reverse=True)

    top_k = data[:k]
    relevant = sum([1 for x in top_k if x[0] == 1])

    return relevant / k


def train_model():

    users = pd.read_csv("users.csv")
    items = pd.read_csv("items.csv")
    interactions = pd.read_csv("interactions.csv")

    data = create_features(users, items, interactions)

    data["budget_segment"] = data["budget_segment"].map({"budget":0, "premium":1})

    features = ["budget_segment", "veg_preference", "veg_flag",
                "price", "hour", "is_peak", "high_price"]

    X = data[features]
    y = data["added"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    print("Model trained successfully")

    # 🔹 AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print("AUC Score:", auc_score)

    # 🔹 Step 2 goes HERE 👇
    precision_k = precision_at_k(y_test.tolist(), y_pred_prob.tolist(), k=10)
    print("Precision@10:", precision_k)

    return model


if __name__ == "__main__":
    train_model()