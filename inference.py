import pandas as pd

def recommend(model, user_features, item_df, hour):

    recommendations = []

    for _, item in item_df.iterrows():

        input_data = {
             "budget_segment": 0 if user_features["budget_segment"] == "budget" else 1,
            "veg_preference": user_features["veg_preference"],
            "veg_flag": item["veg_flag"],
            "price": item["price"],
            "hour": hour,
            "is_peak": 1 if 12<=hour<=14 or 19<=hour<=22 else 0,
            "high_price": 1 if item["price"]>300 else 0
        }

        prob = model.predict_proba(pd.DataFrame([input_data]))[0][1]
        recommendations.append((item["item_id"], prob))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:10]