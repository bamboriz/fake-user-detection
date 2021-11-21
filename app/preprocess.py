import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    X = data.copy()
    if 'Unnamed: 0' in X.columns:
        X = X.drop(['Unnamed: 0'], axis=1)
    X_preprocessed = engineer_features(X)
    X_preprocessed.fillna(0, inplace=True)
    return X_preprocessed

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    new_df_list = []
    users = list(data['UserId'].unique())

    for user_id in users:
        user = data[data['UserId'] == user_id]
        fake = user.Fake.to_list()[0]
        user_number_of_categories = user['Category'].nunique()
        user_category_counts = user['Category'].value_counts()
        user_number_of_events = user['Event'].nunique()
        user_event_counts = user['Event'].value_counts()
        user_event_counts_df = pd.DataFrame(user_event_counts).transpose().reset_index(drop=True)
        user_category_counts_df = pd.DataFrame(user_category_counts).transpose().reset_index(drop=True)
        df = pd.concat([user_event_counts_df, user_category_counts_df], axis=1)
        df['user_number_of_categories'] = user_number_of_categories
        df['user_number_of_events'] = user_number_of_events
        df['fake'] = fake
        df['user'] = user_id
        new_df_list.append(df.iloc[0].to_dict())

    return pd.DataFrame(new_df_list)
