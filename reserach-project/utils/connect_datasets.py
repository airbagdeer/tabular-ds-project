import pandas as pd

def combine_data_sets(user_reviews_df, sales_df, target_feature=None):
    e_user_reviews_df, p_user_reviews_df = preproccess_user_reviews_df(user_reviews_df=user_reviews_df)
    e_sales_df, p_sales_df = preproccess_sales_df(sales_df=sales_df)
    return e_user_reviews_df, p_user_reviews_df, e_sales_df, p_sales_df

def preproccess_sales_df(sales_df):
    # Drop the Rank column
    e_df = sales_df.drop(columns=["Rank"])
    # One-hot encode Name, Platform, and Genre
    p_df = pd.get_dummies(e_df, columns=["Name", "Platform", "Genre"])
    # Drop the Rank column
    return e_df, p_df

def preproccess_user_reviews_df(user_reviews_df):
    # Drop the Rank column
    e_df = user_reviews_df.drop(columns=["summary"])
    e_df['user_review'] = pd.to_numeric(e_df['user_review'], errors='coerce').round().astype('Int64')
    # One-hot encode Name, Platform, and Genre
    p_df = pd.get_dummies(e_df, columns=["platform"])
    # Drop the Rank column
    return e_df, p_df

def main():
    USER_REVIEWS_PATH = '/Users/itaygradenwits/Documents/biu/tablur_data_science/tabular-ds-project/reserach-project/data/video-games/all_games.csv'
    SALES_PATH = '/Users/itaygradenwits/Documents/biu/tablur_data_science/tabular-ds-project/reserach-project/data/video-games/vgsales.csv'

    user_reviews_df = pd.read_csv(USER_REVIEWS_PATH)
    sales_df = pd.read_csv(SALES_PATH)

    e_user_reviews_df, p_user_reviews_df, e_sales_df, p_sales_df = combine_data_sets(user_reviews_df, sales_df)
