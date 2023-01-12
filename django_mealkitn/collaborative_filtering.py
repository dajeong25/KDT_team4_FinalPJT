import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def collaborative(df):
    rating_dummy = df.pivot_table('rating', index='상품', columns='id')
    # dummy_rating = df.pivot_table('rating', index='id', columns='상품')
    rating_dummy.fillna(0, inplace=True)
    item = cosine_similarity(rating_dummy)
    item = pd.DataFrame(data=item, index=rating_dummy.index, columns=rating_dummy.index)
    return item


def get_item(item, 상품):
    return item[상품].sort_values(ascending=False)[1:7]
