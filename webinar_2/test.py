# %% [markdown]
# # Вебинар 2. Бейзлайны и детерминированные алгоритмы item-item

# %% [markdown]
# ![recsys_types.png](attachment:recsys_types.png)


# %% [markdown]
# [Implicit](https://implicit.readthedocs.io/en/latest/quickstart.html) - очень быстрая и 
# эффективная библиотека для рекоммендаций
# 
# Основные фичи:
#     - Cython под капотом - высокая скорость
#     - Множество приближенных алгоритмов - быстрее, чем оригинальные
#     - Содежрит большинство популярных алгоритмов
#     - Есть алгоритмы ранжирования
#     - Поиск похожих товаров / юзеров
#     - Есть возможность пересчета "холодного" юзера "на лету"
#     - Возможность фильтровать товары при рекомендациях (Например, не рекомендовать женские товары мужчинам)
#     - Есть метрики качества

# %% [markdown]
# # Подготовка данных

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Для работы с матрицами
from scipy.sparse import csr_matrix, coo_matrix

# Детерминированные алгоритмы
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender

# Метрики
from implicit.evaluation import train_test_split
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, AUC_at_k, ndcg_at_k

# %%
data = pd.read_csv('retail_train.csv')
data.head()

# %%
data['week_no'].nunique()

# %%
users, items, interactions = data.user_id.nunique(), data.item_id.nunique(), data.shape[0]

print('# users: ', users)
print('# items: ', items)
print('# interactions: ', interactions)

# %%
popularity = data.groupby('item_id')['sales_value'].sum().reset_index()
popularity.describe()

# %%
popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
popularity.describe()

# %% [markdown]
# **Note:**  
# Еще есть данные по характеристикам товаров и пользователей. Они нам пригодятся через несколько вебинаров

# %%
item_features = pd.read_csv('product.csv')
item_features.head()

# %%
user_features = pd.read_csv('hh_demographic.csv')
user_features.head()

# %% [markdown]
# ### Train-test split

# %% [markdown]
# В рекомендательных системах корректнее использовать train-test split по времени, а не случайно  
# Я возьму последние 3 недели в качестве теста

# %%
test_size_weeks = 3

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

# %%
data_train.shape[0], data_test.shape[0]

# %% [markdown]
# # 1. Бейзлайны

# %% [markdown]
# Создадим датафрейм с покупками юзеров на тестовом датасете (последние 3 недели)

# %%
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns=['user_id', 'actual']
result.head(2)

# %%
test_users = result.shape[0]
new_test_users = len(set(data_test['user_id']) - set(data_train['user_id']))

print('В тестовом дата сете {} юзеров'.format(test_users))
print('В тестовом дата сете {} новых юзеров'.format(new_test_users))

# %% [markdown]
# ### 1.1 Random recommendation

# %%
def random_recommendation(items, n=5):
    """Случайные рекоммендации"""
    
    items = np.array(items)
    recs = np.random.choice(items, size=n, replace=False)
    
    return recs.tolist()

# %%


items = data_train.item_id.unique()

result['random_recommendation'] = result['user_id'].apply(lambda x: random_recommendation(items, n=5))

result.head(2)

# %% [markdown]
# ### 1.2 Popularity-based recommendation

# %%
def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""
    
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    
    recs = popular.head(n).item_id
    
    return recs.tolist()

# %%

# Можно так делать, так как рекомендация не зависит от юзера
popular_recs = popularity_recommendation(data_train, n=5)

result['popular_recommendation'] = result['user_id'].apply(lambda x: popular_recs)
result.head(2)

# %% [markdown]
# ### 1.3 Weighted random recommender

# %% [markdown]
# - Можно сэмплировать товары случайно, но пропорционально какому-либо весу
# - Например, прямопропорционально популярности. Вес = log(sales_sum товара)

# %% [markdown]
# *Пример*  
# item_1 - 5, item_2 - 7, item_3 - 4  # / sum  
# item_1 - 5 / 16, item_2 - 7 / 16, item_3 - 4 / 16

# %%
def weighted_random_recommendation(items_weights, n=5):
    """Случайные рекоммендации
    
    Input
    -----
    items_weights: pd.DataFrame
        Датафрейм со столбцами item_id, weight. Сумма weight по всем товарам = 1
    """
    
    # Подсказка: необходимо модифицировать функцию random_recommendation()
    # your_code
    
    return recs.tolist()

# %% [markdown]
# ### Выводы по бейзлайнам
# - Фиксируют базовое качество;
# - Бейзлайны могут быть фильтрами;
# - Иногда бейзлайны лучше ML-модели

# %% [markdown]
# # 2. Детерминированные алгоритмы

# %% [markdown]
# ## 2.1 Item-Item Recommender / ItemKNN

# %% [markdown]
# ![user_item_matrix.png](attachment:user_item_matrix.png)

# %% [markdown]
# То, что именно находится в матрице user-item нужно определять из бизнес-логики
# 
# Варианты для нашего датасета(не исчерпывающий список):
#     - Факт покупки (0 / 1)
#     - Кол-во покупок (count)
#     - Сумма покупки, руб
#     - ...
#     
# **Детерминированные алгоритмы**:
#     - Предсказывают те числа, которые стоят в матрице
# 
# **ML-алгоритмы (большинство)**:
#     - В качестве *таргетов* "под капотом" принимают 0 и 1 (в ячейке не 0 -> таргет 1)
#     - А абсолютные значения воспринимают как *веса ошибок*
#     
# *P.S.* На самом деле есть много трюков, как можно заполнять матрицу user-item.

# %% [markdown]
# **Как работает Item-Item Recommender**

# %% [markdown]
# ![item_item_recommender.png](attachment:item_item_recommender.png)

# %% [markdown]
# *Шаг 1:* Ищем K ближайших товаров к товару  
# *Шаг 2*: predict "скора" товара = среднему "скору" этого товара у его соседей  
# *Шаг 3*: Сортируем товары по убыванию predict-ов score и берем топ-k

# %% [markdown]
# ----
# **(!) Важно** 
# - У item-item алгоритмов большая сложность ($O(I^2 log(I))$ или $O(I^3)$, в зависимости от реализации 
# - Если в датасете много item_id, то item-item модели ОЧЕНЬ долго предсказывают. Со всеми товарами predict на тесте ~2 часа
# - Давайте возьмем из ~90к товаров только 5k самых популярных 
# 
# *P.S.*  Брать топ-Х популярных и рекомендовать только из них - очень популярная стратегия.   
# *P.P.S.*  В рекомендательных системах много таких трюков. Что-то подобное в курсе вы увидите еще не раз

# %%
popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

popularity.head()

# %%
top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()

# %%
top_5000

# %%
data_train.head(100)

# %%
# Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999
data_train.head(100)

# %%
user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )

user_item_matrix[user_item_matrix > 0] = 1 # так как в итоге хотим предсказать 
user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

# переведем в формат sparse matrix
sparse_user_item = csr_matrix(user_item_matrix).tocsr()



# %%
user_item_matrix

# %%
user_item_matrix.shape

# %%
user_item_matrix

# %%
user_item_matrix.sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100

# %%
user_item_matrix.shape[0] * user_item_matrix.shape[1]

# %%
user_item_matrix.sum().sum()

# %%
userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))

# %%


model = ItemItemRecommender(K=5, num_threads=4) # K - кол-во билжайших соседей


model.fit(csr_matrix(user_item_matrix).T.tocsr(),  # На вход item-user matrix
          show_progress=True)


recs = model.recommend(userid=userid_to_id[2],  # userid - id от 0 до N
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=5, # кол-во рекомендаций 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=True)

# %%
userid_to_id[2]

# %%
recs

# %%
[id_to_itemid[rec[0]] for rec in recs]

# %%


result['itemitem'] = result['user_id'].apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)])

# %%
result.head(5)

# %% [markdown]
# ### 4.2 Косинусное сходство и CosineRecommender

# %% [markdown]
# ![cosine_similarity.png](attachment:cosine_similarity.png)

# %%


model = CosineRecommender(K=5, num_threads=4) # K - кол-во билжайших соседей

model.fit(csr_matrix(user_item_matrix).T.tocsr(), 
          show_progress=True)

recs = model.recommend(userid=userid_to_id[1], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=5, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)

# %%
[id_to_itemid[rec[0]] for rec in recs]

# %%


result['cosine'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)])

# %%
result.head(10)

# %% [markdown]
# ### 4.3 TF-IDF взвешивание и TFIDFRecommender

# %% [markdown]
# ![tf_idf.png](attachment:tf_idf.png)

# %% [markdown]
# Если 2 юзера оба купили очень популярный товар, то это еще не значит,что они похожи   
# Если 2 юзера оба купили редкий товар, то они похожи
# 
# Занижаем вес популярных товаров при расчете расстояний между пользователями

# %%


model = TFIDFRecommender(K=5, num_threads=4) # K - кол-во билжайших соседей

model.fit(csr_matrix(user_item_matrix).T.tocsr(), 
          show_progress=True)

recs = model.recommend(userid=userid_to_id[1], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=5, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)

# %%
[id_to_itemid[rec[0]] for rec in recs]

# %%


result['tfidf'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=False)])

# %%
result.head(5)

# %% [markdown]
# ### 4.4 Трюк

# %%


model = ItemItemRecommender(K=1, num_threads=4) # K - кол-во билжайших соседей


model.fit(csr_matrix(user_item_matrix).T.tocsr(), 
          show_progress=True)

recs = model.recommend(userid=userid_to_id[1], 
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=5, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)

# %%
# print map item_id to row_id
itemid_to_id
print(*[f"Item ID {id_to_itemid[rec[0]]} --> Row ID {rec[0]}" for rec in recs],sep='\n')

# %%


result['own_purchases'] = result['user_id'].\
    apply(lambda x: [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[x], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=5, 
                                    filter_already_liked_items=False, 
                                    filter_items=[itemid_to_id[999999]], 
                                    recalculate_user=False)])

# %% [markdown]
# ### 4.5 Измерим качество по precision@5

# %%
result.head(5)

# %%
#import os
#os.makedirs('../predictions', exist_ok=True)
result.to_csv('predictions_basic.csv', index=False)

# %% [markdown]
# Можно ли улучшить бейзлайны, если считать их на топ-5000 товарах?

# %%
# your_code

# %%
# Функции из 1-ого вебинара
import os, sys
    
from metrics import precision_at_k, recall_at_k

# %% [markdown]
# # Metrics

# %%
for name_col in result.columns[1:]:
    print(f"{round(result.apply(lambda row: precision_at_k(row[name_col], row['actual']), axis=1).mean(),4)}:{name_col}")

# %% [markdown]
# # Articles

# %% [markdown]
# https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
# 
# https://en.wikipedia.org/wiki/Collaborative_filtering
# 
# https://core.ac.uk/download/pdf/83041762.pdf
# 
# 
# 
# User-based and item-based CF explanation
# 
# https://medium.com/@cfpinela/recommender-systems-user-based-and-item-based-collaborative-filtering-5d5f375a127f

# %%



