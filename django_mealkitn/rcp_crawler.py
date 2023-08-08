import pandas as pd
import numpy as np
import re
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from gensim.models import Word2Vec
from mk_recs.models import ManRecipe


##### 요리재료와 쿠팡 구성정보 일치 #####
def get_ingredients_pre(word):
    word_list = ['가루', '꼬막', '어묵', '비엔나 소시지', '대파', '피망', '파프리카', '무우', '양파', '미역', '오이', '식초', '간장', '오일', '스프',
                 '아보카도', '물엿', '올리고당', '후추', '젓갈', '즙', '와인', '정종', '액젓', '매생이', '아귀']
    if ('소고기' in word) or ('쇠고기' in word):
        word = '소고기'
    elif ('건면' in word) or ('생면' in word) or ('파스타 면' in word) or ('파스타면' in word) or ('라면 사리' in word) or (
            '우동 사리' in word) or ('라면사리' in word) or ('우동사리' in word) or ('칼국수 면' in word) or ('칼국수면' in word) or (
            '스파게티 면' in word) or ('스파게티면' in word) or ('페투치네' in word) or ('우동면' in word) or ('우동 면' in word) or (
            '숙면' in word) or ('소면' in word) or ('쫄면' in word):
        word = '면'
    elif '고추장' in word:
        word = '고추장'
    elif ('고추' in word) and ('고추장' not in word):
        word = '고추'
    elif '새우젓' in word:
        word = '새우젓'
    elif ('새우' in word) and ('새우젓' not in word):
        word = '새우'
    # elif '명태알 소스' in word: word = '명란'
    elif ('닭' in word) or ('수비드 찜닭' in word):
        word = '닭'
    elif ('계란' in word) or ('달걀' in word):
        word = '계란'
    elif ('소금' in word) or ('천일염' in word):
        word = '소금'
    elif '다진' == word:
        word = '다진마늘'
    elif ('케첩' in word) or ('케찹' in word):
        word = '케첩'
    elif ('식용유' in word) or ('올리브유' in word) or ('카놀라유' in word) or ('기름' in word):
        word = '식용유'

    else:
        # 단어 리스트에 있는 재료명 통일
        for num in range(len(word_list)):
            if word_list[num] in word:
                word = word_list[num]
    return word


##### input 값 전처리 후 list에 단어별 저장 #####
def get_product_list(num, product_df):
    # 상품명 전처리
    product = product_df.at[num, '상품_전처리']  # string으로 반환
    product_list = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\(\)]', '', product).split(' ')
    # 특수문자제거(만개의레시피 데이터엔 특수문자가 없어서 인식 불가)
    product_list = list(filter(None, product_list))  # null 제거(안 할 경우 레시피 전체가 선택되어 작동 불가)

    # 구성정보 전처리
    str_list = product_df.at[num, '구성정보_전처리']
    ingredients_list = str_list.split(',')

    # 구성정보 요리재료명 통일
    for idx in range(len(ingredients_list)):
        ingredients_list[idx] = get_ingredients_pre(ingredients_list[idx])

    # 구성정보 요리재료의 중복값 제거
    result = []
    [result.append(x) for x in ingredients_list if x not in result]

    return product_list, result  # 쿠팡의 상품명, 구성정보_전처리의 요리재료명 반환


##### input 상품의 상세카테고리 추출 #####
def get_product_category(num, product_df):
    # 만개의레시피의 카테고리를 기준으로 쿠팡의 세부 카테고리를 분류
    ckg_dict = {
        '요리방법별명': ['끓이기', '굽기', '볶음', '튀김', '기타', '부침', '조림', '찜', '무침', '절임', '비빔', '데치기', '삶기', '회'],
        '요리상황별명': ['일상', '술안주', '손님접대', '도시락', '야식', '간식', '영양식', '명절', '초스피드', '다이어트', '해장', '이유식'],
        '요리재료별명': ['쌀', '건어물류', '가공식품류', '버섯류', '닭고기', '해물류', '채소류', '육류', '밀가루', '콩/견과류', '돼지고기', '달걀/유제품', '소고기',
                   '곡류', '과일류'],
        '요리종류별명': ['밥/죽/떡', '메인반찬', '면/만두', '양식', '밑반찬', '찌개', '국/탕', '김치/젓갈/장류', '퓨전']
    }
    # 한식의 '밥/죽/떡', '메인반찬'
    # 순대의 '밥/죽/떡', '메인반찬'
    ## 각각 같은 카테고리로 충돌 > 메인반찬의 카테고리 레시피만 남음 >> 한식의 비빔밥, 순대의 국밥은 아예 결과가 제대로 안 나옴
    ### >> 쿠팡에서 세부카테고리를 조정하면 어떨지?
    coupang_dict = {
        '전골': '국/탕',
        '생선탕': '해물류',
        '찌개/짜글이': '찌개',
        '한식': '메인반찬,굽기',
        '알탕': '해물류',
        '해물탕': '국/탕',
        '기타': 'null',
        '국': '국/탕',
        '씨푸드': '해물류',
        '구이': '굽기',
        '찜': '찜',
        '닭': '닭고기',
        '면': '면/만두',
        '국수': '면/만두',
        '스테이크': '굽기,양식',
        '파스타': '면/만두,양식',
        '감바스': '양식',
        '볶음': '볶음',
        '떡볶이': '밥/죽/떡',
        '마파두부': '볶음',
        '마라탕': '국/탕',
        '쌈': '메인반찬',
        '무침': '무침',
        '샌드위치': '일상,기타',
        '족발': '돼지고기',
        '리조또': '밥/죽/떡',
        '카레': '일상',
        '덮밥/비빔밥': '밥/죽/떡'
    }

    product_category = coupang_dict[product_df.at[num, '상세카테고리']].split(',')  # 리스트 반환
    ckg_key_value = {}
    if product_category == ['null']:  # 쿠팡의 세부카테고리==기타인 경우, null 반환
        ckg_key_value = {'null': 'null'}
    else:
        for category in product_category:
            for ckg_key in ckg_dict.keys():  # 만개의레시피 해당 컬럼의 세부카테고리 리스트 저장
                ckg_value = ckg_dict.get(ckg_key)
                if category in ckg_value:  # 만개의레시피 세부카테고리에 쿠팡 세부카테고리가 있으면
                    ckg_key_value[ckg_key] = category  # { key=만개의레시피 해당컬럼 : value=해당컬럼의 세부 카테고리}로 저장
    return ckg_key_value


##### input 상품과 동일한 요리명 및 카테고리 의 레시피만 추출 #####
def get_recipe_df(ckg_df, product_list, ckg_key_value):
    # 1차필터링
    recipe_df = pd.DataFrame()  # 추출한 만개의 레시피를 저장할 빈 df 생성
    for word in product_list:
        # 쿠팡상품명과 레시피요리명을 동일하게 변경
        if '보일링' in word:
            word = '해물찜'
        elif '씨 키트' in word:
            word = '해물'
        elif '냉동' in word:
            word = 'NaN'  # 쿠팡 상품명에 냉동이 있는 경우 제외하기 위한 값
        else:
            pass
        # 해당 쿠팡 상품명 단어와 일치하는 요리명의 index만 저장
        recipe = ckg_df[ckg_df['요리명'].str.contains(word)]
        # 위에서 생성한 df와 결합
        recipe_df = pd.concat([recipe_df, recipe])
    # 2차 필터링
    recipe_df_filter = pd.DataFrame()
    for ckg_key in ckg_key_value.keys():
        if ckg_key == 'null':
            # 세부카테고리==기타 인 경우, 위의 df 그대로 사용
            recipe_df_filter = recipe_df
        else:
            # 해당하는 컬럼의 일치하는 세부카테고리만 저장
            recipe_df_key = recipe_df[recipe_df[f'{ckg_key}'] == ckg_key_value.get(ckg_key)]
            recipe_df_filter = pd.concat([recipe_df_filter, recipe_df_key])
    # 한번에 중복값 제거
    recipe_df_filter.drop_duplicates(subset=['레시피일련번호'], keep='first', inplace=True)

    return recipe_df_filter


##### 추출한 만개의 레시피의 요리재료명 통일 및 word2vec을 위한 이중리스트 생성 #####
def get_recipe_pre(recipe_index, recipe_df):
    # 만개의 레시피에서 사용된 양념 재료 사전
    sauce_dict = ['가루', '설탕', '간장', '된장', '식초', '오일', '액젓', '소금', '물', '소주', '스프', '고추장', '고춧가루', '사이다', '소스', '케첩',
                  '물엿', '올리고당', '매실청', '식용유', '후추', '젓갈', '즙', '정종', '와인', '맛술']

    recipe_lists = []
    for index in recipe_index:
        man_ingredients = recipe_df.loc[index, '요리재료_전처리']
        # 요리재료가 word2vec을 위해 list형태로 저장
        # 반복문으로 재료명 통일
        recipe_list = []
        for man_ingredient in man_ingredients:
            man_ingredient = get_ingredients_pre(man_ingredient)
            # 추가할 필요가 없는 양념재료 제거
            # 양념 사전와 불일치하는 경우에만 recipe_list에 추가
            if man_ingredient in sauce_dict:
                pass
            else:
                recipe_list.append(man_ingredient)
        # 이중 리스트로 '요리재료_전처리' 전체 반환
        recipe_lists.append(recipe_list)
    return recipe_lists


###### 단어 벡터 평균구하기 ######
def vectors(embedding, embedding_model):
    word_embedding = []
    dec2vec = None
    count = 0
    for word in embedding:
        try:
            # word에 해당하는 단어를 워드투백에서 찾아 리스트에 붙힘
            # 100차원으로 형성됨
            word_embedding.append(embedding_model.wv[word])
            count += 1
        except:
            continue

    word_embedding2 = np.array(word_embedding)
    v = word_embedding2.sum(axis=0)

    if type(v) != np.ndarray:
        return np.zeros(100)

    return (v / count)


###### 최종 알고리즘 ######
def wv_mealkit(sim, mealkit_index, top_n, recipe_df):
    # 코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
    sim_sorted_ind = sim.argsort()[:, ::-1]

    # sim_indexes는 이중리스트 상태이므로 이중리스트를 해제한 후 인덱스를 이용해 해당 내용 추출
    similar_indexes = sim_sorted_ind[mealkit_index, 1:(top_n + 1)]
    recipe = recipe_df.iloc[similar_indexes]  # .sort_values(by='weighted_clipping',ascending=False)
    return recipe


###### 레시피 only 재료 추출 ######
def get_recipe_only(ingredients_list, sim_mealkit_df):
    recipe_ingredient = []
    for idx in range(len(sim_mealkit_df)):
        value_list = sim_mealkit_df.요리재료_전처리.values[idx]

        value_set = []  # 중복제거
        [value_set.append(x) for x in value_list if x not in value_set]

        for ingredient in ingredients_list:
            # input 구성정보와 요리재료가 일치하면 제외
            if ingredient in value_set:
                value_set.remove(ingredient)
        # 레시피에만 있는 요리재료 추가
        recipe_ingredient.append(value_set)
    return recipe_ingredient

def add_new_recipes(data, data_id, data_name, data_description, data_ingredient, data_allergy):

    # print("new item added!! : " + data['레시피제목'])

    ManRecipe(recipe_id=data_id,
              recipe_name=data_name,
              recipe_description=data_description,
              recipe_ingredient=data_ingredient,
              recipe_allergy=data_allergy).save()

    return data