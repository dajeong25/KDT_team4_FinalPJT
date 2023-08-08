import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import urllib.request
import pickle
import os
from urllib.parse import urlparse
import pytesseract
import cv2
from PIL import Image
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault('DJANGO_SETTINGS_MODULE', "config.settings")

import django

django.setup()

from mk_recs.models import CoupangMealkit

#records = CoupangMealkit.objects.all()
#records.delete()

nltk.download('punkt')


def get_coupang_item(URL, user_agt):
    headers = {"User-Agent":user_agt
               , "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
    res = requests.get(URL, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'lxml')
    return soup


# 카테고리 대분류 라벨링 함수
def category_number(categoryId, category):
    return category.get(categoryId)


# 해당 카테고리의 페이지 전체 수 가져오기
def page_number(categoryId, user_agt):
    URL = 'https://www.coupang.com/np/categories/' + categoryId + "?page=1"
    cg_soup_base = get_coupang_item(URL, user_agt)
    page_all = cg_soup_base.select('#product-list-paging > div > a')
    page_list = []
    for page_num in page_all:
        page_num = page_num.text
        try:
            page_num = int(page_num)
            page_list.append(page_num)
        except:
            pass
    return page_list[-1]


# 1차 쿠팡 상품 리스트 전처리 함수
def p_list_preprocessing(cg_soup, category_num):
    # 상품id, data-item-id, data-vendor-item-id
    bady_pdt = cg_soup.select_one("a.baby-product-link")
    productId = bady_pdt['data-product-id']
    itemsId = bady_pdt['data-item-id']
    vendorItemId = bady_pdt['data-vendor-item-id']

    # 상품명
    name = cg_soup.select_one("div.name").text.replace('\n', '').strip()
    name_only = name.split(', ')[0]

    # 판매가
    price_value = cg_soup.select_one("strong.price-value").text.replace(',', '')

    try:
        # 정가
        base_price = cg_soup.select_one("del.base-price").text.replace("\n", '').strip().replace(',', '')
        # 할인율
        discount_pcg = cg_soup.select_one("span.discount-percentage").text
    except:
        base_price, discount_pcg = price_value, 0

    # 100g당 가격
    try:
        unit_price = cg_soup.select_one("span.unit-price").text.replace("\n", '').strip()
    except:
        unit_price = np.NaN

    try:
        # 별점
        star = cg_soup.select_one("span.star > em").text
        # 리뷰수
        rating_count = cg_soup.select_one("span.rating-total-count").text
        rating_count = re.sub(r'[^0-9]', '', rating_count)
    except:
        star, rating_count = 0, 0

    # 품절여부 > 품절이면 1, 아니면 0
    try:
        out_of_stock = cg_soup.select_one("div.out-of-stock").text.replace("\n", '').strip()
        out_of_stock = 1
    except:
        out_of_stock = 0

    # DataFrame으로 병합
    coupang_item = pd.DataFrame(
        {'카테고리명': category_num, '상품id': productId, 'data-item-id': itemsId, 'data-vendor-item-id': vendorItemId
            , '상품': name_only, '상품명': name, '정가': base_price, '할인율': discount_pcg, '판매가': price_value
            , '100g당_가격': unit_price, '별점': star, '리뷰수': rating_count, '품절여부': out_of_stock}
        , index=[0])
    return coupang_item


# 2차 : 구성정보 추출 함수
def p_c_extraction(productId, p_c_page):
    product_composition = '0'
    for item in p_c_page:
        s = item.text.replace('\n', '').strip()
        if product_composition == '2':  # 3: 재료부분만 추출
            product_composition = s
        elif product_composition == '1':  # 2: 불필요한 text 제외
            product_composition = '2'
        elif '구성 정보' in s:  # 1: '구성 정보' 존재여부 확인
            product_composition = '1'

    # 4: 없으면 결측값 처리
    if product_composition.isdigit():
        product_composition = np.NaN

    data_item = pd.DataFrame({'상품id': productId, '구성정보': product_composition}, index=[0])
    return data_item


def get_coupang_image(img_URL, user_agt):
    headers = {"User-Agent":user_agt
               , "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
    page = requests.get(img_URL, headers= headers).text
    #coupang_image_soup = BeautifulSoup(page.text, "html.parser")
    #return coupang_image_soup
    return page


# 구성정보 재료 전처리
def p_c_preprocessing(product_composition):
    product_composition = re.sub(r'\\n', '', product_composition)
    product_composition = re.sub(r'\)', ',', product_composition)
    product_composition = re.sub(r'[^ㄱ-ㅣ가-힣a-zA-Z\,\(\:]', ' ', product_composition)
    product_composition = product_composition.strip()

    replace_list = ['으로 구성되어 있습니다', '으로 구성되어있습니다', '로 구성되어 있습니다', '로 구성되어 있어요', '로 구성되어있습니다']
    for r in replace_list:
        product_composition = product_composition.replace(r, '')

    replace_list = ['과', '와']
    for r in replace_list:
        product_composition = product_composition.replace(r, ',')

        # 추가 작업에 따라 추가 여부 결정
    # if ('야채' in product_composition) or ('채소(' in product_composition) or ('채소 :' in product_composition):
    #     replace_list = ['야채 세트', '혼합 채소', '혼합 해물', '채소(', '채소 :']
    #     for r in replace_list:
    #         product_composition = re.sub('[\(\)\:]', ',', product_composition)
    #         product_composition = product_composition.replace(r, '')
    # else:
    product_composition = re.sub('[\(\)\:]', ',', product_composition)

    return product_composition

def deleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'

def category_name(product):
    try:
        if ('닭' in product) or ('백숙' in product):
            name = '닭'
        elif '구이' in product:
            name = '구이'
        elif ('전골' in product) or ('나베' in product) or ('스키야키' in product) or ('샤브샤브' in product):
            name = '전골'
        elif '족발' in product:
            name = '족발'
        elif '국수' in product:
            name = '국수'
        elif ('해물탕' in product) or ('연포탕' in product) or ('해신탕' in product) or ('꽃게탕' in product) or (
                '해물 누룽지탕' in product) or ('조개탕' in product) or ('바지락탕' in product):
            name = '해물탕'
        elif ('해물뚝배기' in product) or ('문어오리탕' in product) or ('홍합탕' in product):
            name = '해물탕'
        elif ('매운탕' in product) or ('대구탕' in product) or ('동태탕' in product) or ('지리탕' in product) or ('맑은탕' in product) or (
                '맑음탕' in product):
            name = '생선탕'
        elif ('알탕' in product) or ('알이 맛있탕' in product) or ('곤이탕' in product):
            name = '알탕'

        elif ('찌개' in product) or ('짜글이' in product) or ('짬뽕 순두부' in product):
            name = '찌개/짜글이'

        elif ('면' in product) or ('소바' in product) or ('짜장' in product) or ('누들' in product) or ('짬뽕' in product):
            name = '면'
        elif ('우동' in product) or ('라면' in product):
            name = '면'
        elif '스테이크' in product:
            name = '스테이크'

        elif ('파스타' in product) or ('라자냐' in product):
            name = '파스타'

        elif ('비빔밥' in product) or ('덮밥' in product):
            name = '덮밥/비빔밥'

        elif '리조또' in product:
            name = '리조또'

        elif '마라탕' in product:
            name = '마라탕'
        elif '마파두부' in product:
            name = '마파두부'

        elif ('떡볶이' in product) or ('떡볶잉' in product):
            name = '떡볶이'

        elif '무침' in product:
            name = '무침'
        elif '감바스' in product:
            name = '감바스'

        elif ('콩불' in product) or ('불고기' in product) or ('부침개' in product) or ('조림' in product):
            name = '한식'

        elif ('해물' in product) or ('씨푸드' in product) or ('씨키트' in product) or ('꽃게' in product) or ('새우' in product):
            name = '씨푸드'
        elif ('쉬림프' in product) or ('연어' in product) or ('조개' in product) or ('문어' in product):
            name = '씨푸드'


        elif ('커리' in product) or ('카레' in product):
            name = '카레'

        elif '샌드위치' in product:
            name = '샌드위치'
        elif ('볶음' in product) or ('잡채' in product) or ('순대볶음' in product):
            name = '볶음'

        elif '쌈' in product:
            name = '쌈'
        elif ('국' in product) or ('육개장' in product) or ('파개장' in product) or ('수제비' in product) or ('칼제비' in product) or (
                '순대국밥' in product):
            name = '국'
        elif '찜' in product:
            name = '찜'
        else:
            name = '기타'
    except:
        name = '기타'
    return name

def allergy(dataframe, column, column2):
    allergy_dict = {'갑각류': ['새우', '꽃게', '랍스터', '가재', '대하', '쉬림프', '감바스'],
                    '견과류': ['밤', '호두', '땅콩', '아몬드', '잣', '은행', '헤이즐넛', '크랜베리', '캐슈넛', '참깨', '참기름', '들깨', '들기름'],
                    '달걀': ['달걀', '계란', '오코노미야끼', '수란', '알 가열제품', '지단', '반숙란'],
                    '메밀': ['메밀'],
                    '밀가루': ['통돈까스', '밀가루', '또띠아', '부침개', '수제비', '칼국수', '라면', '우동', '유탕면', '탄탄면', '중화면', '건면', '생면',
                            '생중면', '파스타', '스파게티', '숙면', '소면', '쫄면', '만두', '밀떡', '바게트', '바게트 빵', '빵', '중화 면', '사리'],
                    '우유': ['우유', '치즈', '버터', '마가린', '요거트', '요구르트', '빠네', '투움바', '까르보나라', '크림', '리조또'],
                    '연체동물류': ['조개', '홍합', '굴', '가리비', '바지락', '전복', '꼬막', '해산물', '해물', '문어', '오징어', '낙지', '연포탕', '해신탕',
                              '낙곱새', '성게', '우렁살', '봉골레', '관자'],
                    '콩': ['콩나물', '두부', '대두', '간장', '된장', '콩기름'],
                    '닭': ['닭', '치킨'],
                    '돼지': ['통돈까스', '돼지', '부대찌개', '돔베', '햄', '코테기노', '돈마호크', '베이컨', '탄탄면', '볼로네제', '비엔나', '족발', '보쌈',
                           '마제소바', '목살', '양장피', '통삼겹', '포크'],
                    '복숭아': ['복숭아'],
                    '소': ['소고기', '쇠고기', '소 스지', '샤브샤브', '도가니', '양지', '육개장', '우삼겹', '찹스테이크', '볼로네제', '비프', '비엔나'],
                    '토마토': ['토마토', '볼로네제'],
                    '생선류': ['굴비', '조기', '고등어', '갈치', '꽁치', '전어', '명태', '노가리', '메기', '동태', '곤이', '아귀', '아구', '코다리', '대구',
                            '복지리', '우럭', '가자미', '도다리', '해물탕', '어묵']}

    allergy_list = []
    for idx, product in enumerate(dataframe[column]):
        sublist = []
        for key in list(allergy_dict.keys()):
            for i in allergy_dict[key]:
                if i in product:
                    sublist.append(key)
                if i in dataframe[column2][idx]:
                    sublist.append(key)
        sublist = list(set(sublist))
        sublist = ', '.join(sublist)
        allergy_list.append(sublist)
    return allergy_list


def weighted_rating(v, m, R, C):
    '''
    Calculate the weighted rating

    Args:
    v -> average rating for each item (float)
    m -> minimum votes required to be classified as popular (float)
    R -> average rating for the item (pd.Series)
    C -> average rating for the whole dataset (pd.Series)
    # ( (각 리뷰수 / (각 리뷰수+최소리뷰수)) * 각 별점 ) + ( (최소리뷰수/(각 리뷰수+최소리뷰수)) * 평균별점 )

    Returns:
    pd.Series
    '''
    return ((v / (v + m)) * R) + ((m / (v + m)) * C)


# 데이터프레임과 사전의 text를 비교해서 필요한 내용만 추출하는 함수
def dict_to_dataframe(dataframe, dict, column):
    menu_list = []
    for idx, product in enumerate(dataframe[column]):
        sublist = []
        for key in list(dict.keys()):
            for i in dict[key]:
                if i in product:
                    sublist.append(key)
        # 중복 제거
        sublist = list(set(sublist))
        sublist = ', '.join(sublist)
        menu_list.append(sublist)
    return menu_list

# price_mean에서 추출한 메뉴별 평균 가격을 coupang 데이터에 apply
# 이 때 price_mean.index에 없는 x의 경우에는, split으로 각 메뉴를 분리후 각각의 가격을 합계해주어 리턴해주는 방식
def func(x, price_mean) :
    if x == '' :
        return 0
    if x in list(price_mean.index) :
        return price_mean[x]
    else :
        total_price = 0
        a = x.split(', ')
        for i in a:
            total_price += price_mean[i]
        return total_price

def add_new_items(dataframe):
    '''
    last_inserted_items = BoardData.objects.last()
    if last_inserted_items is None:
        last_inserted_specific_id = ""
    else:
        last_inserted_specific_id = getattr(last_inserted_items, 'product_id')

    items_to_insert_into_db = []
    for item in dataframe:
        if item['product_id'] == last_inserted_specific_id:
            break
        items_to_insert_into_db.append(item)
    items_to_insert_into_db.reverse()
    '''
    for i in range(len(dataframe)):
        item = dataframe.loc[i]
        # print("new item added!! : " + item['상품'])

        CoupangMealkit(category_name=item['카테고리명'],
                  category_second=item['상세카테고리'],
                  allergy=item['알레르기'],
                  product_id=item['상품id'],
                  data_item_id=item['data-item-id'],
                  data_vendor_item_id=item['data-vendor-item-id'],
                  product=item['상품'],
                  product_name=item['상품명'],
                  full_price=item['정가'],
                  discounted_rate=item['할인율'],
                  discounted_price=item['판매가'],
                  yogiyo_price=item['요기요가격'],
                  ratings=item['별점'],
                  review_count=item['리뷰수'],
                  issoldout=item['품절여부'],
                  ingredient=item['구성정보'],
                  edited_ingredient=item['구성정보_전처리'],
                  price_per_100g=item['100g당_가격'],
                  weighted_rating=item['가중치']).save()

    return dataframe

def coupang_naver_categories(category, naver_df):
    dict1 = {'밥/죽': ['리조또', '한식', '덮밥/비빔밥'],
             '찌개/국': ['전골', '생선탕', '찌개/짜글이', '알탕', '해물탕', '국', '마라탕'],
             '면/파스타': ['면', '국수', '파스타'],
             '구이': ['구이', '스테이크'],
             '볶음/튀김': ['볶음', '감바스'],
             '조림/찜': ['찜', '마파두부'],
             '간식/디저트': ['샌드위치', '족발'],
             '전체': ['씨푸드', '닭', '떡볶이', '무침', '쌈', '카레']}
    if category in dict1['밥/죽']:
        naver_value = naver_df.loc[naver_df['밥/죽_카테고리'] == category]
        naver_value = naver_value['밥/죽'].values
        naver_value = naver_value.tolist()
    elif category in dict1['찌개/국']:
        naver_value = naver_df.loc[naver_df['찌개/국_카테고리'] == category]
        naver_value = naver_value['찌개/국'].values
        naver_value = naver_value.tolist()
    elif category in dict1['구이']:
        naver_value = naver_df.loc[naver_df['구이_카테고리'] == category]
        naver_value = naver_value['구이'].values
        naver_value = naver_value.tolist()
    elif category in dict1['면/파스타']:
        naver_value = naver_df.loc[naver_df['면/파스타_카테고리'] == category]
        naver_value = naver_value['면/파스타'].values
        naver_value = naver_value.tolist()
    elif category in dict1['볶음/튀김']:
        naver_value = naver_df.loc[naver_df['볶음/튀김_카테고리'] == category]
        naver_value = naver_value['볶음/튀김'].values
        naver_value = naver_value.tolist()
    elif category in dict1['조림/찜']:
        naver_value = naver_df.loc[naver_df['조림/찜_카테고리'] == category]
        naver_value = naver_value['조림/찜'].values
        naver_value = naver_value.tolist()
    elif category in dict1['간식/디저트']:
        naver_value = naver_df.loc[naver_df['간식/디저트_카테고리'] == category]
        naver_value = naver_value['간식/디저트'].values
        naver_value = naver_value.tolist()
    elif category in dict1['전체']:
        naver_value = naver_df.loc[naver_df['전체_카테고리'] == category]
        naver_value = naver_value['전체'].values
        naver_value = naver_value.tolist()
    else:
        pass

    product_list = naver_value

    return category, product_list


def cv_commend_system(category, product_list, coupang):
    extracted_productid = []
    try:
        # 네이버 트렌드 상위 5개 키워드와 유사한 쿠팡 밀키트 추출
        for num in range(5):
            # TfidfVectorizer과 코사인 유사도를 이용한 추천시스템
            docs = coupang.loc[coupang['상세카테고리'] == category].상품.values
            docs = np.append(docs, product_list[num])
            vect = CountVectorizer()
            countvect = vect.fit_transform(docs)
            countvect_df = pd.DataFrame(countvect.toarray(), columns=sorted(vect.vocabulary_))

            # 위의 Data Frame 형태의 유사도를 계산
            cosine_matrix = cosine_similarity(countvect_df, countvect_df)
            productid = {}
            for i, c in enumerate(coupang.loc[coupang['상세카테고리'] == category].상품): productid[i] = c
            productid[len(productid)] = product_list[num]

            product = {}
            for i, c in productid.items(): product[c] = i

            idx = product[product_list[num]]
            sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            exproduct = productid[sim_scores[0][0]]
            if exproduct not in extracted_productid:
                if sim_scores[0][1] <= 0.01:
                    continue
                extracted_productid.append(exproduct)
                #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[0][1])

            else:
                exproduct = productid[sim_scores[1][0]]
                if exproduct not in extracted_productid:
                    if sim_scores[1][1] <= 0.01:
                        continue
                    extracted_productid.append(exproduct)
                    #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[1][1])

                else:
                    exproduct = productid[sim_scores[2][0]]
                    if exproduct not in extracted_productid:
                        if sim_scores[2][1] <= 0.01:
                            continue
                        extracted_productid.append(exproduct)
                        #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[2][1])

                    else:
                        exproduct = productid[sim_scores[3][0]]
                        if exproduct not in extracted_productid:
                            if sim_scores[3][1] <= 0.01:
                                continue
                            extracted_productid.append(exproduct)
                            #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[3][1])

                        else:
                            exproduct = productid[sim_scores[4][0]]
                            if exproduct not in extracted_productid:
                                if sim_scores[4][1] <= 0.01:
                                    continue
                                extracted_productid.append(exproduct)
                                #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[4][1])

        if len(extracted_productid) != 5:
            raise Exception()

    # 선택한 카테고리가 네이버 트렌드 데이터에 포함되어있지 않을 때
    except:
        try:
            coupang_sorted = coupang.loc[coupang['상세카테고리'] == category]
            coupang_sorted = coupang_sorted.astype({'리뷰수': int})
            coupang_sorted = coupang_sorted.sort_values('리뷰수', ascending=False)
            if len(extracted_productid) == 4:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 3:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 2:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 1:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 0:
                for number in range(5):
                    exproduct = coupang_sorted.iloc[number].상품
                    extracted_productid.append(exproduct)
                    #print(exproduct)


        except:
            pass

    return extracted_productid


def tfidf_commend_system(category, product_list, coupang):
    extracted_productid = []
    try:
        # 네이버 트렌드 상위 5개 키워드와 유사한 쿠팡 밀키트 추출
        for num in range(5):
            # TfidfVectorizer과 코사인 유사도를 이용한 추천시스템
            docs = coupang.loc[coupang['상세카테고리'] == category].상품.values
            docs = np.append(docs, product_list[num])
            vect = TfidfVectorizer()
            tfidfvect = vect.fit_transform(docs)
            tfidfvect_df = pd.DataFrame(tfidfvect.toarray(), columns=sorted(vect.vocabulary_))

            # 위의 Data Frame 형태의 유사도를 계산
            cosine_matrix = cosine_similarity(tfidfvect_df, tfidfvect_df)
            productid = {}
            for i, c in enumerate(coupang.loc[coupang['상세카테고리'] == category].상품): productid[i] = c
            productid[len(productid)] = product_list[num]

            product = {}
            for i, c in productid.items(): product[c] = i

            idx = product[product_list[num]]
            sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            exproduct = productid[sim_scores[0][0]]
            if exproduct not in extracted_productid:
                if sim_scores[0][1] <= 0.01:
                    continue
                extracted_productid.append(exproduct)
                #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[0][1])

            else:
                exproduct = productid[sim_scores[1][0]]
                if exproduct not in extracted_productid:
                    if sim_scores[1][1] <= 0.01:
                        continue
                    extracted_productid.append(exproduct)
                    #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[1][1])

                else:
                    exproduct = productid[sim_scores[2][0]]
                    if exproduct not in extracted_productid:
                        if sim_scores[2][1] <= 0.01:
                            continue
                        extracted_productid.append(exproduct)
                        #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[2][1])

                    else:
                        exproduct = productid[sim_scores[3][0]]
                        if exproduct not in extracted_productid:
                            if sim_scores[3][1] <= 0.01:
                                continue
                            extracted_productid.append(exproduct)
                            #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[3][1])

                        else:
                            exproduct = productid[sim_scores[4][0]]
                            if exproduct not in extracted_productid:
                                if sim_scores[4][1] <= 0.01:
                                    continue
                                extracted_productid.append(exproduct)
                                #print('현재 트렌드 -', docs[-1], '과 유사한 상품은', exproduct, sim_scores[4][1])

        if len(extracted_productid) != 5:
            raise Exception()

    # 선택한 카테고리가 네이버 트렌드 데이터에 포함되어있지 않을 때
    except:
        try:
            coupang_sorted = coupang.loc[coupang['상세카테고리'] == category]
            coupang_sorted = coupang_sorted.sort_values('가중치', ascending=False)
            if len(extracted_productid) == 4:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 3:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 2:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 1:
                for number in range(7):
                    exproduct = coupang_sorted.iloc[number].상품
                    if exproduct in extracted_productid:
                        pass
                    else:
                        extracted_productid.append(exproduct)
                        #print(exproduct)
                        if len(extracted_productid) == 5:
                            break

            elif len(extracted_productid) == 0:
                for number in range(5):
                    exproduct = coupang_sorted.iloc[number].상품
                    extracted_productid.append(exproduct)
                    #print(exproduct)
        except:
            pass

    return extracted_productid