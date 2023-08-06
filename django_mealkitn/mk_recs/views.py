import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import numpy as np
from collections import Counter
import urllib.request
import pickle
import pytesseract
import cv2
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from konlpy.tag import Okt
from django.shortcuts import render, redirect
from coupang_crawler import *
from collaborative_filtering import *
from mk_recs.models import *
from user_datas.models import User
import rcp_crawler
from gensim.models import Word2Vec
import ast


def run(request, *args):
    # categoryId 기반 카테고리 대분류 정의
    category = {'502483': 1, '502484': 2, '502485': 3, '502486': 4, '502487': 5, '502490': 6, '502491': 7}
    category_keys = list(category.keys())

    # 상품 리스트를 담을 빈 dataframe 생성
    item_list = ['카테고리명', '상품id', 'data-item-id', 'data-vendor-item-id', '상품', '상품명', '정가',
                 '할인율', '판매가', '100g당_가격', '별점', '리뷰수', '품절여부']
    coupang_items = pd.DataFrame(columns=item_list)

    # 쿠팡 상품 리스트 추출
    cg_url = 'https://www.coupang.com/np/categories/'
    user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"

    for categoryId in category_keys:
        page_num = page_number(categoryId, user_agt)  # 해당 카테고리의 페이지 마지막 페이지 번호 추출
        for page in range(1, page_num + 1):
            URL = cg_url + categoryId + "?page=" + str(page) + '&filter=1%23attr_12406%2419087%40DEFAULT'

            # 페이지 호출
            cg_soup_base = get_coupang_item(URL, user_agt)
            cg_soup2 = cg_soup_base.select("#productList li")

            # 상품 리스트 추출 및 전처리
            for cg_soup in cg_soup2:
                coupang_item = p_list_preprocessing(cg_soup, category_number(categoryId,
                                                                             category))
                coupang_items = pd.concat([coupang_items, coupang_item], ignore_index=True)

    # 중복행 삭제
    coupang_items.drop_duplicates('상품id', inplace=True)
    coupang_items.reset_index(drop=True, inplace=True)

    # 2차 실행부
    data_items = pd.DataFrame(columns=['상품id', '구성정보'])

    for idx in range(len(coupang_items)):
        # 각 상품 상세페이지 호출
        product_url = 'https://www.coupang.com/vp/products/'
        productId = coupang_items.loc[idx, '상품id']  # data-product-id
        itemsId = coupang_items.loc[idx, 'data-item-id']  # data-item-id
        vendorItemId = coupang_items.loc[idx, 'data-vendor-item-id']  # data-vendor-item-id
        URL = product_url + str(productId) + '/items/' + str(itemsId) + '/vendoritems/' + str(vendorItemId)
        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"

        pdt_soup = get_coupang_item(URL, user_agt)

        # '구성정보' 추출
        p_c_page = pdt_soup.find('body').children
        data_item = p_c_extraction(productId, p_c_page)
        data_items = pd.concat([data_items, data_item], ignore_index=True)

    # 구성정보 페이지에서 전체 이미지 링크 추출 -> 리스트화
    image_link = []
    product_url = 'https://www.coupang.com/vp/products/'
    image_productId = list(coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['상품id'].values)
    image_dataitemId = list(coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['data-item-id'].values)
    image_datavendorId = list(
        coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['data-vendor-item-id'].values)

    for product in range(len(image_productId)):
        img_URL = product_url + str(image_productId[product]) + '/items/' + str(
            image_dataitemId[product]) + '/vendoritems/' + str(image_datavendorId[product])
        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"

        page = get_coupang_image(img_URL, user_agt)

        link_start = []
        for text in re.finditer('//thumbnail', page):
            link_start.append(text.start())

        link_end = []
        for text in re.finditer('g"}],', page):
            link_end.append(text.start())

        for num in range(len(link_start)):
            image_link.append('https:' + page[int(link_start[num]):int(link_end[num]) + 1])

    # 기존에 존재하는 이미지 제거
    deleteAllFiles('C:/coding/multicampus/image/')

    # 상세페이지에서 전체 이미지 저장
    count = 0
    for url in image_link:
        count += 1
        path = "C:/coding/multicampus/image/mealkitn" + (str(count) + ".jpg")
        urllib.request.urlretrieve(url, path)
    # 이미지에서 텍스트 추출 코드 with pytesserect
    extracted_text = []
    for num in range(1, count + 1):
        file = 'C:/coding/multicampus/image/mealkitn' + str(num) + '.jpg'
        image = Image.open(file)
        if image.size[1] >= 4000:
            croppedImage = image.crop((0, 2000, image.size[0], 3350))
            croppedImage.save('C:/coding/multicampus/image/mealkitn_cropped' + str(num) + '.jpg')
            file = 'C:/coding/multicampus/image/mealkitn_cropped' + str(num) + '.jpg'
            image = cv2.imread(file)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb_image, lang='kor')
            extracted_text.append(text)
        else:
            image = cv2.imread(file)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb_image, lang='kor')
            extracted_text.append(text)

    # 결측값 중 곰곰 밀키트 구성정보만 테서렉트로 이미지 텍스트 추출 후,
    # 오타 교정 위해 전처리 작업
    gomgom_text = [i for i in extracted_text if "구성되어 있어요" in i]
    gomgom = []
    for i in gomgom_text:
        sentence = re.sub('(<([^>]+)>)', '', i)
        sentence = re.sub('[^ ㄱ-ㅎㅏ-ㅣ각-힣]', ' ', sentence)
        gomgom.append(sentence)

    gom_typo_list = []

    for gom in gomgom:
        gg = gom.split(' ')
        gom_typo = []
        for i in range(len(gg)):
            if len(gg[i]) > 1:
                gom_typo.append(gg[i])
            else:
                pass
        gom_typo_list.append(gom_typo)

    # gom_type_list 피클화
    with open('gom_typo.pkl', 'wb') as f:
        pickle.dump(gom_typo_list, f)

    # 곰곰 상품id 추출
    gomgom_productId = list(coupang_items.loc[coupang_items['상품'].str.contains('곰곰')]['상품id'].values)

    # 곰곰 이미지에서 추출한 구성정보 list
    with open('gom_typo_fix.pkl', 'rb') as fr:
        gomgom_data = pickle.load(fr)

    # 상품id + 구성정보
    gomgom_df = pd.DataFrame(zip(gomgom_productId, gomgom_data), columns=['상품id', '구성정보'])

    ### 곰곰 결측치 채우기
    for i in range(len(gomgom_df)):
        idx = data_items[data_items['상품id'] == gomgom_productId[i]].index.values
        gomgom = gomgom_df.loc[i, '구성정보']
        data_items.loc[idx, '구성정보'] = ', '.join(gomgom)

    # 요리재료내용 결측치 삭제
    data_items_drop = data_items[data_items.loc[:, '구성정보'].isnull()].index
    data_items_drop = data_items.drop(data_items_drop)
    data_items_drop.reset_index(drop=True, inplace=True)
    data_items_drop.info()

    # 요리재료_전처리(new column)
    data_items_drop['구성정보_전처리'] = np.NaN

    for idx in range(len(data_items_drop)):
        ingredient = data_items_drop.loc[idx, '구성정보']
        ingredients = p_c_preprocessing(ingredient)
        ingredients = ingredients.split(',')

        ingre_lists = []
        for i in ingredients:
            i = i.strip()
            replace_list = ['칼국수 면', '다진 마늘', '대구 곤이', '명태 곤이']
            if i in replace_list:
                i = i.split(' ')
                i = ''.join(i)

            i = i.split(' ')
            if i[-1] == '등':  # 등을 그대로 두는 것이 나을지 지금처럼 빼는게 나을지..?
                i.pop(-1)
                i = ' '.join(i)
                ingre_lists.append(i)
            else:
                i = ' '.join(i)
                ingre_lists.append(i)

        # null값 제거
        ingre_lists = list(filter(None, ingre_lists))

        # 요리재료_전처리(new column)에 추가
        data_items_drop.loc[idx, '구성정보_전처리'] = ', '.join(ingre_lists)

    # 만개의 레시피 재료 목록 read
    ingre_10000 = []
    with open('만개의레시피_재료_목록.txt', 'r') as f:
        lines = f.readlines()
        lines = str(lines).split(',')
        for line in lines:
            ingre_10000.append(line)

    # 쿠팡 재료 목록 생성
    ingre_coupang_list = data_items_drop['구성정보_전처리'].dropna()
    ingre_coupang_lists = ingre_coupang_list.to_list()

    ingre_coupangs = []
    for item in ingre_coupang_lists:
        ingre_coupang = []
        items = item.split(', ')
        for i in items:
            ingre_coupang.append(i)
        ingre_coupangs.append(ingre_coupang)

    # 쿠팡에만 있는 재료만
    ingre_not_10000 = []
    for items in ingre_coupangs:
        not_10000 = []
        for i in items:
            if not i in ingre_10000 and i != '':
                not_10000.append(i)
        ingre_not_10000.append(not_10000)

    # 일치하는 재료 리스트로 만들고 데이터 프레임으로 수정
    ingre_10000s = []
    for items in ingre_coupangs:
        ingre_10000_list = []
        for i in items:
            if i in ingre_10000:
                ingre_10000_list.append(i)
        ingre_10000s.append(ingre_10000_list)

    for idx in range(len(ingre_10000s)):
        data_items_drop.loc[idx, '구성정보_전처리'] = ','.join(ingre_10000s[idx])
    data_items_drop.head()

    # 쿠팡 상품 리스트와 구성정보 병합
    coupang = pd.merge(coupang_items, data_items_drop)
    coupang.to_csv('coupang_filtering.csv')

    # 결측치 제거
    coupang_null_list = coupang.loc[coupang['구성정보_전처리'] == ''].index
    coupang.drop(coupang_null_list, inplace=True)

    # 품절 상품 제거
    coupang = coupang.loc[coupang['품절여부'] == 0]

    coupang['상세카테고리'] = coupang['상품'].apply(lambda x: category_name(x))
    coupang.reset_index(inplace=True)

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

    allergy_list = allergy(coupang, '상품', '구성정보')

    allergy_series = pd.Series(allergy_list)
    coupang['알레르기'] = allergy_series
    coupang.drop(['index'], axis=1, inplace=True)

    # calcuate input parameters
    coupang['별점'] = coupang['별점'].astype('float')
    coupang['리뷰수'] = coupang['리뷰수'].astype('float')
    C = np.mean(coupang['별점'])
    m = np.percentile(coupang['리뷰수'], 70)
    vote_count = coupang[coupang['리뷰수'] >= m]
    R = coupang['별점']
    v = coupang['리뷰수']

    coupang['가중치'] = weighted_rating(v, m, R, C)

    # 요기요 - 쿠팡 밀키트 가격 비교에 사용할 사전
    yogiyo_dict = {'곱창전골': ['곱창전골', '곱창 전골'],
                   '나베': ['나베'],
                   '샤브샤브': ['샤브샤브'],
                   '부대전골': ['부대전골'],
                   '불고기전골': ['불고기전골', '불고기 전골', '불고기 버섯전골'],
                   '낙곱새': ['낙곱새'],
                   '만두전골': ['만두전골', '만두 전골'],
                   '스키야키': ['스키야키'],

                   '순두부찌개': ['순두부찌개', '순두부 찌개'],
                   '된장찌개': ['된장찌개'],
                   '청국장찌개': ['청국장'],
                   '고추장찌개': ['고추장찌개'],
                   '김치찌개': ['김치찌개'],
                   '부대찌개': ['부대찌개'],
                   '돼지고기짜글이': ['삼겹짜글이', '돼지고기 김치짜글이', '돼지짜글이', '돼지김치짜글이'],

                   '투움바파스타': ['투움바 파스타', '투움바파스타'],
                   '크림파스타': ['크림파스타', '크림 파스타', '크림 빠네 파스타'],
                   '로제파스타': ['로제파스타', '로제 파스타'],
                   '봉골레파스타': ['봉골레'],
                   '해장파스타': ['해장파스타', '해장 파스타'],
                   '토마토파스타': ['토마토파스타', '토마토 파스타'],
                   '냉파스타': ['냉파스타'],
                   '바지락파스타': ['바지락 파스타'],
                   '새우파스타': ['새우 파스타', '쉬림프파스타', '쉬림프 파스타'],
                   '오일파스타': ['오일파스타', '오일 파스타'],

                   '메기탕': ['메기매운탕', '메기 맑은', '메기 매운'],
                   '대구탕': ['대구탕', '대구지리', '대구매운', '대구뽈탕', '대구머리', '대구 맑은', '대구 매운'],
                   '아구탕': ['아구탕', '아구 매운탕', '아구 지리탕', '아구지리', '아귀 매운', '아귀 맑은'],
                   '동태탕': ['동태탕', '동태내장탕'],
                   '우럭탕': ['우럭탕', '우럭매운탕', '우럭 매운탕', '우럭 통매운탕', '우럭맑은지리'],

                   '조개탕': ['조개탕'],
                   '낙지탕': ['연포', '낙지탕'],
                   '누룽지탕': ['해물누룽지탕', '해물 누룽지탕'],
                   '꽃게탕': ['꽃게탕'],
                   '홍합탕': ['홍합탕', '홍합짬뽕탕'],
                   '새우탕': ['새우탕'],
                   '해물탕': ['해물탕', '해물 뚝배기', '해물뚝배기'],

                   '스테이크': ['스테이크'],

                   '조기구이': ['조기구이', '제주도 참조기'],
                   '갈치구이': ['갈치구이'],
                   '고등어구이': ['고등어구이', '고등어 구이'],

                   '감바스': ['감바스'],

                   '미역국': ['미역국'],
                   '순대국밥': ['순대국밥'],
                   '육개장': ['육개장'],
                   '떡국': ['떡국'],
                   '소고기무국': ['소고기무국', '소고기 무국'],

                   '닭발': ['닭발'],
                   '닭갈비': ['닭갈비'],
                   '백숙': ['백숙'],
                   '닭볶음탕': ['닭볶음'],

                   '우동': ['야끼우동', '청춘우동', '짬뽕우동', '포장마차우동', '사누끼우동', '사누끼 우동', '구르미우동', '온우동', '볶음우동', '어묵우동', '즉석우동',
                          '얼큰우동'],
                   '탄탄면': ['탄탄면'],

                   '콩불': ['콩불'],
                   '퀘사디아': ['퀘사디아'],
                   '갈치조림': ['갈치조림', '갈치 조림'],
                   '순대볶음': ['순대볶음', '순대 볶음'],

                   '고추잡채': ['고추잡채', '고추 잡채'],
                   '숙주볶음': ['숙주볶음', '숙주 볶음'],

                   '꼬막무침': ['꼬막 무침', '꼬막무침'],

                   '알탕': ['알탕', '알곤이탕'],

                   '칼국수': ['칼국수'],
                   '비빔국수': ['비빔국수'],
                   '쌀국수': ['쌀국수'],
                   '초계국수': ['초계국수'],

                   '떡볶이': ['떡볶이', '떡볶잉'],

                   '아귀찜': ['아귀찜', '아구찜'],
                   '대구뽈찜': ['대구뽈찜', '대구왕뽈찜'],
                   '술찜': ['술찜'],
                   '묵은지찜': ['묵은지 김치찜（돼지）', '돼지김치찜', '삼겹 김치찜', '흑돈 김치찜'],
                   '해물찜': ['해물찜', '보일링쉬림프', '보일링 씨푸드', '보일링씨푸드', '갈릭버터쉬림프'],
                   '마라탕': ['마라탕', '마라우육면'],

                   '도가니탕': ['도가니탕'],
                   '어묵탕': ['어묵탕', '오뎅탕'],
                   '김치찜': ['묵은지 김치찜（돼지）', '돼지김치찜', '삼겹 김치찜', '흑돈 김치찜'],
                   '양장피': ['양장피'],
                   '야끼만두': ['야끼만두', '야끼 만두']
                   }

    # 요기요 데이터프레임 간단한 전처리
    yogiyodata = pd.read_csv('yogiyo_dataset.csv', index_col=0)
    yogiyodata.drop_duplicates(inplace=True)
    yogiyodata.reset_index(inplace=True)
    yogiyodata.drop('index', axis=1, inplace=True)

    # data의 menu_name 컬럼에서, 사전의 value값에 해당하는 text가 존재한다면 사전의 key를 원소롤 받음
    # 위 함수에서 뽑아낸 리스트를 시리즈로 변환 -> data dataframe에 추가
    yogiyo_list = dict_to_dataframe(yogiyodata, yogiyo_dict, 'menu_name')
    yogiyo_series = pd.Series(yogiyo_list)
    yogiyodata['menu_edited'] = yogiyo_series

    # 유사한 음식이 묶인 menu_edited 컬럼을 기준으로 그룹화후, 평균 price 값 추출 및 반올림
    grouped_data = yogiyodata.groupby(yogiyodata['menu_edited'])
    price_mean = grouped_data.price.mean()
    roundd = price_mean.values.round(-2)
    price_mean = price_mean.round(-2)

    # menu_edited의 결측값(yogiyo_dict의 value와 일치하는 값이 없는 메뉴들)제거
    price_mean = price_mean.drop([''])

    commalist = []
    for i in range(len(price_mean)):
        if ',' in list(price_mean.index)[i]:
            commalist.append(list(price_mean.index)[i])

    for i in commalist:
        price_mean.drop(i, inplace=True)

    # 만들어 놓은 함수를 활용하여, coupang의 '상품' 컬럼에서, 사전의 value값에 해당하는 text가 존재한다면 사전의 key를 원소롤 받음
    coupang_list = dict_to_dataframe(coupang, yogiyo_dict, '상품')
    coupang_series = pd.Series(coupang_list)
    coupang['카테고리_요기요'] = coupang_series
    coupang['요기요가격'] = coupang['카테고리_요기요'].apply(lambda x: func(x, price_mean))

    # 필요없는 컬럼을 제거, 시각화에 유리하게끔 yogiyo_price를 float -> int 로 형 변환
    # coupang.drop('menu_edited', axis = 1, inplace= True)
    coupang.요기요가격 = coupang.요기요가격.astype(int)
    coupang.정가 = coupang.정가.astype(int)
    coupang.판매가 = coupang.판매가.astype(int)

    add_new_items(coupang)

    coupang.to_csv('coupang_filtering.csv')

    return render(request, 'index.html')


def home(request):
    return render(request, 'index.html')


# 밀키트 추천
def mk_recs_result(request, *args):
    # 네이버 트랜드 불러오기
    naver_df = pd.read_csv('2023-1-13edited_navertrend.csv', index_col=0)
    naver_df.replace('밀키트', '', regex=True, inplace=True)
    naver_df['전체_카테고리'] = naver_df['전체'].apply(lambda x: category_name(x))
    naver_df['밥/죽_카테고리'] = naver_df['밥/죽'].apply(lambda x: category_name(x))
    naver_df['찌개/국_카테고리'] = naver_df['찌개/국'].apply(lambda x: category_name(x))
    naver_df['면/파스타_카테고리'] = naver_df['면/파스타'].apply(lambda x: category_name(x))
    naver_df['구이_카테고리'] = naver_df['구이'].apply(lambda x: category_name(x))
    naver_df['볶음/튀김_카테고리'] = naver_df['볶음/튀김'].apply(lambda x: category_name(x))
    naver_df['조림/찜_카테고리'] = naver_df['조림/찜'].apply(lambda x: category_name(x))
    naver_df['간식/디저트_카테고리'] = naver_df['간식/디저트'].apply(lambda x: category_name(x))
    naver_df.to_csv('2023-1-13edited_navertrend.csv')

    # 밀키트 추천
    coupang_df = pd.read_csv('coupang_filtering.csv')
    # 최근에 추가된 유저 데이터에서 알레르기, 카테고리, 선호, 비선호 가져오기
    user = User.objects.values_list().order_by('-id')[0]
    user_name = user[1]
    user_alleray = user[7]
    user_category2 = user[9]

    # 알레르기 유발 성분 포함 밀키트 제거
    user_value = ast.literal_eval(user_alleray)
    coupang_df['알레르기'] = coupang_df['알레르기'].fillna('-')

    idxlist = []
    for i in range(len(coupang_df.알레르기.values)):
        value = coupang_df.알레르기.values[i].split(', ')
        total_len = len(user_value) + len(value)
        merge_len = len(set(user_value + value))

        if total_len != merge_len:
            a = coupang_df.알레르기.index[i]
            idxlist.append(a)

    for idx in idxlist:
        coupang_df.drop(index=idx, inplace=True)

    coupang_df.to_csv('coupang_data.csv')  # 알레르기 포함된 밀키트 제거
    coupang_df_copy = coupang_df.copy()    # 협업필터링 추천을 위한 df 저장

    # 상세카테고리 상품만 필터링
    coupang_df = coupang_df.loc[coupang_df.상세카테고리 == user_category2]
    category, product_list = coupang_naver_categories(user_category2, naver_df)
    tfidf_commend = tfidf_commend_system(category, product_list, coupang_df)

    mealkit_detail_list = []
    mealkit_id = []
    for mealkit_X in tfidf_commend:
        mealkitDB = CoupangMealkit.objects.get(product=mealkit_X)

        mealkitDB_name = mealkitDB.product
        mealkitDB_discounted_price = mealkitDB.discounted_price
        mealkitDB_price = mealkitDB.full_price
        mealkitDB_yogiyo_price = mealkitDB.yogiyo_price
        mealkitDB_ratings = int(round(mealkitDB.ratings))
        mealkitDB_edited_ingredient = mealkitDB.edited_ingredient
        mealkitDB_link = 'https://www.coupang.com/vp/products/' + mealkitDB.product_id
        mealkit_id.append(mealkitDB.product_id)

        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
        headers = {"User-Agent": user_agt, "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
        res = requests.get(mealkitDB_link, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        mealkitDB_image_link = "http:" + soup.select_one('#repImageContainer > img')['src']

        mealkit_detail_list.append({'name': mealkitDB_name,
                                    'discounted_price': mealkitDB_discounted_price,
                                    'price': mealkitDB_price,
                                    'yogiyo_price': mealkitDB_yogiyo_price,
                                    'ratings': mealkitDB_ratings,
                                    'edited_ingredient': mealkitDB_edited_ingredient,
                                    'link': mealkitDB_link,
                                    'image_link': mealkitDB_image_link})
    # print(mealkit_detail_list)

    # 협업필터링
    # name, 상세카테고리, 점수
    # 1. 선호 4, 비선호 2, 기본3 카테고리 점수화
    # 2. 가중치 정렬 카테고리 기반 추천 or 랜덤?
    cf = User.objects.all().values()
    cf_df = pd.DataFrame(list(cf))

    cf_pos = cf_df[['user_name', 'pos_category']]
    cf_pos.rename(columns={'pos_category': 'category'}, inplace=True)
    cf_pos['weighting'] = 4

    cf_neg = cf_df[['user_name', 'neg_category']]
    cf_neg.rename({'neg_category': 'category'})
    cf_neg.rename(columns={'neg_category': 'category'}, inplace=True)
    cf_neg['weighting'] = 2

    cf = pd.concat([cf_pos, cf_neg])

    item = collaborative_df(cf)
    try:
        get_category = get_item(item, user_category2)
        coupang_index = coupang_df[
            (coupang_df.상세카테고리 == get_category[0]) | (coupang_df.상세카테고리 == get_category[1])].index
        coupang_id = list(coupang_df.loc[coupang_index, '상품id'])
        for i in coupang_id:
            if i in mealkit_id:
                coupang_id.remove(i)
    except:
        categorys = coupang_category.objects.values_list('category_keys', flat=True).order_by(
            'category_keys').distinct()
        coupang_id = []
        while len(coupang_id) < 2:
            random_choice = random.sample(list(categorys), 2)
            coupang_index = coupang_df_copy[
                (coupang_df_copy.상세카테고리 == random_choice[0]) | (coupang_df_copy.상세카테고리 == random_choice[1])].index

            coupang_id = list(coupang_df_copy.loc[coupang_index, '상품id'])
            for i in coupang_id:
                if i in mealkit_id:
                    coupang_id.remove(i)

    if len(coupang_id) >= 4:
        coupang_id = random.sample(list(coupang_id), 4)

    cf_list = []
    for key in coupang_id:
        db = CoupangMealkit.objects.get(product_id=key)
        db_name = db.product
        db_discounted_price = db.discounted_price
        db_price = db.full_price
        db_ratings = int(round(db.ratings))
        db_yogiyo_price = db.yogiyo_price
        db_link = 'https://www.coupang.com/vp/products/' + db.product_id

        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
        headers = {"User-Agent": user_agt, "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
        res = requests.get(db_link, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        db_image_link = "http:" + soup.select_one('#repImageContainer > img')['src']

        cf_list.append({'name': db_name,
                        'discounted_price': db_discounted_price,
                        'price': db_price,
                        'yogiyo_price': db_yogiyo_price,
                        'ratings': db_ratings,
                        'link': db_link,
                        'image_link': db_image_link})

    context = {'user_name': user_name, 'mealkit_detail_list': mealkit_detail_list, 'cf_list': cf_list}
    return render(request, 'product.html', context)


def rcp_recs_result(request):
    # 가중치까지 계산한 df을 불러옴
    ckg = pd.read_csv('ckg_weighted.csv', index_col=0)

    # word2vec 사용을 위한 데이터 포멧 변경
    ckg['요리재료_전처리'] = ckg['요리재료_전처리'].apply(lambda x: x.split(', '))  # 1차원 list 형태

    # coupang 상세정보
    naver_df = pd.read_csv('2023-1-13edited_navertrend.csv', index_col=0)
    coupang_df = pd.read_csv('coupang_data.csv')

    # 최근에 추가된 유저 데이터에서 알레르기, 카테고리, 선호, 비선호 가져오기
    user = User.objects.values_list().order_by('-id')[0]
    user_name = user[1]
    user_pos = user[5]
    user_neg = user[6]
    user_allergy = user[7]
    user_category1 = user[8]
    user_category2 = user[9]

    # 알레르기 유발 성분 포함 밀키트 제거
    user_value = ast.literal_eval(user_allergy)

    category, product_list = coupang_naver_categories(user_category2, naver_df)
    tfidf_commend = tfidf_commend_system(category, product_list, coupang_df)

    mealkit_detail_list = []
    for mealkit_X in tfidf_commend:
        mealkitDB = CoupangMealkit.objects.get(product=mealkit_X)

        mealkitDB_name = mealkitDB.product
        mealkitDB_discounted_price = mealkitDB.discounted_price
        mealkitDB_price = mealkitDB.full_price
        mealkitDB_yogiyo_price = mealkitDB.yogiyo_price
        mealkitDB_ratings = int(round(mealkitDB.ratings))
        mealkitDB_edited_ingredient = mealkitDB.edited_ingredient
        mealkitDB_link = 'https://www.coupang.com/vp/products/' + mealkitDB.product_id

        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
        headers = {"User-Agent": user_agt, "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
        res = requests.get(mealkitDB_link, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        mealkitDB_image_link = "http:" + soup.select_one('#repImageContainer > img')['src']

        mealkit_detail_list.append({'name': mealkitDB_name,
                                    'discounted_price': mealkitDB_discounted_price,
                                    'price': mealkitDB_price,
                                    'yogiyo_price': mealkitDB_yogiyo_price,
                                    'ratings': mealkitDB_ratings,
                                    'edited_ingredient': mealkitDB_edited_ingredient,
                                    'link': mealkitDB_link,
                                    'image_link': mealkitDB_image_link})

    # coupang에서 input 값의 df 추출
    coupang_df['상품_전처리'] = coupang_df['상품']
    man_list = []
    for i in range(len(mealkit_detail_list)):
        name = mealkit_detail_list[i].get('name')

        product_df = coupang_df.loc[coupang_df['상품'] == name]
        num = coupang_df.index[(coupang_df['상품'] == name)].tolist()[0]

        # input 상품명 및 구성정보 전처리 후 list로 추출
        product_list, ingredients_list = rcp_crawler.get_product_list(num, product_df)

        # input 상품의 만개의 레시피와 일치하는 카테고리만 추출 {dict}
        ckg_key_value = rcp_crawler.get_product_category(num, coupang_df)

        # input 상품명과 일치하는 레시피만 추출
        recipe_df = rcp_crawler.get_recipe_df(ckg, product_list, ckg_key_value)
        #if len(recipe_df) == 0:  # 일치하는 레시피가 아예 없는 경우 에러발생
        #   raise
        recipe_index = recipe_df.index.tolist()

        # 일치하는 만개의 레시피 요리재료명 통일
        word = rcp_crawler.get_recipe_pre(recipe_index, recipe_df)

        # 통일한 요리재료명 업데이트
        recipe_df['요리재료_전처리'] = word

        # word list의 가장 마지막에 input 구성정보 추가
        word.append(ingredients_list)

        # word2vec 모델 생성
        wv_model = Word2Vec(sentences=word, vector_size=100, window=20, min_count=1, sg=1, batch_words=1000, alpha=0.03)

        # 레시피 별 벡터 평균 추출
        wordvec = [rcp_crawler.vectors(x, wv_model) for x in word]

        # 코사인유사도 계산
        w2v_sim = cosine_similarity(wordvec, wordvec)

        # 코사인유사도 행렬에서 input 상품의 인덱스 저장
        mealkit_index = len(w2v_sim) - 1

        # input 상품과 유사한 레시피 df 추출
        sim_mealkit_df = rcp_crawler.wv_mealkit(w2v_sim, mealkit_index, 5, recipe_df)
        # print(list(sim_mealkit_df.레시피일련번호))

        i = list(sim_mealkit_df.레시피일련번호)
        for idx in range(len(i)):
            data = ckg.loc[ckg.레시피일련번호 == i[idx]]
            data_id = str(data.레시피일련번호.values)
            data_name = str(data.레시피제목.values)
            data_description = data.요리소개.values
            data_ingredient = str(data.요리재료_전처리.values)
            data_allergy = str(data.알레르기.values)

            rcp_crawler.add_new_recipes(data, data_id, data_name, data_description, data_ingredient, data_allergy)

            pattern_punctuation = re.compile(r'[^\w\s]')
            data_id_edited = pattern_punctuation.sub('', data_id)

            # 레시피에만 있는 재료 추출
            recipe_ingredients = rcp_crawler.get_recipe_only(ingredients_list, sim_mealkit_df)
            # recipe_ingredients = list(filter(None, recipe_ingredients))  # input 구성정보와 요리재료가 모두 일치(빈리스트)하면 제외
            data_filter_ingredient = ','.join(recipe_ingredients[idx])

            man_list.append({'data_id': data_id,
                            'data_name': str(data_name).replace("['",'').replace("']",''),
                            'data_description': str(data_description).replace("['",'').replace("']",''),
                            'data_ingredient': data_ingredient,
                            'data_allergy': data_allergy,
                            'data_link': 'https://www.10000recipe.com/recipe/'+data_id_edited,
                            'data_filter_ingredient': data_filter_ingredient})

    context = {'user_name': user_name, 'mealkit_detail_list': mealkit_detail_list, 'man_list': man_list}
    return render(request, 'recipe.html', context)

