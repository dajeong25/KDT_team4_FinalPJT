from django.shortcuts import render
import re
import pandas as pd
import numpy as np
import urllib.request
import pickle
import pytesseract
import cv2
from PIL import Image
from coupang_crawler import *
from . import models


def home(request, *args):
    # 쿠팡 크롤링 실행코드
    cg_url = 'https://www.coupang.com/np/categories/'
    user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
    coupang_items = get_coupang_items(cg_url, user_agt)

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
    product_url = 'https://www.coupang.com/vp/products/'
    image_productId = list(coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['상품id'].values)
    image_dataitemId = list(
        coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['data-item-id'].values)
    image_datavendorId = list(
        coupang_items.loc[coupang_items['상품명'].str.contains('곰곰|딜리조이|프렙')]['data-vendor-item-id'].values)

    for product in range(len(image_productId)):
        img_URL = product_url + str(image_productId[product]) + '/items/' + str(
            image_dataitemId[product]) + '/vendoritems/' + str(image_datavendorId[product])
        user_agt = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
        page = get_coupang_image_page(img_URL, user_agt)  # 쿠팡 이미지 페이지 추출
    image_link = get_coupang_images(page)  # 쿠팡 이미지 페이지에서 이미지 추출

    # 기존에 존재하는 이미지 제거
    deleteAllFiles('C:/coding/mealkitn/image/')

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
    gom_typo_list = gomgom_pre(extracted_text)

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

    # 요리재료_전처리(new column)
    data_items_drop['구성정보_전처리'] = np.NaN

    for idx in range(len(data_items_drop)):
        ingredient = data_items_drop.loc[idx, '구성정보']
        ingredients = p_c_preprocessing(ingredient)
        ingre_lists = p_c_preprocessing_plus(ingredients)
        # 요리재료_전처리(new column)에 추가
        data_items_drop.loc[idx, '구성정보_전처리'] = ', '.join(ingre_lists)

    # 만개의 레시피 재료 목록 read
    ingre_10000 = []
    with open('만개의레시피_재료_목록.txt', 'r') as f:
        lines = f.readlines()
        lines = str(lines).split(',')
        for line in lines:
            ingre_10000.append(line)

    # 쿠팡과 만개의레시피 일치하는 요리재료만 리스트로 만들고 데이터 프레임으로 수정
    ingre_10000s = ingredient_coupang(data_items_drop)
    for idx in range(len(ingre_10000s)):
        data_items_drop.loc[idx, '구성정보_전처리'] = ','.join(ingre_10000s[idx])

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

    # 알레르기 추가
    coupang['알레르기'] = allergy(coupang, '상품', '구성정보')
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

    # 요기요 추가
    # 요기요 데이터프레임 간단한 전처리
    yogiyodata = pd.read_csv('yogiyo_dataset.csv', index_col=0)
    yogiyodata.drop_duplicates(inplace=True)
    yogiyodata.reset_index(inplace=True)
    yogiyodata.drop('index', axis=1, inplace=True)

    # data의 menu_name 컬럼에서, 사전의 value값에 해당하는 text가 존재한다면 사전의 key를 원소롤 받음
    # 위 함수에서 뽑아낸 리스트를 시리즈로 변환 -> data dataframe에 추가
    yogiyo_list = dict_to_dataframe(yogiyodata, 'menu_name')
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
    coupang_list = dict_to_dataframe(coupang, '상품')
    coupang_series = pd.Series(coupang_list)
    coupang['카테고리_요기요'] = coupang_series
    coupang['요기요가격'] = coupang['카테고리_요기요'].apply(lambda x: func(x, price_mean))

    # 필요없는 컬럼을 제거, 시각화에 유리하게끔 yogiyo_price를 float -> int 로 형 변환
    # coupang.drop('menu_edited', axis = 1, inplace= True)
    coupang.요기요가격 = coupang.요기요가격.astype(int)
    coupang.정가 = coupang.정가.astype(int)
    coupang.판매가 = coupang.판매가.astype(int)

    # db에 저장
    db = add_new_items(coupang)

    return render(request, 'index.html')


def mk_recs(request):
    return render(request, 'mealkit_recs.html')

# 밀키트 추천
def mk_recs_result(request):
    # 네이버 트랜드 불러오기
    naver_df = pd.read_csv('2023-1-2edited_navertrend.csv', index_col=0)
    naver_df.replace('밀키트', '', regex=True, inplace=True)

    naver_df['전체_카테고리'] = naver_df['전체'].apply(lambda x: category_name(x))
    naver_df['밥/죽_카테고리'] = naver_df['밥/죽'].apply(lambda x: category_name(x))
    naver_df['찌개/국_카테고리'] = naver_df['찌개/국'].apply(lambda x: category_name(x))
    naver_df['면/파스타_카테고리'] = naver_df['면/파스타'].apply(lambda x: category_name(x))
    naver_df['구이_카테고리'] = naver_df['구이'].apply(lambda x: category_name(x))
    naver_df['볶음/튀김_카테고리'] = naver_df['볶음/튀김'].apply(lambda x: category_name(x))
    naver_df['조림/찜_카테고리'] = naver_df['조림/찜'].apply(lambda x: category_name(x))
    naver_df['간식/디저트_카테고리'] = naver_df['간식/디저트'].apply(lambda x: category_name(x))
    
    # 밀키트 추천
    category, product_list = coupang_naver_categories('생선탕')
    tfidf_commend = tfidf_commend_system(category, product_list)
    return tfidf_commend


