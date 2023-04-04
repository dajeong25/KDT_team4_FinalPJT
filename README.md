# 소비자를 위한 밀키트 추천 시스템
[멀티잇]데이터 시각화&amp;분석 취업캠프(Python) 최종 프로젝트
> 밀키트엔(김태현, 두민규, 허다정)

- 타겟<br>
  1. 밀키트 추천을 받고 싶은 소비자 
  2. 밀키트를 더욱 맛있게 먹고 싶은 소비자
  3. 외식비용과 밀키트 비용을 비교하고 싶은 소비자

- 목표<br>
**(개발자)** <br>
  머신러닝 알고리즘을 활용한 밀키트&레시피 추천,<br>
  가격비교 서비스 제공 <br>
**(소비자)** <br>
  간단한 재료 추가로 더욱 맛있게 먹기, <br>
  외식&밀키트 비용을 비교해서 합리적인 소비하기<br>

- 데이터 선정 기준
: 가정간편식(HMR)이 아닌 밀키트(Meal Kit).<br>
  음식을 뜻하는 Meal과 세트를 뜻하는 Kit의 합성어<br>
  음식에 필요한 손질된 재료와 양념, 조리법이 세트로 구성된 제품. 즉,‘음식 세트’

- 밀키트 사이트 : 쿠팡(https://www.coupang.com/)
- 활용 데이터 : 네이버데이터랩, 만개의 레시피, 요기요


## 시나리오
- 밀키트 추천
  1. 알레르기 제외 & 카테고리 선택
  2. 네이버 트렌드 기준, 밀키트 추천
  3. 설문조사로 고객 정보 수집 후 협업필터링의 고객맞춤 밀키트 추천

- 레시피 추천
  1. 선택한 밀키트의 구성정보와 만개의 레시피 요리재료 비교 후, 레시피 추천
  2. 밀키트와 레시피에만 있는 재료, 구매 링크 연결(소비유도)

- 비용계산
  1. 요기요 기준, 선택한 밀키트와 평균 메뉴 외식 비용 비교


## 추천시스템 구현
- 밀키트 추천<br>
  * 기준 : 밀키트 인기검색어 (소비 트렌드)와 상품명 유사도, 쿠팡의 리뷰수 & 별점
  * 아이텀기반 밀키트 추천 : TfidfVectorizer 사용
  * 협업필터링 밀키트 추천 : 설문조사로 받은 고객 데이터 기반 코사인 유사도 사용
      > 실제 데이터를 구할 수 없는 한계로 가상의 고객데이터 활용. 하지만 정확한지 테스트 불가로 다양한 모델을 구현해보지 못함.
 
- 레시피 추천<br>
  * 기준 : 쿠팡의 구성정보와 레시피의 요리재료내용 유사도
  * 아이텀기반 추천 : Word2Vec의 skip-gram과 코사인 유사도 사용


## 한계점
- 구현 속도<br>
  : 크롤링으로 인한 러닝 시간이 상당히 오래 걸림
- 추천시스템의 정확도<br>
  : 고객데이터의 부재 <br>
  : 데이터를 구성하고 전처리하는 과정에서 결측치 및 불투명한 데이터를 수작업으로 채워넣어야 하는데 이를 위한 시간과 인력이 부족함.
- 알레르기<br>
  : 원재료명에 포함되어 있는 알레르기 유발식품 정보만 추출 불가능.<br>
    특정 알레르기를 유발할 수 있는 구성식품이 들어가있음에도 걸러주지 못할 가능성 존재.
- 상품의 다양성<br>
  : 크롤링과 데이터 전처리의 시간의 한계로 쿠팡에서만 데이티를 추출하였기 때문에 선택의 폭이 좁음
- 웹페이지<br>
  : 카테고리 선택과 레시피 추천 페이지는 시간 부족으로 구현 미흡
