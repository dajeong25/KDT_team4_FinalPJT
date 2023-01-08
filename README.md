# KDT_team4_FinalPJT
[멀티잇]데이터 시각화&amp;분석 취업캠프(Python) 최종 프로젝트
> 밀키트엔(김태현, 두민규, 허다정)

- 주제 : 소비자를 위한 밀키트 추천 시스템

- 타겟<br>
  1. 밀키트 추천을 받고 싶은 소비자 
  2. 밀키트를 더욱 맛있게 먹고 싶은 소비자
  3. 외식비용과 밀키트 비용을 비교하고 싶은 소비자

- 목표<br>
(개발자)<br>
밀키트를 구매하려는 소비자에게 밀키트&레시피 추천,<br>
가격비교 서비스 제공 <br>
(소비자) <br>
추천 받은 밀키트를 간단한 재료 추가로 더욱 맛있게 먹기, <br>
외식&밀키트 비용을 비교해서 합리적인 소비하기<br>

- 밀키트 사이트 : 쿠팡(https://www.coupang.com/)
- 활용 데이터 : 네이버데이터랩, 만개의 레시피, 요기요

   
## 시나리오
- 밀키트 추천
1. 알레르기 제외 & 카테고리 선택
2. 네이버 트렌드 기준, 밀키트 추천
3. 협업필터링의 고객맞춤 밀키트 추천

- 레시피 추천
1. 선택한 밀키트의 구성정보와 만개의 레시피 요리재료 비교 후, 레시피 추천
2. 밀키트와 레시피에만 있는 재료, 구매 링크 연결(소비유도)

- 비용계산
1. 요기요 기준, 선택한 밀키트와 평균 메뉴 외식 비용 비교

<br>
- 추천시스템 기준
  - 밀키트 추천<br>
   : 밀키트 인기검색어 (소비 트렌드)와 상품명 유사도<br>
   : 쿠팡의 리뷰수 & 별점
  - 레시피 추천<br>
   : 쿠팡의 구성정보와 레시피의 요리재료내용 유사도
