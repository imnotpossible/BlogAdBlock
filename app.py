import io
import os
from urllib import parse
import urllib.request
import json
import re
from bs4 import BeautifulSoup
import time
import pandas as pd
from google.cloud import vision
import requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_mysqldb import MySQL
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/jhyun/Desktop/BlogAdBlock/blockadblockv-98ab756bd082.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/flask/BlogAdBlock/blockadblockv-98ab756bd082.json'
client_options = {'api_endpoint' : 'eu-vision.googleapis.com'}
app = Flask(__name__)
SCRAPER_API_KEY ='134d368d4adbccfac370f3294a09317f'
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
#model.load_state_dict(torch.load('C:/Users/jhyun/Desktop/BlogAdBlock/AdBlocK_model.pth', map_location=torch.device('cpu')))
model.load_state_dict(torch.load('/home/ubuntu/flask/BlogAdBlock/AdBlock_model.pth', map_location=torch.device('cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mysql = MySQL()
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'  # MySQL 비밀번호
#app.config['MYSQL_DB'] = 'BOOKMARK'  # 생성한 데이터베이스 이름 
app.config['MYSQL_DB'] = 'bookmark'  # 생성한 데이터베이스 이름 
mysql.init_app(app)


#def down_file(link, filename, post_dir_name):
#    file_path = os.path.join(post_dir_name, filename)
#    new_link = urllib.parse.quote(link, safe=':/?-=') #한글 인코딩
#    urllib.request.urlretrieve(new_link, file_path) #파일 출력


@app.route('/bookmarks', methods=['GET'])
def get_bookmarks():
    user_ip = request.remote_addr
    cursor = mysql.connection.cursor()
    #cursor.execute("SELECT blog_link FROM bookmarks WHERE user_ip = %s", (user_ip,))
    cursor.execute("SELECT blog_link, blog_image, blog_title, product_name, product_url, product_price FROM bookmarks WHERE user_ip = %s", (user_ip,))
    bookmarks = cursor.fetchall()
    cursor.close()
    # 북마크된 데이터가 있을 경우 JSON으로 반환
    bookmark_list = [
        {
            'link': bookmark[0],
            'image': bookmark[1],
            'title': bookmark[2],
            'productName': bookmark[3],
            'productUrl': bookmark[4],
            'price': bookmark[5]
        } for bookmark in bookmarks
    ]
    # 블로그 링크 목록 반환
    return jsonify(bookmark_list)


@app.route('/bookmark', methods=['POST'])
def bookmark():
    user_ip = request.remote_addr
    print(request.json)  # 요청 데이터 출력
    blog_image = request.json.get('image')
    blog_title = request.json.get('title')
    blog_link = request.json.get('link')
    product_name = request.json.get('productName')
    product_url = request.json.get('productUrl')
    product_price = request.json.get('price')

    
    # 중복 체크
    cursor = mysql.connection.cursor()
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM bookmarks 
            WHERE user_ip = %s AND blog_link = %s
        """, (user_ip, blog_link))
        
        count = cursor.fetchone()[0]
    
        if count > 0:
            return jsonify({"message": "Bookmark already exists"}), 409  # 409 Conflict

        # MySQL에 데이터 삽입
        cursor.execute(""" 
            INSERT INTO bookmarks (user_ip, blog_link, blog_image, blog_title, product_name, product_url, product_price) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) 
        """, (user_ip, blog_link, blog_image, blog_title, product_name, product_url, product_price))
        mysql.connection.commit()

        return jsonify({"message": "Bookmark added successfully"}), 201
    except Exception as e:
        print(f"Error adding bookmark: {str(e)}")  # 오류 출력
        return jsonify({"message": "Error adding bookmark", "error": str(e)}), 500
    finally:
        cursor.close()

    
    
@app.route('/bookmark', methods=['DELETE'])
def delete_bookmark():
    user_ip = request.remote_addr
    blog_link = request.json.get('link')
    
    # 커서 생성
    cursor = mysql.connection.cursor()
    try:
        # MySQL에서 데이터 삭제
        cursor.execute("""
            DELETE FROM bookmarks 
            WHERE user_ip = %s AND blog_link = %s
        """, (user_ip, blog_link))
        
        mysql.connection.commit()
        
        return jsonify({"message": "Bookmark deleted successfully"}), 200
    except Exception as e:
        # 오류 발생 시 응답
        return jsonify({"message": "Error deleting bookmark", "error": str(e)}), 500
    finally:
        cursor.close()  # 커서는 항상 닫기



    

        
    
def delete_file(file):   
    if os.path.isfile(file): #파일이 존재하는지 확인
        os.remove(file) # 삭제
        
def ocr(image_content):
    print("ocr 진입") 
    client = vision.ImageAnnotatorClient(client_options=client_options) #계정 정보 얻어옴
    #client = vision.ImageAnnotatorClient(credentials=AIzaSyAaL9X5W3bHPO7jcAngRGiFfCNhivAKHSg)
    #with io.open(path, 'rb') as image_file: #입력받은 경로에서 사진 읽어오기
    #    content = image_file.read()
        
    image = vision.Image(content=image_content)
    
    response = client.text_detection(image=image) # 사진에서 텍스트 추출
    texts = response.text_annotations
    result = []
    
    for text in texts:
        result.append('{}'.format(text.description))
    
    print("ocr 완료")    
    return result #텍스트 반환
    

def url_parsing(search_word,search_results):
    print("url_parsing 시작")
    client_id = "CR79xkbRGgMxklOIF7_U" #네이버 api id
    client_secret = "YmDHNRvV1s" # 네이버 api 비밀번호
    encText = urllib.parse.quote(search_word)
    url = "https://openapi.naver.com/v1/search/blog?display="+str(search_results)+"&query=" + encText
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    
    links = []
    
    if(rescode==200):
        response_body = response.read().decode('utf-8')
        response_json = json.loads(response_body)
        for items in response_json['items']:
            if 'naver' in items['link']: #타 블로그 링크 필터링
                links.append(items['link'])
                
                #링크를 리스트에 순차적으로 저장한다
    
    else:
        print("Error Code:" + rescode) # rescode 가 200이 아닐 떄 에러 코드를 출력한다.
        
    print("url_parsing 완료")
    return links
    


def predict_flag(blog_full_content):
    input_text = blog_full_content
    input_encoding = tokenizer.encode_plus(
        input_text,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    input_ids = input_encoding['input_ids'].to(device)
    attention_mask = input_encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
    predicted_labels = predicted_labels.item()
    
    return predicted_labels

def extract_firstpic(pic_link):
    print("firstpic 함수진입")
    url_pic = ""
    
    first_pic = str(pic_link[0]) #배열의 마지막 원소를 통해 첫번째 사진 접근
    link_start = "https"
    link_end = re.compile("type=w[0-9]") #정규표현식으로 링크 위치를 얻어내서 마지막 인덱스 반환
    
    start = first_pic.find(link_start)
    
    end = link_end.search(first_pic)
    end = end.end()
    
    
    for i in range(0,6): #w뒤에 숫자가 몇개나 있는지 파악
        if first_pic[end+i] == '"':
            end = end+i
            break
        
        
    url_pic = first_pic[start:end]
        
    print("extract_firstpic 완료")
        
    return url_pic

def extract_lastpic(pic_link):
    print("lastpic 함수진입")
    url_pic = ""
    
    link_start = "https"
    link_end = re.compile("type=w[0-9]") #정규표현식으로 링크 위치를 얻어내서 마지막 인덱스 반환
    
    final_pic = str(pic_link[-1]) #배열의 마지막 원소를 통해 마지막 사진 접근
    start = final_pic.find(link_start)
    
    end = link_end.search(final_pic)
    end = end.end()
    
    
    for i in range(0,6): #w뒤에 숫자가 몇개나 있는지 파악
        if final_pic[end+i] == '"':
            end = end+i
            break
        
        
    for i in range(start, end): #최종 url
        url_pic = url_pic + final_pic[i]
        
    print("extract_lastpic 완료")
        
    return url_pic

def extract_blog(modified_link):
    print("blog 함수 진입")
    contents = ''
    res = requests.get(modified_link)
    soup_text = BeautifulSoup(res.text, 'html.parser')
    txt_contents = soup_text.find_all('div', class_=re.compile('^se-module se-module-tex.*'))
    for p_span in txt_contents:
        for txt in p_span.find_all('span'):
            contents += txt.get_text() + '\n'
            
    titles = soup_text.find_all('div', class_=re.compile('^se-module se-module-text se-title-tex.*'))
    post_title = titles[0].text
    post_title = post_title.replace('\n', '')

    special_char = '\/:*?"<>|.'
    for c in special_char:
        if c in post_title:
            post_title = post_title.replace(c, '')
            

    return contents
    

def blog_content_cut(text_result, title):
    title_length = len(title.strip())
    blog_content = text_result[title_length:].strip()
    
    blog_content = blog_content[:100]
    
    return blog_content


def extract_title(modified_link):
    # 제목 추출
    res = requests.get(modified_link, timeout=10)
    soup_text = BeautifulSoup(res.text, 'html.parser')
    titles = soup_text.find_all('div', class_=re.compile('^se-module se-module-text se-title-tex.*'))
    post_title = titles[0].text
    post_title = post_title.replace('\n', '')

    special_char = '\/:*?"<>|.'
    for c in special_char:
        if c in post_title:
            post_title = post_title.replace(c, '')
    
    #블로그명 추출
    nick_tag = soup_text.find('meta', property='naverblog:nickname')
    if nick_tag:
        nick = nick_tag.get('content')
    
    #날짜 추출
    date_tag = soup_text.find('span', class_='se_publishDate pcol2')
    if date_tag:
        post_date = date_tag.get_text()
            
    return post_title, nick, post_date

def blog_thumbnail(modified_link):
    res = requests.get(modified_link, timeout=10)
    soup = BeautifulSoup(res.text, 'html.parser')
    img_url = None
    img_tag = soup.find('meta', property='og:image')
    if img_tag:
        img_url = img_tag.get('content')
            
    return img_url
    
def extract_product_name(title, category):
    pattern = r'[a-zA-Z][A-Za-z0-9\s\-]*(?:[A-Z][A-Za-z0-9\s\-]*)+'
    matches=re.findall(pattern, title)
    blog_product_name=None
    
    if matches:
        # 마지막 매칭된 결과를 반환
        blog_product_name = matches[-1].strip()
        #return blog_product_name if blog_product_name else return category = False
    
    if not blog_product_name:
            pattern = r'[a-zA-Z][A-Za-z0-9\s\-]*(?:[a-zA-Z][A-Za-z0-9\s\-]*)+'
            matches=re.findall(pattern, title)
            if matches:
                blog_product_name = matches[-1].strip()
                
    if blog_product_name and blog_product_name.lower() =="feat":
                    blog_product_name = None
    
    print(blog_product_name)
    
    if blog_product_name:
        category += 1

    return blog_product_name, category


def convert_stars(rating):
    try:
        if rating is None:
            return '☆☆☆☆☆'  

        rating_float = float(rating)
        if rating_float >= 5.0:
            rating_stars='★★★★★'
        elif rating_float >= 4.0:
            rating_stars='★★★★☆'
        elif rating_float >= 3.0:
            rating_stars='★★★☆☆'
        elif rating_float >= 2.0:
            rating_stars='★★☆☆☆'
        elif rating_float >= 1.0:
            rating_stars='★☆☆☆☆'
        else:
            rating_stars='☆☆☆☆☆'
    except ValueError:
        rating_stars='☆☆☆☆☆'
    
    return rating_stars

def blog_product_data(blog_product_name):
    if blog_product_name is None:
        print("blog_product_data 진입실패")
        return None, None, None, None
    
    print("blog_product_data 진입") 
    
    encoded_product_name = requests.utils.quote(blog_product_name)
    
    #enc_text = urllib.parse.quote(blog_product_name)
    url = f"https://search.shopping.naver.com/search/all?query=" + encoded_product_name
    print("url 생성")
    scraperapi_url = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={url}&render=true"
    print("scraperapi url 생성")

    response = requests.get(scraperapi_url)
    response.raise_for_status()
    print("페이지 소스 가져오기")
    # 페이지 소스 가져오기

    html = response.text

    print("태그찾기")
    soup = BeautifulSoup(html, 'html.parser')
    
    product_site_tag = soup.find('div', class_='product_item__MDtDF')
    if product_site_tag is None:
        print("제품 사이트 태그를 찾을 수 없습니다.")
        return None, None, None, None
    
    product_site_url_tag = product_site_tag.find('div', class_='product_title__Mmw2K')
    if product_site_url_tag is None:
        print("제품 사이트 URL 태그를 찾을 수 없습니다.")
        return None, None, None, None
    
    product_site_type = None
    product_site_type = product_site_url_tag.find('a', {'data-shp-contents-type': 'catalog_nv_mid'})
    product_site_url = ''
    
    if product_site_type:
        if product_site_tag:
            product_site_url = product_site_url_tag.find('a')['href']
    else:
        product_site_url = url
    
        
    
    #가격
    price_tag = product_site_tag.find('span', class_='price_num__S2p_v')
    product_price = ''
    product_price = price_tag.get_text(strip=True).replace('원','')
    print(product_price)
        
        
    #별점
    point_tag = product_site_tag.find('span', class_='product_grade__IzyU3')
    product_rating = ''
    if point_tag:
        raw_rating = point_tag.get_text(strip=True).replace('별점', '').strip()
        try:
            # 값을 float로 변환 시도
            product_rating = float(raw_rating)
            print(product_rating)
        except ValueError:
            # 변환 실패 시 None
            product_rating = ''
            
    #별점 남긴 사람수
    point_num_tag = product_site_tag.find('div', class_='product_etc_box__ElfVA').find('em', class_='product_num__fafe5')
    review_num = ''
    if point_num_tag:
        review_num = point_num_tag.get_text()
        review_num = review_num.strip()
        review_num = review_num.replace('(', '').replace(')', '')
        print(review_num)
        
            
    return product_price, product_rating, product_site_url, review_num
    

    


#Flask 루트
@app.route("/")
def index():
    return render_template('index.html')

#검색어 처리
@app.route('/search', methods=['GET', 'POST'])
def handle_search():
    if request.method == 'POST':
        word = request.form['search_keyword']
        search_word = word + ' ' + '후기' + ' | ' + '리뷰'
        
        search_results=10
        
        dir_names = word.replace(' ', '')

        if not os.path.exists('naverBlog'):
            os.mkdir('naverBlog')
        else:
            pass
        if not os.path.exists('naverBlog/' + dir_names):
            os.makedirs('naverBlog/' + dir_names)
        else:
            pass
        
        post_dir_name = 'naverBlog/' + dir_names

        #filter_data = ['소정의', '원고료' ,'체험단', '협찬','파트너쉽', '파트너십', '파트너스', '수수료', '광고비','서포터즈','유료광고', '유료 광고', '소정의 원고료를', '제공받아', '제품을 제공받아', '식사권을 제공받아', '숙박권을 제공받아', '체험권을 제공받아', '이용권을 제공받아', '무상으로', '경제적대가', '경제적 대가', '제휴링크', '파트너스활동','파트너스 활동을', '제휴마케팅', '커미션', '무상으로 대여받아', '제품 또는 서비스를 제공받아', '일정액의 수수료를', '파트너스 활동을 통해', '제품대여', '고료']
        #unfilter_data = ['내돈내산']

        links= url_parsing(search_word, search_results)
    

        blog_keyword = []
        blog_name = []
        post_title = []
        post_titles = []
        blog_product_names=[]
        product_prices=[]
        blog_ratings=[]
        product_urls=[]
        img_urls = []
        blog_contents=[]
        review_nums = []
        nicks = []
        post_dates = []
        product_prices_marks = []
        review_nums_marks = []
        blog_full_contents_list = []
        category = 0 
        



        for link in links:
            blog_keyword.append(link.replace('https://blog.naver.com/', '').split('/'))

        modified_links = []

        for keyword in blog_keyword:
            modified_links.append('https://blog.naver.com/PostView.naver?blogId='+keyword[0]+'&logNo='+keyword[1])
            blog_name.append(keyword[0])
            ## iframe이 아닌 바로 페이지 소스 보기가 가능한 주소로 변환한다.
            
        
        filtered_link_words = []
        filtered_yes_no = []
        filtered_word = []
        print_link = []
        count = 0
        i = 0

        print("modified_links:", modified_links)

        for index, modified_link in enumerate(modified_links):
            html = urllib.request.urlopen(modified_link)
            soup = BeautifulSoup(html, 'html.parser')
            pic_link1 = soup.find_all('img',{"data-lazy-src":re.compile(".*")})
            pic_link2 = soup.find_all("img",{"src":re.compile("^https://(www|storep|blogfiles).*")})

            ##태그와 정규표현식을 이용해서 사진 url이 포함된 부분을 긁어온다.

            
            text_result= extract_blog(modified_link)

            flag = 0
            pics = []
            all_pics = []


            if(len(pic_link1)!=0):
                first_pic = extract_firstpic(pic_link1)
                count +=1
                pics.append(first_pic)
            if(len(pic_link1)!=0):
                last_pic = extract_lastpic(pic_link1)
                count +=1
                pics.append(last_pic)
            for img_tag in pic_link2:
                pic_url = img_tag['src']
                pics.append(pic_url)
        
            ocr_texts = []
            pic_count = len(pics)

            for pic_url in pics:
                if pic_url != "":
                    #이미지 요청
                    image_response = requests.get(pic_url, timeout=10)
                    # 이미지 데이터를 바이너리 형태로 가져옴
                    image_content = image_response.content
                    # OCR 함수 호출하여 텍스트 추출
                    result = ocr(image_content)
                    # 추출된 텍스트 출력
                    print(result)
                    #ocr_texts.append(" ".join(result))
                    ocr_texts.append(" ".join(result))

            ocr_text = ""
            title, nick, post_date= extract_title(modified_link)
            ocr_text = " ".join(ocr_texts)
            blog_full_content = title + " " +  text_result + " " +ocr_text
            
            
            #flag 판단 후 넣어야함
            #post_titles.append(post_title)
            #blog_full_contents_list.append(blog_full_content)       
            extracted_texts = []
            
            flag = predict_flag(blog_full_content)


            
            if flag == 1:
                print(title)
                print("광고가 맞습니다")
                filtered_yes_no.append("O")
                
            else:
                print(title)
                print("광고가 아닙니다")
                filtered_yes_no.append("X")
                print_link.append(modified_link)
                post_title.append(title)
                nicks.append(nick)
                post_dates.append(post_date)
                img_url = blog_thumbnail(modified_link)
                img_urls.append(img_url)
                blog_content = blog_content_cut(text_result, title)
                blog_contents.append(blog_content)
                blog_product_name, category = extract_product_name(title, category)
                if category > 0:
                    blog_product_names.append(blog_product_name)
                    product_price, blog_rating, product_site_url, review_num = blog_product_data(blog_product_name)
                    product_prices_marks.append(product_price)
                    if product_price:
                        product_prices.append(int(product_price.replace(',', '')))
                    blog_ratings.append(blog_rating)
                    product_urls.append(product_site_url)
                    review_nums_marks.append(review_num)
                    if review_num:
                        review_nums.append(int(review_num.replace(',','')))
                    if product_prices is not None:
                        price_check = [price for price in product_prices if price is not None]
                        
                        if price_check:
                            min_price = min(price_check)
                            max_price = max(price_check)
                        else:
                            min_price=max_price = 0
                    else:
                        min_price=max_price = 0
                    resBox = 1
                else:
                    blog_product_names.append(None)
                    product_prices_marks.append(None)
                    product_prices.append(None)
                    blog_ratings.append(None)
                    product_urls.append(None)
                    review_nums.append(None)
                    min_price=max_price = None
                    resBox = 0
                    
                    
        
        #data_f = pd.DataFrame.from_dict({'블로그 이름':blog_name,'url':modified_links,'광고글 유무':filtered_yes_no,'광고 키워드':filtered_word}, orient='index').T
        #data_f.to_csv("./result"+dir_names+".csv", encoding='utf-8-sig')
    return render_template('result.html', search_keyword=word, links=print_link, img_urls=img_urls, 
                           titles=post_title, nicks=nicks, post_dates=post_dates, blog_product_names=blog_product_names, 
                           blog_contents = blog_contents, review_nums = review_nums,review_nums_marks=review_nums_marks,
                           product_prices=product_prices, product_prices_marks = product_prices_marks,
                           blog_ratings=blog_ratings, product_urls=product_urls,
                           min_price = min_price,
                           max_price = max_price, resBox = resBox)



# if __name__ == '__main__':
#     app.debug = True
#     app.run(port=5500)
    
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
