import os
import time
from selenium import webdriver
import chromedriver_binary
from PIL import Image
import io
import requests
import argparse
import hashlib


def get_image(query, category, download_num, save_path):
    sleep_between_interactions = 2
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    wd = webdriver.Chrome()
    wd.get(search_url.format(q=query))
    thumbnail_results = wd.find_elements_by_css_selector("img.rg_i")
    print("number of thumbnails:", len(thumbnail_results))

    # サムネイルをクリックして、各画像URLを取得
    image_urls = set()
    for img in thumbnail_results[:download_num]:
        try:
            img.click()
            time.sleep(sleep_between_interactions)
        except Exception:
            continue

        url_candidates = wd.find_elements_by_class_name('n3VNCb')
        for candidate in url_candidates:
            url = candidate.get_attribute('src')
            if url and 'https' in url:
                image_urls.add(url)

    time.sleep(5)
    wd.quit()
    image_urls = list(image_urls)
    print("number of valid image urls:", len(image_urls))

    # 画像のダウンロード
    for i, url in enumerate(image_urls):
        try:
            image_content = requests.get(url).content
        except Exception as e:
            # print(f"ERROR - Could not download {url} - {e}")
            continue

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            base_path = os.path.join(save_path, category)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            file_name = os.path.join(base_path, f"{query}_{i}.png")
            with open(file_name, 'wb') as f:
                image.save(f, "PNG", quality=90)
        except Exception as e:
            # print(f"ERROR - Could not save {url} - {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train louvre')
    parser.add_argument("--query", "-q", default=None)
    parser.add_argument("--category", "-c", default="class1")
    parser.add_argument("--download_num", "-n", type=int, default=50)
    parser.add_argument("--save_path", "-p", default="./img")
    args = parser.parse_args()

    print("search word:", args.query)

    get_image(args.query, args.category, args.download_num, args.save_path)
