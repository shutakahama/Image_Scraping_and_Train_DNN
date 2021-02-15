import os
import requests
import random
import shutil
import bs4
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def image_urls(data, num):
    Res = requests.get("https://www.google.com/search?hl=jp&q=" + data + "&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
    Html = Res.text
    Soup = bs4.BeautifulSoup(Html, 'lxml')
    links = Soup.find_all("img")
    srcs = []
    i = 0
    while len(srcs) < num and i < len(links):
        src = links[i].get("src")
        if src.startswith("https:"):
            srcs.append(src)
        i += 1

    return srcs


def get_image(query, category, num, save_path):
    srcs = image_urls(data, num)
    base_path = os.path.join(save_path, category)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i, path in enumerate(srcs):
        file_name = f"/{query}_{i}.png"
        r = requests.get(path, stream=True)
        if r.status_code == 200:
            with open(file_name, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

    print("Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train louvre')
    parser.add_argument("--query", "-q", default=None)
    parser.add_argument("--category", "-c", default="class1")
    parser.add_argument("--download_num", "-n", type=int, default=50)
    parser.add_argument("--save_path", "-p", default="./img")
    args = parser.parse_args()

    print("search word:", args.query)

    get_image(args.query, args.category, args.download_num, args.save_path)
