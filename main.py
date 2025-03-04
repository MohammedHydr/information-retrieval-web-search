import os
import time
import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/108.0.0.0 Safari/537.36"
    )
}

MAX_WORKERS = 10  # Number of threads


def is_same_domain(base_domain, url):
    return urlparse(url).netloc == base_domain


def save_page(html, page_number, folder="crawled_pages"):
    os.makedirs(folder, exist_ok=True)
    filename = f"page_{page_number:05d}.html"
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def extract_article_text(soup):
    article_selectors = [
        ('article', None),
        ('div', 'article-body'),
        ('div', 'article__content'),
        ('section', 'articleContent'),
    ]
    for tag, class_name in article_selectors:
        container = soup.find(tag, class_=class_name) if class_name else soup.find(tag)
        if container:
            return container.get_text(separator=' ', strip=True)
    return soup.get_text(separator=' ', strip=True)


def process_page(url, base_domain, seen_content):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None, []

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        article_text = extract_article_text(soup)

        saved = False
        if "real madrid" in article_text.lower():
            page_hash = hash(html)
            if page_hash not in seen_content:
                seen_content.add(page_hash)
                return html, extract_links(soup, url, base_domain)

        return None, extract_links(soup, url, base_domain)
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, []


def extract_links(soup, base_url, base_domain):
    links = []
    for a in soup.find_all('a', href=True):
        next_url = urljoin(base_url, a['href'])
        if is_same_domain(base_domain, next_url):
            links.append(next_url)
    return links


def crawl_bein_sports_real_madrid(start_url, max_pages=10000):
    visited_urls = set()
    seen_content = set()
    queue = deque([start_url])
    base_domain = urlparse(start_url).netloc
    page_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while queue and page_count < max_pages:
            futures = {}

            while queue and len(futures) < MAX_WORKERS:
                current_url = queue.popleft()
                if current_url in visited_urls:
                    continue
                visited_urls.add(current_url)
                futures[executor.submit(process_page, current_url, base_domain, seen_content)] = current_url

            for future in as_completed(futures):
                html, new_links = future.result()
                if html:
                    page_count += 1
                    save_page(html, page_count)
                    print(f"Saved page #{page_count}")
                queue.extend(new_links)

    print(f"\nDone! Pages mentioning 'Real Madrid' in the ARTICLE BODY: {page_count}")


if __name__ == "__main__":
    START_URL = "https://www.goal.com/en"
    crawl_bein_sports_real_madrid(START_URL, max_pages=10000)
