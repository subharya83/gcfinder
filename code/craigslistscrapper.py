import spacy
import re
import logging
import argparse
import csv
from typing import List, Set, Tuple, Optional

from requests import Session
from bs4 import BeautifulSoup
import numpy as np
from random import randint
from time import sleep
from string import punctuation
from io import StringIO
from html.parser import HTMLParser

# Constants
BASE_URL = 'https://{loc}.craigslist.org/d/services/search/bbb?query=handyman&sort=rel'
RESULTS_PER_PAGE = 120

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d: str) -> None:
        self.text.write(d)

    def get_data(self) -> str:
        return self.text.getvalue()

def strip_tags(html: str) -> str:
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class CraigslistScraper:
    def __init__(self, loc: str, page: int = 0, output: str = None):
        self.nlp = spacy.load("en_core_web_md")
        self.querystring = BASE_URL.format(loc=loc)
        self.pagenum = page
        self.output = output
        self.session = Session()

    def get_hotwords(self, text: str) -> Set[str]:
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        doc = self.nlp(text.lower())
        return {token.text for token in doc if token.pos_ in pos_tag and token.text not in self.nlp.Defaults.stop_words and token.text not in punctuation}

    def parse_posts(self) -> None:
        page_cnt = 0
        response = self.session.get(self.querystring)
        response.raise_for_status()
        html_soup = BeautifulSoup(response.text, 'html.parser')

        results_num = html_soup.find('div', class_='search-legend')
        if not results_num:
            logging.error("No results found on the page.")
            return

        results_total = int(results_num.find('span', class_='totalcount').text)
        pages = np.arange(page_cnt, results_total + 1, RESULTS_PER_PAGE)
        pages = pages[self.pagenum:]

        logging.info('Fetching records from index [%04d] of [%04d]', pages[0], pages[-1])

        with open(self.output, mode='w', newline='', encoding='utf-8') as out_file:
            fieldnames = ['post_datetime', 'post_link', 'lat', 'lng', 'acc', 'post_contacts', 'post_keywords']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

            for page in pages:
                response = self.session.get(self.querystring + "&s=" + str(page))
                response.raise_for_status()
                sleep(randint(1, 5))

                page_html = BeautifulSoup(response.text, 'html.parser')
                posts = page_html.find_all('li', class_='result-row')

                for post in posts:
                    post_data = self.extract_post_data(post)
                    if post_data:
                        writer.writerow(post_data)
                        logging.info('Post fetched from %s', post_data['post_datetime'])

                page_cnt += 1
                logging.info("Page [%02d] scraped successfully", page_cnt)

    def extract_post_data(self, post) -> Optional[dict]:
        try:
            post_datetime = post.find('time', class_='result-date')['datetime']
            post_title = post.find('a', class_='result-title hdrlnk')
            post_title_text = post_title.text
            post_link = post_title['href']
            lat, lng, acc, post_contacts, post_keywords = self.parse_post_link(post_link)

            return {
                'post_datetime': post_datetime,
                'post_link': post_link,
                'lat': lat,
                'lng': lng,
                'acc': acc,
                'post_contacts': ', '.join(post_contacts),
                'post_keywords': ', '.join(post_keywords)
            }
        except Exception as e:
            logging.error(f"Error extracting post data: {e}")
            return None

    def parse_post_link(self, post_url: str) -> Tuple[Optional[float], Optional[float], int, Set[str], Set[str]]:
        lat, lng, acc = None, None, 0

        response = self.session.get(post_url)
        response.raise_for_status()
        sleep(randint(1, 5))

        post_page_html = BeautifulSoup(response.text, 'html.parser')
        post_geoloc = post_page_html.find_all('div', class_='viewposting')

        for elem in post_geoloc:
            lat = elem.get('data-latitude')
            lng = elem.get('data-longitude')
            acc = elem.get('data-accuracy')

        post_body = post_page_html.find_all('section', {"id": "postingbody"})
        post_body = strip_tags(str(post_body))
        post_contacts = set(re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', post_body))
        post_keywords = self.get_hotwords(str(post_body))

        return lat, lng, acc, post_contacts, post_keywords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handyman app data scraping tool from Craigslist')
    parser.add_argument('-l', type=str, required=True, help='Location code [sfbay|charlotte|seattle]')
    parser.add_argument('-o', type=str, required=True, help='Path to output csv file')
    parser.add_argument('-p', type=int, default=0, required=False, help='Page number to resume from last session')

    args = parser.parse_args()
    scraper = CraigslistScraper(loc=args.l, page=args.p, output=args.o)
    scraper.parse_posts()