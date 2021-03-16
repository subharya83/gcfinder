import spacy
import re
import logging
import argparse
import csv

from requests import get
from bs4 import BeautifulSoup
import numpy as np
from random import randint
from time import sleep
from string import punctuation
from io import StringIO
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class Craigslistscraper:
    def __init__(self, loc=None, page=0, output=None):
        self.nlp = spacy.load("en_core_web_md")
        self.querystring = 'https://' + loc + '.craigslist.org/d/services/search/bbb?query=handyman&sort=rel'
        self.pagenum = page
        self.output = output

    def get_hotwords(self, text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']  # 1
        doc = self.nlp(text.lower())  # 2
        for token in doc:
            if token.text in self.nlp.Defaults.stop_words or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                result.append(token.text)
        return result

    def parse_posts(self):
        page_cnt = 0
        response = get(self.querystring)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        results_num = html_soup.find('div', class_='search-legend')
        results_total = int(results_num.find('span', class_='totalcount').text)
        pages = np.arange(page_cnt, results_total + 1, 120)
        # For resuming from a different page
        pages = pages[self.pagenum: -1]
        logging.info('Fetching records from index [%04d] of [%04d]' % (pages[0], pages[-1]))
        with open(self.output, mode='w') as out_file:
            out_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for page in pages:

                # get request
                response = get(self.querystring + "s=" + str(page) + "&availabilityMode=0")
                sleep(randint(1, 5))

                # define the html text
                page_html = BeautifulSoup(response.text, 'html.parser')

                # define the posts
                posts = html_soup.find_all('li', class_='result-row')

                # extract data item-wise
                for post in posts:
                    # posting date
                    post_datetime = post.find('time', class_='result-date')['datetime']

                    # title text
                    post_title = post.find('a', class_='result-title hdrlnk')
                    post_title_text = post_title.text

                    # post link
                    post_link = post_title['href']
                    lat, lng, acc, post_contacts, post_keywords = self.parse_post_link(post_link)

                    logging.info('Post fetched from %s' % post_datetime)
                    out_writer.writerow([post_datetime, post_link, lat, lng, acc, post_contacts, post_keywords])
                page_cnt += 1
                logging.info("Page [%02d] scraped successfully" % page_cnt)

    def parse_post_link(self, post_url):
        lat = None
        lng = None
        acc = 0

        # Traverse to each post link and get relevant information
        post_page = get(post_url)
        sleep(randint(1, 5))

        # define the html text
        post_page_html = BeautifulSoup(post_page.text, 'html.parser')

        # gather post location if available
        post_geoloc = post_page_html.find_all('div', class_='viewposting')
        for elem in post_geoloc:
            lat = elem.get('data-latitude')
            lng = elem.get('data-longitude')
            acc = elem.get('data-accuracy')

        post_body = post_page_html.find_all('section', {"id": "postingbody"})
        post_body = strip_tags(str(post_body))
        post_contacts = set(re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', post_body))
        post_keywords = set(self.get_hotwords(str(post_body)))
        return lat, lng, acc, post_contacts, post_keywords


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Handyman app data scraping tool from Craigslist')
    parser.add_argument('-l', type=str, required=True, help='Location code [sfbay|charlotte|seattle]')
    parser.add_argument('-o', type=str, required=True, help='Path to output csv file')
    parser.add_argument('-p', type=int, default=0, required=False, help='Page number to resume from last session')

    args = parser.parse_args()
    cls = Craigslistscraper(loc=args.l, page=int(args.p), output=args.o)
    cls.parse_posts()



