# Web Scraper for General Contractor Services

This project is a web scraping tool designed to extract information about handyman services from multiple platforms, including **Craigslist**, **Yelp**, **Nextdoor**, and **Google Reviews**. The scraped data is saved in a CSV file for further analysis.

---

## Features

- **Multi-Platform Support**: Scrape data from Craigslist, Yelp, Nextdoor, and Google Reviews.
- **Customizable Search**: Specify the location and platforms to scrape.
- **Data Extraction**: Extract details such as:
  - Posting date and time
  - Title and description
  - Contact information
  - Location (latitude, longitude, accuracy)
  - Keywords (using NLP)
- **CSV Output**: Save scraped data in a structured CSV format.

---

## Prerequisites

Before running the scraper, ensure you have the following installed:

1. **Python 3.8+**
2. **Required Libraries**:
   - Install the required libraries using the following command:
     ```bash
     pip install requests beautifulsoup4 numpy spacy selenium
     ```
3. **SpaCy Language Model**:
   - Download the English language model for SpaCy:
     ```bash
     python -m spacy download en_core_web_md
     ```
4. **ChromeDriver**:
   - Install ChromeDriver for Selenium (required for Yelp, Nextdoor, and Google Reviews scraping):
     - Download from [here](https://sites.google.com/chromium.org/driver/).
     - Ensure the `chromedriver` executable is in your system's PATH.

---

## Usage

### Command-Line Arguments

The scraper supports the following command-line arguments:

| Argument       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `-l`           | Location code (e.g., `sfbay`, `charlotte`, `seattle`).                      |
| `-o`           | Path to the output CSV file (e.g., `output.csv`).                           |
| `-p`           | Page number to resume from (Craigslist only, default: `0`).                 |
| `--platforms`  | Platforms to scrape (default: `craigslist`). Options: `craigslist`, `yelp`, `nextdoor`, `google`. |

### Example Commands

1. **Scrape Craigslist Only**:
   ```bash
   python gcdatascraper.py -l sfbay -o output.csv
   ```

2. **Scrape Yelp and Nextdoor**:
   ```bash
   python gcdatascraper.py -l sfbay -o output.csv --platforms yelp nextdoor
   ```

3. **Scrape All Platforms**:
   ```bash
   python gcdatascraper.py -l sfbay -o output.csv --platforms craigslist yelp nextdoor google
   ```

---

## Output CSV Format

The scraped data is saved in a CSV file with the following columns:

| Column          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `platform`      | Platform name (e.g., `craigslist`, `yelp`).                                 |
| `post_datetime` | Date and time of the post.                                                  |
| `post_link`     | URL of the post.                                                            |
| `lat`           | Latitude of the service location (if available).                            |
| `lng`           | Longitude of the service location (if available).                           |
| `acc`           | Accuracy of the location data (if available).                               |
| `post_contacts` | Extracted contact information (e.g., phone numbers).                        |
| `post_keywords` | Keywords extracted from the post description (using NLP).                   |

---

## Platform-Specific Notes

### Craigslist
- No authentication required.
- Supports pagination.

### Yelp
- Requires Selenium for scraping due to anti-scraping measures.
- May need proxies or rotating user agents to avoid being blocked.

### Nextdoor
- Requires authentication (login) to access search results.
- Use Selenium to handle login and cookies.

### Google Reviews
- Requires Selenium to scrape reviews from Google Maps.
- May need to handle CAPTCHA or other anti-bot measures.

---

## Limitations

1. **Anti-Scraping Measures**:
   - Platforms like Yelp and Google Reviews have strict anti-scraping measures. Use proxies, rotating user agents, or official APIs if available.
2. **Rate Limiting**:
   - Avoid sending too many requests in a short period to prevent being blocked.
3. **Dynamic Content**:
   - Platforms like Yelp and Nextdoor use JavaScript to load content. Selenium is required to render the pages.
