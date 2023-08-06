import requests
from bs4 import BeautifulSoup
import concurrent.futures
from dateutil.parser import parse
import json

def process_word(word):
    w = word.strip().replace(" ", "-").lower()
    s, e = scrape_slang(w)
    year_submitted = extract_years(s)
    year_edited = extract_years(e)
    return w, year_submitted, year_edited

def extract_years(dates):
    years = []
    for date in dates:
        parsed_date = parse(date)
        years.append(parsed_date.year)
    return years


def scrape_word_list(url):
    words = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the words
    word_table = soup.find('table', {'class': 'wordlist'})

    # Extract the words from the table
    word_links = word_table.find_all('a')
    for link in word_links:
        word = link.text.strip()
        words.append(word)

    return words

def scrape_words():
    base_url = "http://onlineslangdictionary.com"
    start_url = base_url + "/word-list/0-a/"
    response = requests.get(start_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the links to the word lists
    word_list_links = soup.find('table', {'id': 'wl0'}).find_all('a')

    # Create a list of the URLs to the word lists
    word_list_urls = [base_url + link['href'] for link in word_list_links]

    words = []

    # Use a ThreadPoolExecutor to scrape the word lists in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(scrape_word_list, url): url for url in word_list_urls}
        for future in concurrent.futures.as_completed(future_to_url):
            words.extend(future.result())

    return words

def scrape_slang(word):
    base_url = "http://onlineslangdictionary.com/meaning-definition-of/"
    url = base_url + word

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the paragraphs containing the submission and edit dates
    date_paragraphs = soup.find_all('p', {'class': 'attrib'})

    submitted_dates = []
    edited_dates = []

    for paragraph in date_paragraphs:
        text = paragraph.text
        if "Submitted by" in text:
            date = text.split("on")[-1].strip()
            submitted_dates.append(date)
        if "Last edited on" in text:
            date = text.split("Last edited on")[-1].split(".")[0].strip()
            edited_dates.append(date)

    return submitted_dates, edited_dates



if __name__ == "__main__":
    words = scrape_words()
    word_data = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_word = {executor.submit(process_word, word): word for word in words}
        for future in concurrent.futures.as_completed(future_to_word):
            w, year_submitted, year_edited = future.result()
            word_data[w] = {
                'year_submitted': year_submitted,
                'year_edited': year_edited
            }

    with open("onlineslangdictionary_words.json", 'w') as fp:
        json.dump(word_data, fp)
