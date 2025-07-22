import sys; sys.path.append('.') # loading the parent directory paths
import cloudscraper
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Union
import json
from pathlib import Path
from scripts.scrapers.utils import News

"""
nepse_alpha_scraper.py

A simple scraper for fetching the latest news articles from nepsealpha.com.
Provides functionality to retrieve titles, links, article content and postdate.

Example Usage:
    from pathlib import Path
    from nepse_alpha_scraper import NepseAlphaScraper

    # Initialize scraper
    scraper = NepseAlphaScraper()

    # 1. Fetch latest news and inspect in memory
    latest_news = scraper.get_latest_news()
    print(latest_news)

    # 2. Save latest news to a JSON file
    save_path = Path("latest_news.json")
    scraper.save_latest_news(save_path)
    print(f"Latest news saved to {save_path}")
    
    # 3. Fetch latest news as a list of dictionaries
    latest_news_dict = scraper.get_latest_news_dict()
   print(latest_news_dict)
"""
class NepseAlphaScraper:
    """
    A scraper class for extracting news data from merolagani.com.

    Attributes:
        base_url (str): The root URL of the Nepse Alpha website.
        latest_news_url (str): URL endpoint for the latest news section.
    """

    def __init__(self) -> None:
        """
        Initialize the scraper with the base and latest news URLs.
        """
        self.base_url: str = "https://nepsealpha.com/"
        self.breaking_news_url: str = "https://nepsealpha.com/api/smx9841/get_breaking_news"
        self.remaining_news_url: str = "https://nepsealpha.com/api/smx9841/get_remaining_news"
        self._scraper = cloudscraper.create_scraper()

    def _get_news_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Internal method to fetch HTML content and parse it into a BeautifulSoup object.

        Args:
            url (str): The URL to fetch.

        Returns:
            BeautifulSoup or None: Parsed soup object if the request succeeds, else None.
        """
        try:
            response = self._scraper.get(url)
            response.encoding = 'utf-8'  # Ensure proper encoding for Nepali characters
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except cloudscraper.exceptions.CloudflareChallengeError as e:
            print(f"Cloudflare challenge error: {e}")
            return None
        except cloudscraper.exceptions.CloudflareCode1020 as e:
            print(f"Access denied by Cloudflare: {e}")
            return None
        except Exception as e:
            print(f"Error fetching news page: {e}")
            return None

    def get_news_content(self, soup: Optional[BeautifulSoup]) -> str:
        """
        Extract the main article content from a parsed page.

        Args:
            soup (BeautifulSoup | None): Parsed HTML of the article page.

        Returns:
            str: Text content of the article, or an error message if unavailable.
        """
        if not soup:
            return "No content available"

        content_div = soup.select_one("#postDescriptions")
        if content_div:
            return content_div.get_text(strip=True)

        return "Content not found"

    def get_post_date(self, soup: Optional[BeautifulSoup]) -> str:
        """
        Extract the publication date from an article page.

        Args:
            soup (BeautifulSoup | None): Parsed HTML of the article page.

        Returns:
            str: The publication date as a string, or an empty string if not found.
        """
        if not soup:
            return ""

        date_element = soup.find("p", class_=["detail", 'date'])
        return date_element.get_text(strip=True) if date_element else ""

    def _fetch_latest_news_title_and_url(self) -> List[dict]: 
        """
        Fetch the latest news articles from the Nepse Alpha API.

        Returns:
            dict: JSON response containing the latest news articles.
        """
        try:
            response = self._scraper.get(self.remaining_news_url)
            response.encoding = 'utf-8'  # Ensure proper encoding for Nepali characters
            response.raise_for_status()
            remaining_news = response.json()
            
            response = self._scraper.get(self.breaking_news_url)
            response.encoding = 'utf-8'  # Ensure proper encoding for Nepali characters
            response.raise_for_status()
            breaking_news = response.json()
        except Exception as e:
            print(f"Error fetching latest news: {e}")
            return []

        all_news = [*remaining_news.get("breaking", []), *breaking_news.get("breaking", [])]
        return all_news

    def get_latest_news(self, limit: Optional[int] = None) -> List[News]:
        """
        Fetch and parse the list of latest news articles.

        Returns:
            List[News]: A list of News objects, each containing:
                - title: The headline of the article.
                - link: The URL to the full article.
                - content: The main article text.
        """

        # extract latest news
        news_items = self._fetch_latest_news_title_and_url()
        articles: List[News] = []

        for i, item in enumerate(news_items, start=1):
            if not item:
                continue
            link = self.base_url + "/post/detail/" + str(item.get('id')) + '/' + str(item.get('slug'))
            title = str(item.get('title')) if item and item.get('title') else "No title"

            article_soup = self._get_news_soup(link)
            articles.append(News(
                title= title,
                link= link,
                content= self.get_news_content(article_soup),
                post_date= self.get_post_date(article_soup)
            ))
            if limit and i >= limit:
                break

        # Remove duplicates based on title and link
        unique = self._filter_duplicate_articles(articles)
        return unique

    def get_latest_news_dict(self) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Fetch the latest news and return a list of dictionaries.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: List of dictionaries with news details.
        """
        news_data = self.get_latest_news()
        return [news.to_dict() for news in news_data]
    
    def _filter_duplicate_articles(self, articles: List[News]) -> List[News]:
        """
        Remove duplicate articles from the list based on title and link.

        Args:
            articles (List[News]): List of News objects.

        Returns:
            List[News]: List of unique News objects.
        """
        seen = {}
        
        for news in articles: 
            if not news.title in seen:
                seen[news.title] = news
        return list(seen.values())

    def save_latest_news(self, save_path: Path, latest_news: Optional[List[News]] = None) -> None:
        """
        Save the fetched latest news to a JSON file.

        Args:
            save_path (Path): File system path to save the JSON data.
            latest_news (Optional[List[News]]): List of News objects to save. If None, fetch latest news.

        Raises:
            IOError: If the file cannot be written.
        """
        news_data = latest_news if latest_news is not None else self.get_latest_news()
        try:
            if save_path is not None:
                with save_path.open('w', encoding='utf-8') as file:
                    json.dump([news.to_dict() for news in news_data], file, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Error saving news to {save_path}: {e}")
    
if __name__ == "__main__":
    # Initialize scraper instance
    scraper = NepseAlphaScraper()

    # Example: fetch latest news into memory
    latest_news = scraper.get_latest_news()
    # [news.to_english(inplace=True) for news in latest_news]

    # Example: save latest news to JSON file
    save_path = Path("latest_news.json")
    scraper.save_latest_news(save_path, latest_news=latest_news)
    print(f"Latest news saved to {save_path}")