from typing import List, Dict, Union, Optional
from transformers.pipelines import pipeline

pipe = pipeline("text2text-generation", model="iamTangsang/MarianMT-Nepali-to-English")

class News: 
    """
        Represents a news article with various attributes.
    """
    def __init__(
        self, 
        title: str, 
        link: str, 
        content: str, 
        post_date: str
    ) -> None: 
        
        self.title = title
        self.link = link
        self.content = content
        self.post_date = post_date
    
    def to_dict(self) -> Dict[str, Union[str, list[str]]]:
        """
        Convert the News object to a dictionary representation.

        Returns:
            Dict[str, Union[str, list[str]]]: Dictionary containing all attributes of the News object.
        """
        return {
            "title": self.title,
            "link": self.link,
            "content": self.content,
            "post_date": self.post_date
        }
        
    def __str__(self) -> str:
        """
        String (tabular) representation of the News object using pandas.

        Long content is truncated for display purposes.
        """
        output = f"""
            {'-' * 50}
            🔹 News Article
            📌  Title: {self.title}
            🔗  Link: {self.link}
            📝  Content: {self.content}
            🗓️  Post Date: {self.post_date}
            {'-' * 50}
        """
        return output
    
    def __repr__(self) -> str:
        """
        String representation of the News object.

        Returns:
            str: Formatted string containing the title and link of the news article.
        """
        return f"News(title={self.title}, link={self.link})"

    def to_english(self, inplace: bool = False) -> Optional["News"]:
        """Translate the news article content to English.

        Returns:
            None
            A new News object with translated content if not inplace.
        """
        def translate_text(text: str) -> str:
            """Translate a single text string to English."""
            translated = pipe(text, max_length=512, truncation=True)
            
            # Check if translation was successful
            if not translated or type(translated) is not list or len(translated) == 0: 
                return ""
            
            return translated[0].get('generated_text', "")

        # Translate all text fields first
        translated = {
            'title': translate_text(self.title),
            'content': translate_text(self.content),
        }

        if inplace:
            self.title = translated['title']
            self.content = translated['content'] 
            return
        
        return News(
            title=translated['title'],
            link=self.link,
            content=translated['content'],
            post_date=self.post_date
        )