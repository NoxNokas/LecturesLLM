# в LangChain есть WebBaseLoader, который скачивает HTML-страницу по указанной ссылке и извлекает из неё текст.
# Ниже приводится пример работы компонента со статьей на Habr. На выходе получается один документ,
# у которого в метадате будет исходная ссылка, а в page_content сам текст статьи без картинок и комментариев.
import bs4
from langchain_community.document_loaders import WebBaseLoader


page_url = "https://habr.com/ru/companies/sherpa_rpa/articles/847058/"
loader = WebBaseLoader(
    web_path=page_url,
    bs_kwargs={"parse_only": bs4.SoupStrainer(attrs={"id": "post-content-body"})}
    )
web_pages = loader.load()
print(len(web_pages))
print(web_pages[0].metadata, web_pages[0].page_content)