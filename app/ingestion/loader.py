from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


async def load_url(url: str) -> str:
    headers = {"User-Agent": "ResearchAgent/1.0"}

    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "html" not in content_type:
        return response.text

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    return soup.get_text("\n", strip=True)


def load_pdf(path: str):
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_docx(path: str):
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def load_csv(path: str):
    dataframe = pd.read_csv(path)
    return dataframe.to_csv(index=False)


def load_text_file(path: str):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_file(path: str):
    extension = Path(path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(path), "pdf"
    if extension == ".docx":
        return load_docx(path), "docx"
    if extension == ".csv":
        return load_csv(path), "csv"

    return load_text_file(path), "text"
