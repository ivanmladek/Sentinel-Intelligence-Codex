# World History Book Collection

> [!NOTE]
> The books and PDFs referenced in this library are not hosted in this repository. They are freely available for download from the following public resource:
> 
> https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/

This project is a curated collection of historical texts, offering a broad and deep exploration of human history from prehistory to the modern era. The library is organized into a comprehensive set of categories, with a significant emphasis on ancient and classical civilizations, as well as detailed accounts of various historical periods and regions.

The collection is particularly strong in its coverage of **Prehistory** and **Ancient & Classical Civilizations**. The prehistory section delves into archaeology, the Bronze Age, the Iron Age, and the Neolithic period, providing a rich foundation in the early chapters of human history. The section on ancient and classical civilizations is extensive, with in-depth collections on:

*   **Alexander the Great**: Covering his life, conquests, and the Hellenistic world that followed.
*   **Ancient Egypt**: A vast collection of texts on Egyptian dynasties, mythology, art, and archaeology.
*   **Ancient Greece**: Exploring the Archaic, Classical, and Hellenistic periods, with a focus on Athens, Sparta, and the broader Greek world.
*   **Ancient Rome**: A comprehensive look at the Roman Republic and Empire, including its military, society, and key historical figures.
*   **Other Civilizations**: The library also includes significant collections on the civilizations of the Near East, the Celts, the Vikings, and ancient China, among others.

Beyond the ancient world, the collection extends to **Medieval and Modern History**, with detailed sections on:

*   **The Middle Ages**: Covering the Byzantine Empire, the Crusades, and the development of European kingdoms.
*   **Early Modern History**: Exploring the Renaissance, the Reformation, and the Age of Discovery.
*   **Modern History**: With extensive material on World War I, World War II, the Cold War, and various regional conflicts.

The library is not limited to political and military history. It also includes substantial collections on:

*   **Art and Architecture**: With a focus on historical periods and styles.
*   **Philosophy and Religion**: Exploring the intellectual and spiritual traditions of various cultures.
*   **Science and Technology**: Charting the history of scientific discovery and technological innovation.

The ten most important subjects, based on the number of books and depth of coverage, appear to be:

1.  **Ancient & Classical Civilizations**: The most prominent subject, with extensive collections on Egypt, Greece, Rome, and Alexander the Great.
2.  **Prehistory**: A deep dive into archaeology and the early stages of human civilization.
3.  **Military History**: Covering a wide range of historical conflicts, from ancient warfare to modern-day battles.
4.  **Medieval History**: With a strong focus on the Byzantine Empire, the Crusades, and European kingdoms.
5.  **Modern History**: Encompassing World War I, World War II, and the Cold War.
6.  **Regional Histories**: Detailed collections on the history of specific countries and regions, including Britain, China, and the Middle East.
7.  **Art & Architecture**: Exploring the history of art and architecture across different cultures and periods.
8.  **Philosophy & Religion**: A significant collection on the history of ideas and belief systems.
9.  **Biographies**: In-depth accounts of key historical figures, from ancient rulers to modern leaders.
10. **Archaeology**: With a focus on the methods and discoveries that have shaped our understanding of the past.

## Library Structure

The library is organized into a hierarchical structure, with the following main categories:

```
.
├── Prehistory
├── Ancient & Classical Civilizations
├── Medieval History
├── Early Modern History
├── Modern History
├── Regional History
├── Military History
├── Art & Architecture
├── Philosophy & Religion
└── Science & Technology
```

## PDF Text Extraction and Processing

The process of extracting, cleaning, and preparing the text from PDF files for the LLM is a multi-stage pipeline designed to ensure high-quality, structured data. This process is orchestrated by the `process_refactor.ipynb` notebook.

### 1. Environment Setup and PDF Discovery

- **Dependencies**: The process begins by installing necessary Python libraries, including `nougat-ocr` for text extraction, `nltk` for natural language processing, and `langdetect` for language identification.
- **PDF Discovery**: The script recursively scans a specified directory (e.g., a Google Drive folder) to locate all PDF files.

### 2. Text Extraction with Nougat

- **Nougat OCR**: For each PDF, the `nougat` command-line tool is used. Nougat is a state-of-the-art OCR tool specifically designed for academic and scientific documents, capable of recognizing and transcribing complex layouts, mathematical equations, and tables into a structured Markdown format (`.mmd`).
- **Output**: The raw extracted text is saved as a `.mmd` file, preserving the document's structure with Markdown headings.

### 3. Text Cleaning and Garbage Detection

This is a critical step to filter out irrelevant or low-quality text.

- **Cleaning**: A series of regular expressions and cleaning functions are applied to the raw text to:
    - Remove extra newlines, spaces, and non-ASCII characters.
    - Eliminate academic citations, references to tables/figures, and bracketed content.
    - Sanitize garbled punctuation and symbols.
- **Garbage Detection**: Each segment of text is evaluated against a set of criteria to identify and discard "garbage" content. This includes:
    - **Language Detection**: Text that is not identified as English is discarded.
    - **Heuristics**: Checks for "jammed" words (long strings of characters without spaces), an unusually high proportion of single-letter words, and repetitive patterns.
    - **Quality Scoring**: A `text_quality_score` is calculated based on the presence of common English words, proper part-of-speech patterns, and other linguistic features. Text falling below a certain threshold is flagged as garbage.

### 4. Tokenization and Chunking

- **Chunking Strategy**: The cleaned `.mmd` content is chunked into smaller, manageable segments suitable for the LLM. The chunking logic is designed to respect the document's structure:
    - The text is split by Markdown headings (`#`, `##`, `###`).
    - These larger sections are then further divided into sentences using `nltk.sent_tokenize`.
- **Size Constraints**: The sentences are grouped into chunks with a maximum size (e.g., 8192 characters) to ensure they fit within the model's context window, while avoiding splitting sentences in the middle.
- **Final Output**: The cleaned, chunked text is saved to a `.jsonl` file, with each line containing a JSON object with a single "text" key, ready for training the LLM. Garbage text is saved to a separate file for review.
