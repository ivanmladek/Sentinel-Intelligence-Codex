flowchart TD
    A[1. Environment Setup and PDF Discovery] --> B[2. Text Extraction with Nougat]
    B --> C[3. Text Cleaning and Garbage Detection]
    C --> D[4. Tokenization and Chunking]

    subgraph A[1. Environment Setup and PDF Discovery]
        A1[Dependencies: nougat-ocr, nltk, langdetect]
        A2[PDF Discovery: Recursively scan directory]
    end

    subgraph B[2. Text Extraction with Nougat]
        B1[Nougat OCR: Academic document processing]
        B2[Output: Structured .mmd files]
    end

    subgraph C[3. Text Cleaning and Garbage Detection]
        C1[Cleaning Processes]
        C2[Garbage Detection Processes]
        
        subgraph C1
            C11[Remove extra newlines, spaces, non-ASCII]
            C12[Eliminate citations, references, bracketed content]
            C13[Sanitize garbled punctuation and symbols]
        end
        
        subgraph C2
            C21[Language Detection: Discard non-English]
            C22[Heuristics: Jammed words, single-letter words]
            C23[Quality Scoring: English words, POS patterns]
        end
    end

    subgraph D[4. Tokenization and Chunking]
        D1[Chunking Strategy]
        D2[Size Constraints: Max 8192 characters]
        D3[Final Output: .jsonl file with text objects]
        
        subgraph D1
            D11[Split by Markdown headings]
            D12[Divide into sentences with nltk.sent_tokenize]
        end
    end