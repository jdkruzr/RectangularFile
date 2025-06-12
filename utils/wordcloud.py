def get_document_texts(db, doc_id=None, folder=None):
    """
    Extract text from documents for word cloud generation.
    
    Args:
        db: DatabaseManager instance
        doc_id: Optional specific document ID
        folder: Optional folder filter
    
    Returns:
        List of text content from documents
    """
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Build query based on filters
        query = """
            SELECT d.id, tc.text_content, tc.ocr_text 
            FROM pdf_documents d
            JOIN pdf_text_content tc ON d.id = tc.pdf_id
            WHERE d.processing_status = 'completed'
        """
        params = []
        
        if doc_id:
            query += " AND d.id = ?"
            params.append(doc_id)
            
        if folder:
            query += " AND d.folder_path LIKE ?"
            params.append(f"%{folder}%")
            
        cursor.execute(query, params)
        
        all_text = []
        for row in cursor.fetchall():
            # Prefer OCR text if available
            if row['ocr_text'] and row['ocr_text'].strip():
                all_text.append(row['ocr_text'])
            elif row['text_content'] and row['text_content'].strip():
                all_text.append(row['text_content'])
                
        return all_text
    
def process_text_for_wordcloud(texts):
    """
    Process a list of texts to prepare for word cloud generation.
    
    Args:
        texts: List of text strings
        
    Returns:
        Processed text ready for word cloud
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Combine all texts
    combined_text = " ".join(texts)
    
    # Tokenize
    tokens = word_tokenize(combined_text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Return processed text
    return " ".join(tokens)

def generate_wordcloud(text, width=800, height=400):
    """
    Generate a word cloud from the provided text.
    
    Args:
        text: Processed text string
        width: Width of the word cloud image
        height: Height of the word cloud image
        
    Returns:
        WordCloud object and the image as a bytes object
    """
    from wordcloud import WordCloud
    import io
    from matplotlib import pyplot as plt
    import numpy as np
    from PIL import Image
    
    # Create mask (optional - for shaped word clouds)
    # mask = np.array(Image.open("static/mask.png"))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=200,
        contour_width=1,
        contour_color='steelblue',
        # mask=mask,  # Uncomment if using a mask
        collocations=False  # Avoid repeating word pairs
    ).generate(text)
    
    # Convert to image bytes
    img_bytes = io.BytesIO()
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)
    
    return wordcloud, img_bytes