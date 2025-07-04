# utils/wordcloud.py
import re
import string
import logging

def get_document_texts(db, doc_id=None, folder=None, device=None, category=None):
    """
    Extract text from documents for word cloud generation.
    
    Args:
        db: DatabaseManager instance
        doc_id: Optional specific document ID
        folder: Optional folder filter
        device: Optional device filter
        category: Optional category filter
    
    Returns:
        List of text content from documents
    """

    logger = logging.getLogger(__name__)
    
    logger.info(f"Getting document texts - doc_id: {doc_id}, folder: {folder}, device: {device}, category: {category}")
    
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
            query += " AND d.folder_path = ?"
            params.append(folder)
            
        if device:
            # Assuming device is the first component of the folder path
            query += " AND d.folder_path LIKE ?"
            params.append(f"{device}/%")
            
        if category:
            if category == 'Moffitt':
                # Special case for Moffitt
                query += " AND d.folder_path LIKE ?"
                params.append(f"%Moffitt%")
            else:
                # Match category either as first component or second component
                query += " AND (d.folder_path = ? OR d.folder_path LIKE ?)"
                params.append(category)
                params.append(f"%/{category}/%")
        
        logger.info(f"Query: {query}")
        logger.info(f"Params: {params}")
            
        cursor.execute(query, params)
        
        all_text = []
        for row in cursor.fetchall():
            # Prefer OCR text if available
            if row['ocr_text'] and row['ocr_text'].strip():
                all_text.append(row['ocr_text'])
            elif row['text_content'] and row['text_content'].strip():
                all_text.append(row['text_content'])
        
        logger.info(f"Found {len(all_text)} text items")
        return all_text

def process_text_for_wordcloud(texts):
    """
    Process a list of texts to prepare for word cloud generation without NLTK.
    
    Args:
        texts: List of text strings
        
    Returns:
        Processed text ready for word cloud
    """
    # Common English stopwords
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
        't', 'can', 'will', 'just', 'don', 'should', 'now'
    }
    
    # Combine all texts
    combined_text = " ".join(texts).lower()
    
    # Remove punctuation
    combined_text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', combined_text)
    
    # Split into words
    words = combined_text.split()
    
    # Filter words
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Return processed text
    return " ".join(filtered_words)

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
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=200,
        contour_width=1,
        contour_color='steelblue',
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