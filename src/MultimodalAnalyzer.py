import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pytesseract
from PIL import Image
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List, Tuple, Optional
import math

class MultimodalAnalysisEngine:
    def __init__(self):
        """Initialize the multimodal analysis engine with pre-trained models."""
        # Load models
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Load NER model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize TF-IDF for word importance
        self.tfidf_vectorizer = TfidfVectorizer()
        
    def calculate_overall_quality_score(self, q_image: float, q_video: float, q_text: float, 
                                      alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> float:
        """
        Calculate overall quality score using weighted formula.
        Q_total = α·Q_image + β·Q_video + γ·Q_text
        """
        if not abs(alpha + beta + gamma - 1.0) < 1e-6:
            raise ValueError("Alpha, beta, and gamma must sum to 1.0")
        
        return alpha * q_image + beta * q_video + gamma * q_text
    
    def extract_text_from_image(self, image: np.ndarray, word_corpus: List[str] = None) -> Dict:
        """
        Extract text from image using OCR with confidence scoring.
        Text_confidence = Σ(i=1 to n) w_i · conf_i / n
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Perform OCR with confidence scores
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        confidences = []
        word_importance_weights = []
        
        # Extract words and confidences
        for i, word in enumerate(ocr_data['text']):
            if word.strip():  # Skip empty strings
                words.append(word)
                confidences.append(float(ocr_data['conf'][i]) / 100.0)  # Normalize to [0,1]
        
        # Calculate word importance weights using TF-IDF if corpus provided
        if word_corpus and len(words) > 0:
            try:
                # Fit TF-IDF on corpus and transform current words
                self.tfidf_vectorizer.fit(word_corpus)
                word_text = ' '.join(words)
                tfidf_matrix = self.tfidf_vectorizer.transform([word_text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores for each word
                for word in words:
                    if word.lower() in feature_names:
                        idx = list(feature_names).index(word.lower())
                        weight = tfidf_matrix[0, idx]
                    else:
                        weight = 0.1  # Default weight for unknown words
                    word_importance_weights.append(weight)
            except:
                # Fallback to uniform weights
                word_importance_weights = [1.0] * len(words)
        else:
            word_importance_weights = [1.0] * len(words)
        
        # Calculate text confidence
        if len(words) > 0:
            text_confidence = sum(w * c for w, c in zip(word_importance_weights, confidences)) / len(words)
        else:
            text_confidence = 0.0
        
        return {
            'extracted_text': ' '.join(words),
            'words': words,
            'confidences': confidences,
            'word_weights': word_importance_weights,
            'text_confidence': text_confidence
        }
    
    def extract_menu_structure(self, extracted_text: str) -> List[Dict]:
        """
        Extract menu items with structure: {item_name, price, description, availability_status}
        """
        menu_items = []
        
        # Price pattern - matches various currency formats
        price_pattern = r'\$?\d+\.?\d*|\d+\.\d{2}'
        
        # Split text into potential menu items (by lines or semicolons)
        potential_items = re.split(r'\n|;|\.\.+', extracted_text)
        
        for item_text in potential_items:
            item_text = item_text.strip()
            if len(item_text) < 5:  # Skip very short text
                continue
            
            # Find prices in the item
            prices = re.findall(price_pattern, item_text)
            
            # Extract entities using NER
            doc = self.nlp(item_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Determine availability (look for keywords)
            availability_keywords = ['out of stock', 'unavailable', 'sold out', '86ed', 'not available']
            availability_status = 'available'
            for keyword in availability_keywords:
                if keyword.lower() in item_text.lower():
                    availability_status = 'unavailable'
                    break
            
            # Extract item name (first part before price or description)
            item_name = item_text
            description = ""
            
            if prices:
                # Split on first price occurrence
                price_match = re.search(price_pattern, item_text)
                if price_match:
                    item_name = item_text[:price_match.start()].strip()
                    description = item_text[price_match.end():].strip()
            
            menu_item = {
                'item_name': item_name,
                'price': prices[0] if prices else None,
                'description': description,
                'availability_status': availability_status,
                'entities': entities
            }
            
            menu_items.append(menu_item)
        
        return menu_items
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer-based models.
        Sentiment score computed as softmax(W_s · h_final + b_s)
        """
        # Tokenize and encode text
        inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Map to sentiment labels (assuming negative, neutral, positive)
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment_scores = predictions.cpu().numpy()[0]
        
        # Get primary sentiment
        primary_sentiment_idx = np.argmax(sentiment_scores)
        primary_sentiment = sentiment_labels[primary_sentiment_idx]
        confidence = float(sentiment_scores[primary_sentiment_idx])
        
        return {
            'sentiment': primary_sentiment,
            'confidence': confidence,
            'scores': {label: float(score) for label, score in zip(sentiment_labels, sentiment_scores)}
        }
    
    def assess_review_quality(self, text: str, lambda1: float = 0.4, lambda2: float = 0.3, lambda3: float = 0.3) -> float:
        """
        Assess review quality using multiple factors.
        Review_quality = λ₁·Sentiment_coherence + λ₂·Factual_accuracy + λ₃·Temporal_relevance
        """
        if not abs(lambda1 + lambda2 + lambda3 - 1.0) < 1e-6:
            raise ValueError("Lambda coefficients must sum to 1.0")
        
        # Sentiment coherence - consistency of sentiment throughout text
        sentences = re.split(r'[.!?]+', text)
        sentiments = []
        for sentence in sentences:
            if sentence.strip():
                sent_analysis = self.analyze_sentiment(sentence.strip())
                sentiments.append(sent_analysis['scores'])
        
        # Calculate sentiment coherence (lower variance = higher coherence)
        if sentiments:
            sentiment_variance = np.var([s['positive'] - s['negative'] for s in sentiments])
            sentiment_coherence = 1.0 / (1.0 + sentiment_variance)
        else:
            sentiment_coherence = 0.5
        
        # Factual accuracy proxy - presence of specific details
        factual_indicators = len(re.findall(r'\b\d+\b|\$\d+|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text.lower()))
        factual_accuracy = min(1.0, factual_indicators / 5.0)  # Normalize
        
        # Temporal relevance - assume recent reviews are more relevant
        temporal_keywords = ['today', 'yesterday', 'recently', 'just', 'now', 'this week']
        temporal_relevance = min(1.0, sum(1 for keyword in temporal_keywords if keyword in text.lower()) / len(temporal_keywords))
        
        review_quality = (lambda1 * sentiment_coherence + 
                         lambda2 * factual_accuracy + 
                         lambda3 * temporal_relevance)
        
        return review_quality
    
    def calculate_relevance_score(self, review_text: str, business_profile: str, 
                                t_current: float, t_submission: float, delta: float = 0.1) -> float:
        """
        Calculate relevance score with semantic similarity and temporal decay.
        Relevance(r,t) = Semantic_similarity(r, business_profile) · e^(-δ·(t_current - t_submission))
        """
        # Calculate semantic similarity using CLIP text encoder
        inputs = self.clip_processor(text=[review_text, business_profile], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(text_embeddings[0:1], text_embeddings[1:2])
        semantic_similarity = float(similarity[0])
        
        # Apply temporal decay
        time_decay = math.exp(-delta * (t_current - t_submission))
        
        relevance_score = semantic_similarity * time_decay
        return max(0.0, relevance_score)  # Ensure non-negative

# Example usage
if __name__ == "__main__":
    engine = MultimodalAnalysisEngine()
    
    # Example quality score calculation
    q_total = engine.calculate_overall_quality_score(0.8, 0.7, 0.9)
    print(f"Overall Quality Score: {q_total}")
    
    # Example sentiment analysis
    sample_text = "The food was absolutely delicious and the service was outstanding!"
    sentiment_result = engine.analyze_sentiment(sample_text)
    print(f"Sentiment Analysis: {sentiment_result}")
    
    # Example review quality assessment
    review_quality = engine.assess_review_quality(sample_text)
    print(f"Review Quality Score: {review_quality}")
