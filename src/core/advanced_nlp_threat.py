"""
Advanced NLP Threat Detection System for ForenSnap Ultimate
Uses BERT/RoBERTa models for context understanding and sophisticated threat analysis
"""

import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
from typing import List, Dict, Any, Optional, Tuple
import logging
import datetime
from dataclasses import dataclass
from enum import Enum
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class ThreatCategory(str, Enum):
    """Detailed threat categories."""
    VIOLENCE_PHYSICAL = "violence_physical"
    VIOLENCE_VERBAL = "violence_verbal"
    HARASSMENT = "harassment"
    STALKING = "stalking"
    BLACKMAIL = "blackmail"
    EXTORTION = "extortion"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    TERRORISM = "terrorism"
    DRUG_RELATED = "drug_related"
    SEXUAL_THREAT = "sexual_threat"
    CYBERBULLYING = "cyberbullying"
    DOXXING = "doxxing"
    NONE = "none"

class ThreatSeverity(str, Enum):
    """Threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmotionalState(str, Enum):
    """Emotional states detected in text."""
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    DISGUST = "disgust"
    JOY = "joy"
    NEUTRAL = "neutral"
    ANXIETY = "anxiety"
    FRUSTRATION = "frustration"

@dataclass
class ThreatAnalysisResult:
    """Comprehensive threat analysis result."""
    overall_threat_level: ThreatSeverity
    threat_score: float
    categories: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, float]
    emotional_state: EmotionalState
    contextual_analysis: Dict[str, Any]
    escalation_indicators: List[str]
    urgency_indicators: List[str]
    target_analysis: Dict[str, Any]
    linguistic_patterns: Dict[str, Any]
    confidence_score: float
    warnings: List[str]
    recommendations: List[str]

class AdvancedNLPThreatDetector:
    """Advanced NLP-based threat detection system."""
    
    def __init__(self):
        """Initialize advanced threat detection models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.threat_classifier = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.nlp_model = None
        
        # Threat pattern databases
        self.threat_patterns = self._initialize_threat_patterns()
        self.escalation_patterns = self._initialize_escalation_patterns()
        self.urgency_patterns = self._initialize_urgency_patterns()
        self.target_patterns = self._initialize_target_patterns()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all NLP models."""
        try:
            logger.info("Initializing advanced NLP models...")
            
            # Load sentiment analysis model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Sentiment analysis model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentiment model: {e}")
                # Fallback to basic sentiment
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Load emotion detection model
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Emotion detection model loaded")
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
            
            # Load BERT model for contextual analysis
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=2  # Threat/No Threat
                ).to(self.device)
                logger.info("BERT model loaded for contextual analysis")
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {e}")
            
            # Load spaCy for linguistic analysis
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
            
            logger.info("Advanced NLP threat detection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
    
    def _initialize_threat_patterns(self) -> Dict[ThreatCategory, Dict[str, Any]]:
        """Initialize comprehensive threat pattern database."""
        return {
            ThreatCategory.VIOLENCE_PHYSICAL: {
                'keywords': [
                    'kill', 'murder', 'stab', 'shoot', 'beat', 'hit', 'punch', 
                    'attack', 'assault', 'harm', 'hurt', 'destroy', 'eliminate',
                    'knife', 'gun', 'weapon', 'blood', 'violence', 'fight'
                ],
                'patterns': [
                    r'\b(?:i|we|they)(?:\s+will|\s+gonna|\s+going to)?\s+(?:kill|murder|hurt|destroy)\s+(?:you|them|him|her)',
                    r'\b(?:kill|murder|hurt)\s+(?:you|them|him|her|yourself)',
                    r'\b(?:beat\s+up|knock\s+out|mess\s+up)\b',
                    r'\b(?:gun|knife|weapon)\s+(?:to|on|at)\s+(?:you|them)',
                    r'\bcome\s+(?:for|after)\s+you\b'
                ],
                'weight': 0.9,
                'severity_multiplier': 1.2
            },
            
            ThreatCategory.VIOLENCE_VERBAL: {
                'keywords': [
                    'threaten', 'intimidate', 'scare', 'terrorize', 'menace',
                    'warning', 'consequences', 'regret', 'sorry', 'pay'
                ],
                'patterns': [
                    r'\byou\s+(?:will|gonna|going to)\s+(?:regret|pay|be sorry)',
                    r'\bi\s+(?:will|gonna|going to)\s+make\s+you\s+(?:regret|pay|suffer)',
                    r'\b(?:this is|consider this)\s+(?:a warning|your warning)',
                    r'\byou\s+(?:better|should)\s+(?:watch|be careful)'
                ],
                'weight': 0.7,
                'severity_multiplier': 0.8
            },
            
            ThreatCategory.HARASSMENT: {
                'keywords': [
                    'follow', 'watch', 'stalk', 'bother', 'annoy', 'pester',
                    'obsessed', 'everywhere', 'always', 'constant', 'never leave'
                ],
                'patterns': [
                    r'\b(?:i|we)\s+(?:know|watch|follow|see)\s+(?:you|where you)',
                    r'\byou\s+(?:can\'t|cannot)\s+(?:hide|escape|run)',
                    r'\b(?:everywhere|anywhere)\s+you\s+go\b',
                    r'\bi\s+know\s+(?:where|what|who)\s+you\b'
                ],
                'weight': 0.8,
                'severity_multiplier': 0.9
            },
            
            ThreatCategory.BLACKMAIL: {
                'keywords': [
                    'expose', 'reveal', 'publish', 'release', 'leak', 'share',
                    'embarrass', 'ruin', 'destroy reputation', 'tell everyone',
                    'secrets', 'photos', 'videos', 'evidence'
                ],
                'patterns': [
                    r'\b(?:i|we)\s+(?:will|gonna|going to)\s+(?:expose|reveal|publish|release|tell)',
                    r'\b(?:pay|give|send)\s+(?:me|us)\s+or\s+(?:i|we)\s+will',
                    r'\bif\s+you\s+don\'?t\s+.*\s+(?:i|we)\s+will\s+(?:expose|reveal|tell)',
                    r'\byour\s+(?:secrets|photos|reputation)\s+(?:will|gonna)'
                ],
                'weight': 0.9,
                'severity_multiplier': 1.1
            },
            
            ThreatCategory.SELF_HARM: {
                'keywords': [
                    'kill myself', 'suicide', 'end my life', 'hurt myself',
                    'cut myself', 'overdose', 'jump', 'hang myself',
                    'better off dead', 'want to die', 'end it all'
                ],
                'patterns': [
                    r'\b(?:i|im|i\'m)\s+(?:going to|gonna)\s+(?:kill myself|commit suicide|end my life)',
                    r'\bi\s+(?:want to|wanna)\s+(?:die|kill myself|end it)',
                    r'\b(?:better off|rather be)\s+dead\b',
                    r'\bno\s+(?:point|reason)\s+(?:to|in)\s+(?:live|living)'
                ],
                'weight': 1.0,
                'severity_multiplier': 1.3
            },
            
            ThreatCategory.HATE_SPEECH: {
                'keywords': [
                    # Note: This would contain actual hate speech keywords in a real implementation
                    # For safety, using generic placeholders
                    'hate', 'inferior', 'subhuman', 'scum', 'vermin',
                    'racial', 'religious', 'ethnic', 'discrimination'
                ],
                'patterns': [
                    r'\b(?:all|every)\s+(?:person|people)\s+(?:like you|of your kind)',
                    r'\byou\s+(?:people|kind|type)\s+(?:are|deserve)',
                    r'\b(?:go back|get out|don\'t belong)'
                ],
                'weight': 0.8,
                'severity_multiplier': 1.0
            },
            
            ThreatCategory.CYBERBULLYING: {
                'keywords': [
                    'loser', 'pathetic', 'worthless', 'nobody', 'failure',
                    'ugly', 'stupid', 'idiot', 'freak', 'weirdo', 'reject'
                ],
                'patterns': [
                    r'\byou\s+(?:are|look)\s+(?:so|really|such a)\s+(?:ugly|stupid|pathetic)',
                    r'\b(?:nobody|everyone)\s+(?:likes|wants|cares about)\s+you',
                    r'\byou\s+(?:have no|don\'t have any)\s+(?:friends|life)',
                    r'\bkill\s+yourself\b'
                ],
                'weight': 0.6,
                'severity_multiplier': 0.7
            },
            
            ThreatCategory.SEXUAL_THREAT: {
                'keywords': [
                    'rape', 'assault', 'force', 'against your will',
                    'sexual', 'touch', 'inappropriate', 'unwanted'
                ],
                'patterns': [
                    r'\b(?:i|we)\s+(?:will|gonna|going to)\s+(?:force|make you)',
                    r'\bwait\s+until\s+(?:i|we)\s+(?:get|find|catch)\s+you',
                    r'\byou\s+(?:will|gonna)\s+(?:like|enjoy)\s+it'
                ],
                'weight': 1.0,
                'severity_multiplier': 1.4
            }
        }
    
    def _initialize_escalation_patterns(self) -> List[str]:
        """Initialize escalation indicator patterns."""
        return [
            r'\b(?:final|last|one more)\s+(?:warning|chance|time)\b',
            r'\b(?:time is|you have)\s+(?:running out|until|left)\b',
            r'\b(?:deadline|ultimatum|or else)\b',
            r'\b(?:this ends|enough|fed up|sick of)\b',
            r'\b(?:next time|if you|when i see you)\b',
            r'\b(?:you forced|made me|no choice)\b',
            r'\b(?:point of no return|crossed the line)\b',
            r'\b(?:prepare|get ready|brace yourself)\b'
        ]
    
    def _initialize_urgency_patterns(self) -> List[str]:
        """Initialize urgency indicator patterns."""
        return [
            r'\b(?:now|today|tonight|asap|immediately|right now)\b',
            r'\b(?:urgent|emergency|critical|important)\b',
            r'\b(?:hurry|quick|fast|soon|minutes|hours)\b',
            r'\b(?:before|by)\s+(?:\d+|tomorrow|tonight|today)\b',
            r'\b(?:time sensitive|can\'t wait|won\'t wait)\b',
            r'\b(?:act fast|move quick|respond now)\b'
        ]
    
    def _initialize_target_patterns(self) -> Dict[str, List[str]]:
        """Initialize target identification patterns."""
        return {
            'family': [
                r'\byour\s+(?:family|parents|children|kids|wife|husband|mom|dad|sister|brother)\b',
                r'\b(?:mother|father|son|daughter|spouse|relatives)\b'
            ],
            'location': [
                r'\b(?:at|near|in|outside)\s+(?:your|the)\s+(?:house|home|work|office|school)\b',
                r'\b(?:address|location|where you live|where you work)\b',
                r'\bi\s+know\s+where\s+you\s+(?:live|work|go|are)\b'
            ],
            'personal_info': [
                r'\bi\s+(?:have|know|found)\s+your\s+(?:address|number|info|details)\b',
                r'\byour\s+(?:real name|full name|identity|information)\b'
            ],
            'online_presence': [
                r'\byour\s+(?:account|profile|posts|photos|social media)\b',
                r'\b(?:facebook|instagram|twitter|social)\s+(?:account|profile)\b'
            ]
        }
    
    def analyze_threats(self, text: str) -> ThreatAnalysisResult:
        """
        Comprehensive threat analysis of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            ThreatAnalysisResult: Comprehensive threat analysis
        """
        try:
            if not text or len(text.strip()) < 3:
                return self._create_empty_result()
            
            # Perform all analyses
            sentiment = self._analyze_sentiment(text)
            emotion = self._analyze_emotion(text)
            contextual = self._analyze_context(text)
            threat_categories = self._detect_threat_categories(text)
            escalation = self._detect_escalation_indicators(text)
            urgency = self._detect_urgency_indicators(text)
            targets = self._analyze_targets(text)
            linguistic = self._analyze_linguistic_patterns(text)
            
            # Calculate overall threat score
            threat_score = self._calculate_overall_threat_score(
                threat_categories, escalation, urgency, sentiment, contextual
            )
            
            # Determine threat level
            threat_level = self._determine_threat_level(threat_score, threat_categories)
            
            # Generate warnings and recommendations
            warnings = self._generate_warnings(threat_categories, escalation, urgency, targets)
            recommendations = self._generate_recommendations(threat_level, threat_categories, targets)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                sentiment, contextual, threat_categories, linguistic
            )
            
            return ThreatAnalysisResult(
                overall_threat_level=threat_level,
                threat_score=threat_score,
                categories=threat_categories,
                sentiment_analysis=sentiment,
                emotional_state=emotion,
                contextual_analysis=contextual,
                escalation_indicators=escalation,
                urgency_indicators=urgency,
                target_analysis=targets,
                linguistic_patterns=linguistic,
                confidence_score=confidence,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using advanced models."""
        try:
            if self.sentiment_pipeline:
                result = self.sentiment_pipeline(text[:512])  # Limit text length
                
                if isinstance(result, list) and len(result) > 0:
                    sentiment_result = result[0]
                    label = sentiment_result.get('label', 'UNKNOWN').lower()
                    score = sentiment_result.get('score', 0.0)
                    
                    # Normalize to consistent format
                    sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                    
                    if 'positive' in label or label == 'pos':
                        sentiment_scores['positive'] = score
                        sentiment_scores['negative'] = 1 - score
                    elif 'negative' in label or label == 'neg':
                        sentiment_scores['negative'] = score
                        sentiment_scores['positive'] = 1 - score
                    else:
                        sentiment_scores['neutral'] = score
                    
                    return sentiment_scores
                    
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
        
        # Fallback to simple sentiment analysis
        return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Simple fallback sentiment analysis."""
        positive_words = [
            'good', 'great', 'happy', 'love', 'like', 'awesome', 'wonderful',
            'excellent', 'fantastic', 'amazing', 'perfect', 'thanks', 'pleased'
        ]
        
        negative_words = [
            'bad', 'hate', 'angry', 'mad', 'upset', 'sad', 'terrible',
            'awful', 'horrible', 'stupid', 'worst', 'disgusting', 'annoying'
        ]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total = len(words)
        
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = max(0, 1 - pos_score - neg_score)
        
        return {'positive': pos_score, 'negative': neg_score, 'neutral': neu_score}
    
    def _analyze_emotion(self, text: str) -> EmotionalState:
        """Analyze emotional state."""
        try:
            if self.emotion_pipeline:
                result = self.emotion_pipeline(text[:512])
                
                if isinstance(result, list) and len(result) > 0:
                    emotion_result = result[0]
                    emotion_label = emotion_result.get('label', 'neutral').lower()
                    
                    # Map to our emotion states
                    emotion_mapping = {
                        'anger': EmotionalState.ANGER,
                        'fear': EmotionalState.FEAR,
                        'sadness': EmotionalState.SADNESS,
                        'disgust': EmotionalState.DISGUST,
                        'joy': EmotionalState.JOY,
                        'surprise': EmotionalState.NEUTRAL,
                        'neutral': EmotionalState.NEUTRAL
                    }
                    
                    return emotion_mapping.get(emotion_label, EmotionalState.NEUTRAL)
                    
        except Exception as e:
            logger.debug(f"Emotion analysis failed: {e}")
        
        # Fallback emotion detection
        return self._simple_emotion_detection(text)
    
    def _simple_emotion_detection(self, text: str) -> EmotionalState:
        """Simple fallback emotion detection."""
        text_lower = text.lower()
        
        anger_indicators = ['angry', 'mad', 'furious', 'rage', 'hate', 'kill', 'destroy']
        fear_indicators = ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'panic']
        sadness_indicators = ['sad', 'depressed', 'hurt', 'cry', 'devastated', 'heartbroken']
        
        anger_score = sum(1 for word in anger_indicators if word in text_lower)
        fear_score = sum(1 for word in fear_indicators if word in text_lower)
        sadness_score = sum(1 for word in sadness_indicators if word in text_lower)
        
        if anger_score > fear_score and anger_score > sadness_score:
            return EmotionalState.ANGER
        elif fear_score > sadness_score:
            return EmotionalState.FEAR
        elif sadness_score > 0:
            return EmotionalState.SADNESS
        else:
            return EmotionalState.NEUTRAL
    
    def _analyze_context(self, text: str) -> Dict[str, Any]:
        """Analyze contextual information using BERT."""
        context_analysis = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'punctuation_intensity': len(re.findall(r'[!]{2,}|[?]{2,}', text)),
            'profanity_detected': False,  # Would use a profanity filter
            'urgency_markers': len(re.findall(r'\b(?:urgent|asap|now|immediately)\b', text, re.IGNORECASE)),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!')
        }
        
        # Additional context using BERT (if available)
        if self.bert_model and self.bert_tokenizer:
            try:
                inputs = self.bert_tokenizer(
                    text[:512], 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                    
                    context_analysis['bert_threat_probability'] = float(probabilities[0][1])
                    context_analysis['bert_confidence'] = float(torch.max(probabilities))
                    
            except Exception as e:
                logger.debug(f"BERT context analysis failed: {e}")
        
        return context_analysis
    
    def _detect_threat_categories(self, text: str) -> List[Dict[str, Any]]:
        """Detect specific threat categories."""
        detected_categories = []
        text_lower = text.lower()
        
        for category, patterns in self.threat_patterns.items():
            category_score = 0.0
            matched_keywords = []
            matched_patterns = []
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    category_score += 1
                    matched_keywords.append(keyword)
            
            # Check regex patterns
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    category_score += 2  # Patterns are weighted higher
                    matched_patterns.extend(matches)
            
            if category_score > 0:
                # Apply category weight and severity multiplier
                final_score = (category_score * patterns['weight'] * patterns['severity_multiplier']) / 10.0
                final_score = min(final_score, 1.0)  # Cap at 1.0
                
                detected_categories.append({
                    'category': category.value,
                    'score': final_score,
                    'matched_keywords': matched_keywords,
                    'matched_patterns': matched_patterns,
                    'severity_multiplier': patterns['severity_multiplier']
                })
        
        # Sort by score
        detected_categories.sort(key=lambda x: x['score'], reverse=True)
        return detected_categories
    
    def _detect_escalation_indicators(self, text: str) -> List[str]:
        """Detect escalation indicators."""
        indicators = []
        text_lower = text.lower()
        
        for pattern in self.escalation_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    def _detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators."""
        indicators = []
        text_lower = text.lower()
        
        for pattern in self.urgency_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    def _analyze_targets(self, text: str) -> Dict[str, Any]:
        """Analyze mentioned targets."""
        targets = {}
        text_lower = text.lower()
        
        for target_type, patterns in self.target_patterns.items():
            matches = []
            for pattern in patterns:
                pattern_matches = re.findall(pattern, text_lower)
                if pattern_matches:
                    matches.extend(pattern_matches)
            
            if matches:
                targets[target_type] = {
                    'mentions': len(matches),
                    'examples': matches[:5]  # Limit examples
                }
        
        return targets
    
    def _analyze_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns using spaCy."""
        patterns = {
            'entities': [],
            'pos_tags': {},
            'dependency_patterns': [],
            'named_entities': []
        }
        
        if self.nlp_model:
            try:
                doc = self.nlp_model(text[:1000])  # Limit text length
                
                # Extract named entities
                for ent in doc.ents:
                    patterns['named_entities'].append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                
                # Extract POS tag distribution
                pos_counts = Counter([token.pos_ for token in doc])
                patterns['pos_tags'] = dict(pos_counts.most_common(10))
                
                # Look for threatening sentence structures
                for sent in doc.sents:
                    if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
                        root_verbs = [token.lemma_ for token in sent if token.dep_ == 'ROOT' and token.pos_ == 'VERB']
                        if any(verb in ['threaten', 'kill', 'hurt', 'destroy', 'harm'] for verb in root_verbs):
                            patterns['dependency_patterns'].append(sent.text)
                
            except Exception as e:
                logger.debug(f"Linguistic analysis failed: {e}")
        
        return patterns
    
    def _calculate_overall_threat_score(self, categories: List[Dict], escalation: List[str],
                                      urgency: List[str], sentiment: Dict[str, float],
                                      contextual: Dict[str, Any]) -> float:
        """Calculate overall threat score."""
        base_score = 0.0
        
        # Category-based score (40% weight)
        if categories:
            category_score = sum(cat['score'] * cat['severity_multiplier'] for cat in categories)
            category_score = min(category_score / len(categories), 1.0)
            base_score += category_score * 0.4
        
        # Escalation indicators (25% weight)
        escalation_score = min(len(escalation) / 3.0, 1.0)  # Normalize by 3 indicators
        base_score += escalation_score * 0.25
        
        # Urgency indicators (15% weight)
        urgency_score = min(len(urgency) / 3.0, 1.0)  # Normalize by 3 indicators
        base_score += urgency_score * 0.15
        
        # Negative sentiment (10% weight)
        sentiment_score = sentiment.get('negative', 0.0)
        base_score += sentiment_score * 0.1
        
        # Contextual factors (10% weight)
        context_score = 0.0
        if contextual.get('uppercase_ratio', 0) > 0.3:
            context_score += 0.2
        if contextual.get('punctuation_intensity', 0) > 0:
            context_score += 0.3
        if contextual.get('bert_threat_probability', 0) > 0.5:
            context_score += 0.5
        
        context_score = min(context_score, 1.0)
        base_score += context_score * 0.1
        
        return min(base_score, 1.0)
    
    def _determine_threat_level(self, score: float, categories: List[Dict]) -> ThreatSeverity:
        """Determine threat severity level."""
        # Check for critical categories
        critical_categories = [
            ThreatCategory.VIOLENCE_PHYSICAL, ThreatCategory.SELF_HARM,
            ThreatCategory.SEXUAL_THREAT, ThreatCategory.TERRORISM
        ]
        
        has_critical = any(cat['category'] in [c.value for c in critical_categories] for cat in categories)
        
        if score >= 0.8 or has_critical:
            return ThreatSeverity.CRITICAL
        elif score >= 0.6:
            return ThreatSeverity.HIGH
        elif score >= 0.4:
            return ThreatSeverity.MEDIUM
        elif score >= 0.2:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.NONE
    
    def _generate_warnings(self, categories: List[Dict], escalation: List[str],
                          urgency: List[str], targets: Dict) -> List[str]:
        """Generate contextual warnings."""
        warnings = []
        
        # Category-based warnings
        for cat in categories:
            if cat['score'] > 0.7:
                warnings.append(f"High-confidence {cat['category'].replace('_', ' ')} threat detected")
        
        # Escalation warnings
        if len(escalation) > 2:
            warnings.append("Multiple escalation indicators detected - immediate attention required")
        
        # Urgency warnings
        if len(urgency) > 1:
            warnings.append("Urgent language detected - time-sensitive threat possible")
        
        # Target warnings
        if 'family' in targets:
            warnings.append("Family members mentioned as potential targets")
        if 'location' in targets:
            warnings.append("Specific locations mentioned - physical threat possible")
        
        return warnings
    
    def _generate_recommendations(self, threat_level: ThreatSeverity,
                                categories: List[Dict], targets: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if threat_level in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
            recommendations.append("Contact law enforcement immediately")
            recommendations.append("Document all evidence and maintain chain of custody")
            recommendations.append("Consider emergency protective measures")
        
        elif threat_level == ThreatSeverity.MEDIUM:
            recommendations.append("Report to appropriate authorities")
            recommendations.append("Increase security awareness")
            recommendations.append("Monitor for escalation")
        
        elif threat_level == ThreatSeverity.LOW:
            recommendations.append("Document for future reference")
            recommendations.append("Monitor subject's communications")
        
        # Category-specific recommendations
        for cat in categories:
            if cat['category'] == ThreatCategory.SELF_HARM.value:
                recommendations.append("Contact mental health crisis services")
                recommendations.append("Implement suicide prevention protocols")
        
        # Target-specific recommendations
        if 'family' in targets:
            recommendations.append("Notify family members of potential threat")
        if 'location' in targets:
            recommendations.append("Enhance physical security at mentioned locations")
        
        return recommendations
    
    def _calculate_confidence(self, sentiment: Dict, contextual: Dict,
                            categories: List[Dict], linguistic: Dict) -> float:
        """Calculate confidence score for the analysis."""
        confidence_factors = []
        
        # Sentiment confidence
        max_sentiment = max(sentiment.values())
        confidence_factors.append(max_sentiment)
        
        # BERT confidence (if available)
        if 'bert_confidence' in contextual:
            confidence_factors.append(contextual['bert_confidence'])
        
        # Category match confidence
        if categories:
            avg_category_confidence = sum(cat['score'] for cat in categories) / len(categories)
            confidence_factors.append(avg_category_confidence)
        
        # Text length factor (longer texts generally more reliable)
        text_length_factor = min(contextual.get('word_count', 0) / 50.0, 1.0)
        confidence_factors.append(text_length_factor)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _create_empty_result(self) -> ThreatAnalysisResult:
        """Create empty result for minimal/no input."""
        return ThreatAnalysisResult(
            overall_threat_level=ThreatSeverity.NONE,
            threat_score=0.0,
            categories=[],
            sentiment_analysis={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            emotional_state=EmotionalState.NEUTRAL,
            contextual_analysis={},
            escalation_indicators=[],
            urgency_indicators=[],
            target_analysis={},
            linguistic_patterns={},
            confidence_score=0.0,
            warnings=[],
            recommendations=[]
        )
    
    def _create_error_result(self, error: str) -> ThreatAnalysisResult:
        """Create error result."""
        return ThreatAnalysisResult(
            overall_threat_level=ThreatSeverity.NONE,
            threat_score=0.0,
            categories=[],
            sentiment_analysis={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            emotional_state=EmotionalState.NEUTRAL,
            contextual_analysis={'error': error},
            escalation_indicators=[],
            urgency_indicators=[],
            target_analysis={},
            linguistic_patterns={},
            confidence_score=0.0,
            warnings=[f"Analysis failed: {error}"],
            recommendations=["Manual review required due to analysis error"]
        )

# Example usage
if __name__ == "__main__":
    detector = AdvancedNLPThreatDetector()
    
    # Test threat detection
    # result = detector.analyze_threats("I'm going to hurt you if you don't stop")
    # print(json.dumps(result.__dict__, indent=2, default=str))
