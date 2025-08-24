"""
Advanced Search and Filtering System for ForenSnap Ultimate
Sophisticated search capabilities for digital forensics investigations
"""

import re
import json
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, not_, func, text

# Configure logging
logger = logging.getLogger(__name__)

class SearchOperator(str, Enum):
    """Search operators for complex queries."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"
    REGEX = "REGEX"
    FUZZY = "FUZZY"
    SIMILAR = "SIMILAR"
    RANGE = "RANGE"
    IN = "IN"
    LIKE = "LIKE"

class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "ASC"
    DESC = "DESC"

class SearchField(str, Enum):
    """Available search fields."""
    TEXT = "detected_text"
    CATEGORY = "category"
    PLATFORM = "platform"
    THREAT_LEVEL = "threat_level"
    LANGUAGE = "detected_language"
    TAGS = "tags"
    FILENAME = "file_path"
    HASH = "file_hash"
    DATE = "created_at"
    CONFIDENCE = "confidence_score"
    NSFW_SCORE = "nsfw_score"
    FILE_SIZE = "file_size"

@dataclass
class SearchFilter:
    """Individual search filter."""
    field: SearchField
    operator: SearchOperator
    value: Union[str, int, float, List[str], Tuple[Any, Any]]
    weight: float = 1.0
    case_sensitive: bool = False

@dataclass
class SearchQuery:
    """Complex search query with multiple filters."""
    filters: List[SearchFilter]
    logical_operator: SearchOperator = SearchOperator.AND
    sort_by: Optional[SearchField] = None
    sort_order: SortOrder = SortOrder.DESC
    limit: int = 100
    offset: int = 0
    include_similarity: bool = False
    similarity_threshold: float = 0.7

@dataclass
class SearchResult:
    """Individual search result with metadata."""
    image_id: str
    relevance_score: float
    match_details: Dict[str, Any]
    similarity_score: Optional[float] = None
    highlighted_text: Optional[str] = None

class AdvancedSearchEngine:
    """Advanced search engine with multiple search capabilities."""
    
    def __init__(self, session_factory, vectorizer=None):
        """
        Initialize advanced search engine.
        
        Args:
            session_factory: SQLAlchemy session factory
            vectorizer: Optional pre-trained TF-IDF vectorizer
        """
        self.Session = session_factory
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.text_vectors = None
        self.image_texts = {}
        self.fuzzy_threshold = 80  # Minimum fuzzy match score
        
        self._initialize_text_search()
    
    def _initialize_text_search(self):
        """Initialize text-based search capabilities."""
        try:
            session = self.Session()
            
            # Load all text data for vectorization
            from forensnap_ultimate import Image  # Import here to avoid circular imports
            
            images = session.query(Image).filter(Image.detected_text.isnot(None)).all()
            
            if images:
                texts = []
                for img in images:
                    text = img.detected_text or ""
                    texts.append(text)
                    self.image_texts[img.id] = text
                
                # Fit vectorizer on all texts
                if texts:
                    self.text_vectors = self.vectorizer.fit_transform(texts)
                    logger.info(f"Text search initialized with {len(texts)} documents")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize text search: {e}")
    
    def execute_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute complex search query.
        
        Args:
            query: SearchQuery object with filters and parameters
            
        Returns:
            List[SearchResult]: Sorted search results
        """
        try:
            session = self.Session()
            from forensnap_ultimate import Image, Tag, ImageTag
            
            # Start with base query
            base_query = session.query(Image)
            
            # Apply filters
            conditions = []
            
            for search_filter in query.filters:
                condition = self._build_condition(search_filter, Image, Tag, ImageTag)
                if condition is not None:
                    conditions.append(condition)
            
            # Combine conditions with logical operator
            if conditions:
                if query.logical_operator == SearchOperator.AND:
                    base_query = base_query.filter(and_(*conditions))
                elif query.logical_operator == SearchOperator.OR:
                    base_query = base_query.filter(or_(*conditions))
            
            # Apply sorting
            if query.sort_by:
                sort_column = getattr(Image, query.sort_by.value, None)
                if sort_column:
                    if query.sort_order == SortOrder.DESC:
                        base_query = base_query.order_by(sort_column.desc())
                    else:
                        base_query = base_query.order_by(sort_column.asc())
            
            # Apply pagination
            base_query = base_query.offset(query.offset).limit(query.limit)
            
            # Execute query
            results = base_query.all()
            
            # Convert to SearchResult objects with relevance scoring
            search_results = []
            
            for img in results:
                relevance_score = self._calculate_relevance_score(img, query.filters)
                match_details = self._extract_match_details(img, query.filters)
                
                result = SearchResult(
                    image_id=img.id,
                    relevance_score=relevance_score,
                    match_details=match_details
                )
                
                # Add similarity if requested
                if query.include_similarity and query.filters:
                    text_filter = next((f for f in query.filters if f.field == SearchField.TEXT), None)
                    if text_filter and isinstance(text_filter.value, str):
                        similarity = self._calculate_text_similarity(img.detected_text or "", text_filter.value)
                        result.similarity_score = similarity
                
                # Add text highlighting
                result.highlighted_text = self._highlight_matches(img.detected_text or "", query.filters)
                
                search_results.append(result)
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            session.close()
            return search_results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def _build_condition(self, search_filter: SearchFilter, Image, Tag, ImageTag):
        """Build SQLAlchemy condition from search filter."""
        try:
            field_name = search_filter.field.value
            operator = search_filter.operator
            value = search_filter.value
            
            # Get the column
            if hasattr(Image, field_name):
                column = getattr(Image, field_name)
            else:
                logger.warning(f"Unknown field: {field_name}")
                return None
            
            # Build condition based on operator
            if operator == SearchOperator.CONTAINS:
                if not search_filter.case_sensitive:
                    return func.lower(column).contains(str(value).lower())
                return column.contains(str(value))
            
            elif operator == SearchOperator.STARTS_WITH:
                if not search_filter.case_sensitive:
                    return func.lower(column).like(f"{str(value).lower()}%")
                return column.like(f"{value}%")
            
            elif operator == SearchOperator.ENDS_WITH:
                if not search_filter.case_sensitive:
                    return func.lower(column).like(f"%{str(value).lower()}")
                return column.like(f"%{value}")
            
            elif operator == SearchOperator.REGEX:
                return column.op('REGEXP')(str(value))
            
            elif operator == SearchOperator.IN:
                if isinstance(value, (list, tuple)):
                    return column.in_(value)
                return column == value
            
            elif operator == SearchOperator.RANGE:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return column.between(value[0], value[1])
                return None
            
            elif operator == SearchOperator.LIKE:
                if not search_filter.case_sensitive:
                    return func.lower(column).like(f"%{str(value).lower()}%")
                return column.like(f"%{value}%")
            
            elif operator == SearchOperator.NOT:
                return column != value
            
            else:  # Default to equality
                return column == value
            
        except Exception as e:
            logger.error(f"Failed to build condition: {e}")
            return None
    
    def _calculate_relevance_score(self, image, filters: List[SearchFilter]) -> float:
        """Calculate relevance score for image based on filters."""
        total_score = 0.0
        total_weight = 0.0
        
        for search_filter in filters:
            weight = search_filter.weight
            score = 0.0
            
            # Calculate individual filter score
            if search_filter.field == SearchField.TEXT:
                score = self._score_text_match(image.detected_text or "", search_filter)
            elif search_filter.field == SearchField.CATEGORY:
                score = 1.0 if image.category == search_filter.value else 0.0
            elif search_filter.field == SearchField.PLATFORM:
                score = 1.0 if image.platform == search_filter.value else 0.0
            elif search_filter.field == SearchField.THREAT_LEVEL:
                score = 1.0 if image.threat_level == search_filter.value else 0.0
            elif search_filter.field == SearchField.CONFIDENCE:
                if search_filter.operator == SearchOperator.RANGE and isinstance(search_filter.value, (list, tuple)):
                    min_val, max_val = search_filter.value
                    if min_val <= image.confidence_score <= max_val:
                        score = 1.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1.0)
    
    def _score_text_match(self, text: str, search_filter: SearchFilter) -> float:
        """Score text match based on operator."""
        if not text or not search_filter.value:
            return 0.0
        
        search_text = str(search_filter.value)
        
        if search_filter.operator == SearchOperator.FUZZY:
            return fuzz.partial_ratio(text.lower(), search_text.lower()) / 100.0
        
        elif search_filter.operator == SearchOperator.CONTAINS:
            if not search_filter.case_sensitive:
                return 1.0 if search_text.lower() in text.lower() else 0.0
            return 1.0 if search_text in text else 0.0
        
        elif search_filter.operator == SearchOperator.REGEX:
            try:
                pattern = re.compile(search_text, re.IGNORECASE if not search_filter.case_sensitive else 0)
                matches = pattern.findall(text)
                return min(len(matches) / 10.0, 1.0)  # Normalize to 0-1
            except:
                return 0.0
        
        elif search_filter.operator == SearchOperator.SIMILAR:
            return self._calculate_text_similarity(text, search_text)
        
        return 0.5  # Default score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF cosine similarity."""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Vectorize the texts
            vectors = self.vectorizer.transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"Similarity calculation failed: {e}")
            # Fallback to simple fuzzy matching
            return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
    
    def _extract_match_details(self, image, filters: List[SearchFilter]) -> Dict[str, Any]:
        """Extract details about what matched in the search."""
        details = {}
        
        for search_filter in filters:
            field_name = search_filter.field.value
            
            if search_filter.field == SearchField.TEXT:
                text = image.detected_text or ""
                if search_filter.operator == SearchOperator.REGEX:
                    try:
                        pattern = re.compile(str(search_filter.value), re.IGNORECASE)
                        matches = pattern.findall(text)
                        details[f"{field_name}_matches"] = matches[:10]  # Limit matches
                    except:
                        pass
                elif search_filter.operator == SearchOperator.FUZZY:
                    score = fuzz.partial_ratio(text.lower(), str(search_filter.value).lower())
                    details[f"{field_name}_fuzzy_score"] = score
            
            # Add matched value
            if hasattr(image, field_name):
                details[field_name] = getattr(image, field_name)
        
        return details
    
    def _highlight_matches(self, text: str, filters: List[SearchFilter]) -> str:
        """Highlight matching text in results."""
        if not text:
            return ""
        
        highlighted = text
        
        for search_filter in filters:
            if search_filter.field == SearchField.TEXT and search_filter.value:
                search_term = str(search_filter.value)
                
                if search_filter.operator == SearchOperator.CONTAINS:
                    if not search_filter.case_sensitive:
                        # Case-insensitive replacement
                        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
                        highlighted = pattern.sub(f"**{search_term}**", highlighted)
                    else:
                        highlighted = highlighted.replace(search_term, f"**{search_term}**")
                
                elif search_filter.operator == SearchOperator.REGEX:
                    try:
                        pattern = re.compile(search_term, re.IGNORECASE if not search_filter.case_sensitive else 0)
                        highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)
                    except:
                        pass
        
        return highlighted
    
    def fuzzy_search(self, query_text: str, field: SearchField = SearchField.TEXT, 
                    limit: int = 10) -> List[SearchResult]:
        """
        Perform fuzzy text search.
        
        Args:
            query_text: Text to search for
            field: Field to search in
            limit: Maximum results
            
        Returns:
            List[SearchResult]: Fuzzy search results
        """
        try:
            session = self.Session()
            from forensnap_ultimate import Image
            
            # Get all images with text
            images = session.query(Image).filter(Image.detected_text.isnot(None)).all()
            
            # Perform fuzzy matching
            fuzzy_results = []
            
            for img in images:
                if field == SearchField.TEXT:
                    target_text = img.detected_text or ""
                elif field == SearchField.FILENAME:
                    target_text = img.file_path or ""
                else:
                    continue
                
                # Calculate fuzzy score
                score = fuzz.partial_ratio(query_text.lower(), target_text.lower())
                
                if score >= self.fuzzy_threshold:
                    fuzzy_results.append({
                        'image': img,
                        'score': score,
                        'matched_text': target_text[:200]  # First 200 chars
                    })
            
            # Sort by fuzzy score
            fuzzy_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Convert to SearchResult objects
            results = []
            for item in fuzzy_results[:limit]:
                result = SearchResult(
                    image_id=item['image'].id,
                    relevance_score=item['score'] / 100.0,
                    match_details={
                        'fuzzy_score': item['score'],
                        'matched_text': item['matched_text']
                    }
                )
                results.append(result)
            
            session.close()
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def similarity_search(self, reference_image_id: str, threshold: float = 0.7,
                         limit: int = 20) -> List[SearchResult]:
        """
        Find similar images based on text content.
        
        Args:
            reference_image_id: ID of reference image
            threshold: Minimum similarity threshold
            limit: Maximum results
            
        Returns:
            List[SearchResult]: Similar images
        """
        try:
            session = self.Session()
            from forensnap_ultimate import Image
            
            # Get reference image
            ref_image = session.query(Image).filter_by(id=reference_image_id).first()
            if not ref_image or not ref_image.detected_text:
                return []
            
            ref_text = ref_image.detected_text
            
            # Get all other images
            other_images = session.query(Image).filter(
                Image.id != reference_image_id,
                Image.detected_text.isnot(None)
            ).all()
            
            # Calculate similarities
            similarities = []
            
            for img in other_images:
                similarity = self._calculate_text_similarity(ref_text, img.detected_text or "")
                
                if similarity >= threshold:
                    similarities.append({
                        'image': img,
                        'similarity': similarity
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Convert to SearchResult objects
            results = []
            for item in similarities[:limit]:
                result = SearchResult(
                    image_id=item['image'].id,
                    relevance_score=item['similarity'],
                    similarity_score=item['similarity'],
                    match_details={
                        'reference_image': reference_image_id,
                        'text_similarity': item['similarity']
                    }
                )
                results.append(result)
            
            session.close()
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    def date_range_search(self, start_date: datetime.datetime, end_date: datetime.datetime,
                         additional_filters: Optional[List[SearchFilter]] = None) -> List[SearchResult]:
        """
        Search by date range with optional additional filters.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            additional_filters: Optional additional search filters
            
        Returns:
            List[SearchResult]: Images in date range
        """
        date_filter = SearchFilter(
            field=SearchField.DATE,
            operator=SearchOperator.RANGE,
            value=(start_date, end_date)
        )
        
        filters = [date_filter]
        if additional_filters:
            filters.extend(additional_filters)
        
        query = SearchQuery(filters=filters)
        return self.execute_search(query)
    
    def advanced_query_parser(self, query_string: str) -> SearchQuery:
        """
        Parse advanced query string into SearchQuery object.
        
        Supported syntax:
        - text:contains:"search term"
        - category:equals:chat
        - threat_level:in:high,critical
        - confidence:range:0.5,1.0
        - date:range:2024-01-01,2024-12-31
        
        Args:
            query_string: Query string to parse
            
        Returns:
            SearchQuery: Parsed query object
        """
        filters = []
        
        # Split by logical operators (simplified parsing)
        parts = re.split(r'\s+(AND|OR)\s+', query_string, flags=re.IGNORECASE)
        logical_op = SearchOperator.AND  # Default
        
        for part in parts:
            if part.upper() in ['AND', 'OR']:
                logical_op = SearchOperator(part.upper())
                continue
            
            # Parse individual filter: field:operator:value
            match = re.match(r'(\w+):(\w+):(.+)', part.strip())
            if match:
                field_name, operator_name, value_str = match.groups()
                
                try:
                    # Map field name
                    field = SearchField(field_name.lower())
                    
                    # Map operator
                    operator = SearchOperator(operator_name.upper())
                    
                    # Parse value
                    if operator == SearchOperator.IN:
                        value = [v.strip() for v in value_str.split(',')]
                    elif operator == SearchOperator.RANGE:
                        parts = value_str.split(',')
                        if len(parts) == 2:
                            # Try to parse as numbers first, then dates
                            try:
                                value = (float(parts[0]), float(parts[1]))
                            except ValueError:
                                try:
                                    value = (
                                        datetime.datetime.fromisoformat(parts[0].strip()),
                                        datetime.datetime.fromisoformat(parts[1].strip())
                                    )
                                except:
                                    value = (parts[0].strip(), parts[1].strip())
                        else:
                            continue
                    else:
                        # Remove quotes if present
                        value = value_str.strip('"\'')
                    
                    search_filter = SearchFilter(
                        field=field,
                        operator=operator,
                        value=value
                    )
                    filters.append(search_filter)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse filter part: {part} - {e}")
                    continue
        
        return SearchQuery(
            filters=filters,
            logical_operator=logical_op
        )
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum suggestions
            
        Returns:
            List[str]: Search suggestions
        """
        suggestions = []
        
        try:
            session = self.Session()
            from forensnap_ultimate import Image, Tag
            
            # Suggest based on existing values
            if len(partial_query) >= 2:
                # Suggest from detected text (most common words)
                text_results = session.query(Image.detected_text).filter(
                    Image.detected_text.isnot(None),
                    func.lower(Image.detected_text).contains(partial_query.lower())
                ).limit(50).all()
                
                # Extract words containing the query
                words = set()
                for result in text_results:
                    if result[0]:
                        text_words = re.findall(r'\b\w+\b', result[0].lower())
                        for word in text_words:
                            if partial_query.lower() in word and len(word) > 2:
                                words.add(word)
                
                # Use fuzzy matching to rank suggestions
                if words:
                    fuzzy_matches = process.extract(partial_query, list(words), limit=limit//2)
                    suggestions.extend([match[0] for match in fuzzy_matches])
                
                # Suggest categories
                categories = ['chat', 'transaction', 'threat', 'adult_content', 'social_media']
                for cat in categories:
                    if partial_query.lower() in cat:
                        suggestions.append(f"category:equals:{cat}")
                
                # Suggest platforms
                platforms = ['whatsapp', 'telegram', 'instagram', 'facebook', 'twitter']
                for platform in platforms:
                    if partial_query.lower() in platform:
                        suggestions.append(f"platform:equals:{platform}")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
        
        return list(set(suggestions))[:limit]

# Example usage
if __name__ == "__main__":
    # This would be used with the actual ForenSnap system
    pass
