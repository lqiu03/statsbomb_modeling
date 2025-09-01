"""
Gemini API integration with soccer-specific prompt engineering.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import google.generativeai as genai
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


class SoccerRAGGenerator:
    """Handles response generation using Gemini API with soccer context."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API client.
        
        Args:
            api_key: Gemini API key (if None, will use GEMINI_API_KEY env var)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.demo_mode = not api_key
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Initialized Gemini API client")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                raise
        else:
            self.model = None
            logger.warning("Running in demo mode - no Gemini API key provided")
    
    def generate_response(self, 
                         query: str, 
                         context_docs: List[Dict], 
                         conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate natural response using retrieved soccer context.
        
        Args:
            query: User's question
            context_docs: Retrieved documents from vector database
            conversation_history: Previous conversation exchanges
            
        Returns:
            Generated response string
        """
        try:
            if self.demo_mode:
                return self._generate_demo_response(query, context_docs)
            
            soccer_prompt = self._build_soccer_prompt(query, context_docs, conversation_history)
            
            response = self.model.generate_content(soccer_prompt)
            
            if not response or not response.text:
                return "I apologize, but I couldn't generate a response to your question. Could you try rephrasing it?"
            
            formatted_response = self._format_response(response.text)
            logger.info(f"Generated response for query: {query[:50]}...")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def _build_soccer_prompt(self, query: str, context: List[Dict], history: Optional[List[Dict]]) -> str:
        """
        Build soccer-specific prompt with context and conversation memory.
        
        Args:
            query: User's question
            context: Retrieved context documents
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        context_text = self._format_context(context)
        history_text = self._format_history(history) if history else ""
        
        prompt = f"""You are a knowledgeable soccer analyst with access to comprehensive StatsBomb data spanning multiple competitions from 1958 to 2025. You provide warm, human-like responses about soccer players, teams, matches, and statistics.

{history_text}

Context from StatsBomb data:
{context_text}

User question: {query}

Guidelines for your response:
- Use natural, conversational language that feels warm and human
- Reference specific matches, players, teams, and statistics when relevant from the provided context
- Avoid robotic symbols like #, $, *, or bullet points
- Provide insights beyond just raw statistics - explain what the numbers mean
- If the context doesn't contain enough information to fully answer the question, acknowledge this limitation honestly
- Keep responses focused and concise while being informative
- When discussing player performance, consider context like opposition strength, match importance, and playing conditions
- If asked about comparisons, provide balanced analysis highlighting strengths and areas for improvement

Remember: Base your response primarily on the provided StatsBomb data context. If you need to acknowledge limitations in the data or your knowledge, do so naturally in conversation."""

        return prompt
    
    def _format_context(self, context_docs: List[Dict]) -> str:
        """Format context documents for the prompt."""
        if not context_docs:
            return "No specific data found for this query."
        
        formatted_context = []
        
        for i, doc in enumerate(context_docs[:8], 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            context_piece = f"{i}. {content}"
            
            if metadata.get('competition'):
                context_piece += f" (Competition: {metadata['competition']}"
                if metadata.get('season'):
                    context_piece += f", Season: {metadata['season']}"
                context_piece += ")"
            
            formatted_context.append(context_piece)
        
        return "\n".join(formatted_context)
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for context."""
        if not history:
            return ""
        
        history_text = "Previous conversation context:\n"
        for exchange in history[-3:]:
            if exchange.get('user_query'):
                history_text += f"User previously asked: {exchange['user_query']}\n"
            if exchange.get('assistant_response'):
                history_text += f"You responded: {exchange['assistant_response'][:200]}...\n"
        
        return history_text + "\n"
    
    def _format_response(self, response_text: str) -> str:
        """Clean and format the generated response."""
        response_text = response_text.strip()
        
        response_text = re.sub(r'[#$*]+', '', response_text)
        
        response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)
        
        response_text = re.sub(r'^\s*[-•]\s*', '', response_text, flags=re.MULTILINE)
        
        return response_text
    
    def _generate_demo_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate demo response when Gemini API is not available."""
        if not context_docs:
            return ("I'd love to help you with that question about soccer! However, I'm currently running in demo mode "
                   "and don't have access to the full dataset. To get detailed answers, please set up the Gemini API key.")
        
        # Create a response based on retrieved context
        context_info = []
        for doc in context_docs[:3]:  # Use top 3 results
            if 'metadata' in doc and doc['metadata']:
                meta = doc['metadata']
                if 'home_team' in meta and 'away_team' in meta:
                    context_info.append(f"{meta['home_team']} vs {meta['away_team']} ({meta.get('match_date', 'Unknown date')})")
                elif 'player_name' in meta:
                    context_info.append(f"Player: {meta['player_name']} - {doc.get('content', '')[:100]}...")
            elif 'content' in doc:
                context_info.append(doc['content'][:100] + "...")
        
        if context_info:
            return (f"Based on the StatsBomb data I found, here are some relevant matches and information:\n\n"
                   f"• {chr(10).join(['• ' + info for info in context_info])}\n\n"
                   f"This is a demo response showing that the RAG system successfully retrieved relevant data for your query: '{query}'. "
                   f"For detailed analysis and insights, please configure the Gemini API key.")
        else:
            return ("I found some relevant data in the StatsBomb database for your query, but I'm running in demo mode. "
                   "Please set up the Gemini API key to get detailed soccer analysis and insights.")


class ConversationMemory:
    """Manages conversation history and context for natural dialogue flow."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of exchanges to remember
        """
        self.conversation_history: List[Dict] = []
        self.max_history = max_history
        self.user_preferences: Dict = {}
        self.session_start = datetime.now()
    
    def add_exchange(self, user_query: str, assistant_response: str, context_used: List[Dict]):
        """
        Add user-assistant exchange to memory.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's response
            context_used: Context documents that were used
        """
        exchange = {
            "timestamp": datetime.now(),
            "user_query": user_query,
            "assistant_response": assistant_response,
            "context_used": context_used,
            "preferences_extracted": self.extract_user_preferences(user_query)
        }
        
        self.conversation_history.append(exchange)
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        self._update_user_preferences(exchange["preferences_extracted"])
        
        logger.info(f"Added exchange to memory. Total exchanges: {len(self.conversation_history)}")
    
    def get_relevant_history(self, current_query: str) -> List[Dict]:
        """
        Retrieve relevant conversation history for current query.
        
        Args:
            current_query: Current user query
            
        Returns:
            List of relevant previous exchanges
        """
        if not self.conversation_history:
            return []
        
        current_query_lower = current_query.lower()
        relevant_exchanges = []
        
        for exchange in self.conversation_history[-5:]:
            prev_query = exchange.get("user_query", "").lower()
            
            if self._queries_related(current_query_lower, prev_query):
                relevant_exchanges.append(exchange)
        
        return relevant_exchanges
    
    def _queries_related(self, query1: str, query2: str) -> bool:
        """Check if two queries are related."""
        common_soccer_terms = [
            'player', 'team', 'match', 'goal', 'assist', 'pass', 'shot', 
            'defense', 'attack', 'midfielder', 'striker', 'goalkeeper',
            'season', 'competition', 'league', 'tournament'
        ]
        
        query1_terms = set(query1.split())
        query2_terms = set(query2.split())
        
        common_terms = query1_terms.intersection(query2_terms)
        soccer_overlap = any(term in common_terms for term in common_soccer_terms)
        
        return len(common_terms) >= 2 or soccer_overlap
    
    def extract_user_preferences(self, query: str) -> Dict:
        """
        Extract user preferences from query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary of extracted preferences
        """
        preferences = {}
        query_lower = query.lower()
        
        team_patterns = [
            r'(?:barcelona|barca|fc barcelona)',
            r'(?:real madrid|madrid)',
            r'(?:manchester united|man united|united)',
            r'(?:manchester city|man city|city)',
            r'(?:liverpool|lfc)',
            r'(?:chelsea|cfc)',
            r'(?:arsenal|afc)',
            r'(?:tottenham|spurs)',
            r'(?:bayern munich|bayern)',
            r'(?:psg|paris saint-germain)'
        ]
        
        for pattern in team_patterns:
            if re.search(pattern, query_lower):
                team_match = re.search(pattern, query_lower)
                if team_match:
                    preferences['favorite_team'] = team_match.group()
                    break
        
        competition_patterns = [
            r'(?:premier league|epl)',
            r'(?:la liga|spanish league)',
            r'(?:champions league|ucl)',
            r'(?:world cup|fifa world cup)',
            r'(?:euros|european championship)',
            r'(?:bundesliga|german league)',
            r'(?:serie a|italian league)'
        ]
        
        for pattern in competition_patterns:
            if re.search(pattern, query_lower):
                comp_match = re.search(pattern, query_lower)
                if comp_match:
                    preferences['favorite_competition'] = comp_match.group()
                    break
        
        if any(word in query_lower for word in ['striker', 'forward', 'attacker']):
            preferences['position_interest'] = 'forward'
        elif any(word in query_lower for word in ['midfielder', 'midfield']):
            preferences['position_interest'] = 'midfielder'
        elif any(word in query_lower for word in ['defender', 'defense', 'centre-back']):
            preferences['position_interest'] = 'defender'
        elif any(word in query_lower for word in ['goalkeeper', 'keeper', 'gk']):
            preferences['position_interest'] = 'goalkeeper'
        
        return preferences
    
    def _update_user_preferences(self, new_preferences: Dict):
        """Update stored user preferences."""
        for key, value in new_preferences.items():
            if key not in self.user_preferences:
                self.user_preferences[key] = value
                logger.info(f"Learned user preference: {key} = {value}")
    
    def get_user_context(self) -> str:
        """Get user context string for personalization."""
        if not self.user_preferences:
            return ""
        
        context_parts = []
        
        if self.user_preferences.get('favorite_team'):
            context_parts.append(f"User shows interest in {self.user_preferences['favorite_team']}")
        
        if self.user_preferences.get('favorite_competition'):
            context_parts.append(f"User frequently asks about {self.user_preferences['favorite_competition']}")
        
        if self.user_preferences.get('position_interest'):
            context_parts.append(f"User is interested in {self.user_preferences['position_interest']} positions")
        
        return ". ".join(context_parts) + "." if context_parts else ""
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        self.user_preferences.clear()
        logger.info("Cleared conversation history and preferences")
