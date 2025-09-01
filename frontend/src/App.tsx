import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, RotateCcw } from 'lucide-react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  sources?: Array<{
    content: string;
    metadata: Record<string, any>;
    relevance_score: number;
  }>;
  timestamp: Date;
}

interface ChatResponse {
  response: string;
  sources: Array<{
    content: string;
    metadata: Record<string, any>;
    relevance_score: number;
  }>;
  session_id: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() => uuidv4());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      content: input.trim(),
      role: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, {
        query: userMessage.content,
        session_id: sessionId
      });

      const assistantMessage: Message = {
        id: uuidv4(),
        content: response.data.response,
        role: 'assistant',
        sources: response.data.sources,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSessionId(response.data.session_id);

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: uuidv4(),
        content: "I'm sorry, I encountered an error while processing your question. Please try again.",
        role: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setSessionId(uuidv4());
  };

  const suggestedQueries = [
    "Who are the top ball-progressing defenders this season?",
    "Compare Messi's passing accuracy across different competitions",
    "Show me players with similar profiles to Kevin De Bruyne",
    "Find promising young strikers from the Premier League"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-semibold text-sm">⚽</span>
              </div>
              <h1 className="text-xl font-semibold text-slate-800">StatsBomb Soccer Assistant</h1>
            </div>
            
            {messages.length > 0 && (
              <button
                onClick={clearConversation}
                className="flex items-center space-x-2 px-3 py-2 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <RotateCcw size={16} />
                <span className="text-sm">New Chat</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="max-w-4xl mx-auto px-6 py-8 pb-32">
        {messages.length === 0 ? (
          <WelcomeScreen onSuggestedQuery={setInput} suggestedQueries={suggestedQueries} />
        ) : (
          <MessageList messages={messages} />
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed at bottom */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-sm border-t border-slate-200">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about soccer players, teams, matches, and statistics..."
                className="w-full px-4 py-3 pr-12 border border-slate-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm"
                rows={1}
                style={{ minHeight: '52px', maxHeight: '120px' }}
                disabled={isLoading}
              />
            </div>
            
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="flex items-center justify-center w-12 h-12 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              {isLoading ? (
                <Loader2 size={20} className="animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface WelcomeScreenProps {
  onSuggestedQuery: (query: string) => void;
  suggestedQueries: string[];
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSuggestedQuery, suggestedQueries }) => (
  <div className="text-center py-16">
    <div className="mb-8">
      <div className="w-16 h-16 bg-blue-600 rounded-2xl mx-auto mb-4 flex items-center justify-center">
        <span className="text-white text-2xl">⚽</span>
      </div>
      <h2 className="text-3xl font-semibold text-slate-800 mb-2">
        Hi there, how can I help you today?
      </h2>
      <p className="text-slate-600 text-lg max-w-2xl mx-auto">
        Ask me anything about soccer players, teams, matches, and statistics from comprehensive StatsBomb data spanning multiple competitions from 1958 to 2025
      </p>
    </div>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl mx-auto mb-8">
      <SuggestionCard 
        icon="🏆" 
        title="Player Analysis"
        description="Compare players across competitions and seasons"
      />
      <SuggestionCard 
        icon="📊" 
        title="Team Performance"
        description="Analyze team tactics and match statistics"
      />
      <SuggestionCard 
        icon="🎯" 
        title="Match Insights"
        description="Get detailed breakdowns of specific matches"
      />
      <SuggestionCard 
        icon="🔍" 
        title="Talent Scouting"
        description="Discover promising players and hidden gems"
      />
    </div>

    <div className="space-y-3">
      <p className="text-slate-500 text-sm font-medium">Try asking:</p>
      <div className="flex flex-wrap gap-2 justify-center">
        {suggestedQueries.map((query, index) => (
          <button
            key={index}
            onClick={() => onSuggestedQuery(query)}
            className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-colors text-sm"
          >
            {query}
          </button>
        ))}
      </div>
    </div>
  </div>
);

interface SuggestionCardProps {
  icon: string;
  title: string;
  description: string;
}

const SuggestionCard: React.FC<SuggestionCardProps> = ({ icon, title, description }) => (
  <div className="p-6 bg-white rounded-xl border border-slate-200 hover:border-slate-300 transition-colors">
    <div className="text-2xl mb-3">{icon}</div>
    <h3 className="font-semibold text-slate-800 mb-2">{title}</h3>
    <p className="text-slate-600 text-sm">{description}</p>
  </div>
);

interface MessageListProps {
  messages: Message[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => (
  <div className="space-y-6">
    {messages.map((message) => (
      <MessageBubble key={message.id} message={message} />
    ))}
  </div>
);

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => (
  <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
    <div className={`max-w-3xl ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
      {message.role === 'assistant' && (
        <div className="flex items-center space-x-2 mb-2">
          <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
            <span className="text-white text-xs">⚽</span>
          </div>
          <span className="text-sm font-medium text-slate-700">StatsBomb Assistant</span>
        </div>
      )}
      
      <div
        className={`px-4 py-3 rounded-2xl ${
          message.role === 'user'
            ? 'bg-blue-600 text-white'
            : 'bg-white border border-slate-200 text-slate-800'
        }`}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
        
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-3 border-t border-slate-100">
            <p className="text-xs font-medium text-slate-500 mb-2">Sources:</p>
            <div className="space-y-2">
              {message.sources.slice(0, 3).map((source, index) => (
                <div key={index} className="text-xs bg-slate-50 rounded-lg p-2">
                  <div className="text-slate-600 mb-1">{source.content}</div>
                  {source.metadata.competition && (
                    <div className="text-slate-500">
                      {source.metadata.competition}
                      {source.metadata.season && ` • ${source.metadata.season}`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="text-xs text-slate-500 mt-1 px-2">
        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  </div>
);
