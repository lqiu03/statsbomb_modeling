import React from 'react';
import { SuggestionCard } from './SuggestionCard';

interface WelcomeScreenProps {
  onSuggestionClick: (suggestion: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSuggestionClick }) => {
  const suggestions = [
    {
      icon: "🏆",
      title: "Player Analysis",
      description: "Compare players across competitions and seasons",
      query: "Who are the top goal scorers in the FA Women's Super League this season?"
    },
    {
      icon: "📊",
      title: "Team Performance", 
      description: "Analyze team tactics and match statistics",
      query: "How did Arsenal perform against Chelsea in their recent matches?"
    },
    {
      icon: "🎯",
      title: "Match Insights",
      description: "Get detailed breakdowns of specific matches",
      query: "What were the key tactical moments in the latest Manchester City vs Liverpool match?"
    },
    {
      icon: "🔍",
      title: "Talent Scouting",
      description: "Discover promising players and hidden gems",
      query: "Find young defenders with high pass completion rates and strong aerial ability"
    }
  ];

  return (
    <div className="text-center py-16">
      <div className="mb-8">
        <div className="w-16 h-16 bg-blue-600 rounded-2xl mx-auto mb-4 flex items-center justify-center">
          <span className="text-white text-2xl">⚽</span>
        </div>
        <h2 className="text-3xl font-semibold text-slate-800 mb-2">
          Hi there, how can I help you today?
        </h2>
        <p className="text-slate-600 text-lg">
          Ask me anything about soccer players, teams, matches, and statistics from StatsBomb data
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
        {suggestions.map((suggestion, index) => (
          <SuggestionCard 
            key={index}
            icon={suggestion.icon}
            title={suggestion.title}
            description={suggestion.description}
            onClick={() => onSuggestionClick(suggestion.query)}
          />
        ))}
      </div>
    </div>
  );
};
