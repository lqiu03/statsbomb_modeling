import React from 'react';

interface SuggestionCardProps {
  icon: string;
  title: string;
  description: string;
  onClick: () => void;
}

export const SuggestionCard: React.FC<SuggestionCardProps> = ({
  icon,
  title,
  description,
  onClick
}) => {
  return (
    <button
      onClick={onClick}
      className="p-4 bg-white border border-slate-200 rounded-xl hover:border-slate-300 hover:shadow-sm transition-all text-left group"
    >
      <div className="flex items-start gap-3">
        <div className="text-2xl">{icon}</div>
        <div className="flex-1">
          <h3 className="font-medium text-slate-900 group-hover:text-blue-600 transition-colors">
            {title}
          </h3>
          <p className="text-sm text-slate-600 mt-1">
            {description}
          </p>
        </div>
      </div>
    </button>
  );
};
