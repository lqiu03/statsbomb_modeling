import React from 'react';
import { MessageBubble } from './MessageBubble';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  sources?: Array<{title: string; url: string}>;
  timestamp: Date;
}

interface MessageListProps {
  messages: Message[];
}

export const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  return (
    <div className="space-y-6 pb-8">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div id="messages-end" />
    </div>
  );
};
