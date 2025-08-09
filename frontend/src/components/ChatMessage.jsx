import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BotAvatar, UserAvatar } from './Avatars.jsx';

export const ChatMessage = ({ message, isBot }) => {
  const [displayedContent, setDisplayedContent] = useState('');

  useEffect(() => {
    if (isBot && message) {
      let i = 0;
      setDisplayedContent('');
      const type = () => {
        if (i < message.length) {
          setDisplayedContent((prev) => prev + message.charAt(i));
          i += 1;
          setTimeout(type, 25);
        }
      };
      type();
    } else {
      setDisplayedContent(message || '');
    }
  }, [message, isBot]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={`flex items-end space-x-3 w-full ${!isBot ? 'justify-end' : ''}`}
    >
      {isBot && <BotAvatar />}
      <div
        className={`max-w-lg p-4 rounded-2xl shadow-md ${!isBot ? 'bg-white text-gray-900 rounded-br-none' : 'bg-gradient-to-br from-red-600 to-rose-800 text-white rounded-bl-none'}`}
      >
        <p className="text-sm md:text-base whitespace-pre-wrap min-h-[1.5rem]">{displayedContent}</p>
      </div>
      {!isBot && <UserAvatar />}
    </motion.div>
  );
};


