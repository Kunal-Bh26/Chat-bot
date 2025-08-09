import React from 'react';
import { motion } from 'framer-motion';
import { BotAvatar } from './Avatars.jsx';

export const TypingIndicator = () => (
  <motion.div
    className="flex items-end space-x-3"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.3 }}
  >
    <BotAvatar />
    <div className="p-4 rounded-2xl bg-white/10 w-20 flex justify-center items-center space-x-1 shadow-md">
      <motion.div className="w-2 h-2 bg-red-400 rounded-full" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }} />
      <motion.div className="w-2 h-2 bg-red-400 rounded-full" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, delay: 0.1, repeat: Infinity, ease: 'easeInOut' }} />
      <motion.div className="w-2 h-2 bg-red-400 rounded-full" animate={{ y: [0, -4, 0] }} transition={{ duration: 0.8, delay: 0.2, repeat: Infinity, ease: 'easeInOut' }} />
    </div>
  </motion.div>
);


