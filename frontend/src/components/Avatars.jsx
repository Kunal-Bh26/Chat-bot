import React from 'react';
import { motion } from 'framer-motion';

export const UserAvatar = () => (
  <motion.div
    className="w-10 h-10 rounded-full bg-gray-700/80 border-2 border-gray-400/50 flex items-center justify-center text-xl shadow-lg flex-shrink-0"
    initial={{ scale: 0, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    transition={{ duration: 0.4, ease: 'backOut' }}
  >
    🧑‍💻
  </motion.div>
);

export const BotAvatar = () => (
  <motion.div
    className="w-10 h-10 rounded-full bg-red-800/80 border-2 border-red-400/50 flex items-center justify-center text-2xl shadow-lg flex-shrink-0"
    initial={{ scale: 0, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    transition={{ duration: 0.4, ease: 'backOut' }}
  >
    🤖
  </motion.div>
);


