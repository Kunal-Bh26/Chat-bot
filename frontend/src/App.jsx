import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PaperAirplaneIcon, ArrowPathIcon } from '@heroicons/react/24/solid';

import { BotAvatar } from './components/Avatars.jsx';
import { ChatMessage } from './components/ChatMessage.jsx';
import { TypingIndicator } from './components/TypingIndicator.jsx';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001/api/chat';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatStarted, setChatStarted] = useState(false);
  const chatEndRef = useRef(null);
  const smartReplies = ['Reset password', 'VPN issues', 'Software install'];

  useEffect(() => {
    try {
      const saved = localStorage.getItem('chatMessagesV3');
      if (saved) {
        setMessages(JSON.parse(saved));
        setChatStarted(true);
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      if (chatStarted) localStorage.setItem('chatMessagesV3', JSON.stringify(messages));
    } catch {}
  }, [messages, chatStarted]);

  useEffect(() => {
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }, [messages, isLoading]);

  const getBotResponse = async (text) => {
    setIsLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'bot', content: data.response }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: "I'm having trouble connecting. Please try again later." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const startChat = () => {
    setChatStarted(true);
    setIsLoading(true);
    setMessages([]);
    setTimeout(() => {
      setMessages([
        { role: 'bot', content: 'Greetings! I am your dedicated IT Helpdesk assistant. How may I help you today?' },
      ]);
      setIsLoading(false);
    }, 800);
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.removeItem('chatMessagesV3');
    startChat();
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const msg = { role: 'user', content: input };
    setMessages((prev) => [...prev, msg]);
    const toSend = input;
    setInput('');
    await getBotResponse(toSend);
  };

  const handleSmartReplyClick = async (reply) => {
    if (isLoading) return;
    const msg = { role: 'user', content: reply };
    setMessages((prev) => [...prev, msg]);
    await getBotResponse(reply);
  };

  return (
    <div className="font-sans bg-gray-900 text-white w-full min-h-screen flex flex-col items-center justify-center p-4 selection:bg-red-500/50">
      <div className="absolute inset-0 z-0 overflow-hidden">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-red-900 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-rose-800 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-red-700 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <main className="relative z-10 w-full max-w-2xl h-[95vh] md:h-[90vh] flex flex-col bg-black/30 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-2xl shadow-red-900/20 overflow-hidden">
        <header className="p-4 border-b border-white/10 flex justify-between items-center flex-shrink-0">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <BotAvatar />
              <span className="absolute bottom-0 right-0 block h-2.5 w-2.5 rounded-full bg-green-400 ring-2 ring-gray-900" />
            </div>
            <h1 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-rose-500">
              IT Helpdesk AI
            </h1>
          </div>
          {chatStarted && (
            <motion.button
              onClick={clearChat}
              whileHover={{ scale: 1.1, rotate: 90 }}
              whileTap={{ scale: 0.9 }}
              className="p-2 rounded-full text-gray-400 hover:text-white hover:bg-white/10"
              title="Start New Chat"
            >
              <ArrowPathIcon className="w-5 h-5" />
            </motion.button>
          )}
        </header>

        <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
          <div className="flex flex-col space-y-6">
            <AnimatePresence>
              {!chatStarted ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                  className="flex flex-col items-center justify-center h-full text-center"
                >
                  <motion.div
                    className="text-6xl mb-4 text-red-400"
                    animate={{ y: [0, -10, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                  >
                    <BotAvatar />
                  </motion.div>
                  <h2 className="text-3xl font-bold mb-2">Welcome to the Future of IT Support</h2>
                  <p className="text-slate-400 mb-6 max-w-sm">Click the button below to start a conversation with our intelligent assistant.</p>
                  <motion.button
                    onClick={startChat}
                    whileHover={{ scale: 1.05, boxShadow: '0px 0px 20px rgba(239, 68, 68, 0.4)' }}
                    whileTap={{ scale: 0.95 }}
                    className="px-8 py-3 font-bold text-white bg-gradient-to-r from-red-600 to-rose-700 rounded-full shadow-lg transition-all duration-300"
                  >
                    Start Chat
                  </motion.button>
                </motion.div>
              ) : (
                messages.map((m, i) => (
                  <ChatMessage key={i} message={m.content} isBot={m.role === 'bot'} />
                ))
              )}
            </AnimatePresence>
            {isLoading && <TypingIndicator />}
            <div ref={chatEndRef} />
          </div>
        </div>

        {chatStarted && !isLoading && messages.length > 0 && (
          <div className="px-6 pb-4 flex flex-wrap gap-2">
            {smartReplies.map((reply, i) => (
              <motion.button
                key={reply}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.15 }}
                onClick={() => handleSmartReplyClick(reply)}
                whileHover={{ scale: 1.05, backgroundColor: 'rgba(255, 255, 255, 0.2)' }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 text-sm bg-white/10 border border-white/20 rounded-full transition-colors"
              >
                {reply}
              </motion.button>
            ))}
          </div>
        )}

        {chatStarted && (
          <div className="p-4 border-t border-white/10 flex-shrink-0">
            <form onSubmit={handleSend} className="flex items-center space-x-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything about IT..."
                disabled={isLoading}
                className="flex-1 w-full px-5 py-3 bg-white/5 border-2 border-transparent rounded-full focus:outline-none focus:ring-2 focus:ring-red-500/50 focus:border-red-500 transition-all duration-300 disabled:opacity-50"
              />
              <motion.button
                type="submit"
                disabled={isLoading || !input.trim()}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="w-12 h-12 flex items-center justify-center bg-gradient-to-r from-red-600 to-rose-700 rounded-full text-white text-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PaperAirplaneIcon className="w-6 h-6" />
              </motion.button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}


