import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';
import { Send, Sparkles, RefreshCw, MessageCircle, FlaskConical, MapPin, BookOpen } from 'lucide-react';

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

// Simple markdown-like renderer for bullet points, bold, emojis
const FormattedText = ({ text }: { text: string }) => {
  const lines = text.split('\n');
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-2" />;
        // Bold: **text**
        const boldParts = line.split(/\*\*(.*?)\*\*/g);
        const rendered = boldParts.map((part, pi) =>
          pi % 2 === 1 ? <strong key={pi}>{part}</strong> : <span key={pi}>{part}</span>
        );
        if (line.startsWith('• ') || line.startsWith('- ') || line.match(/^[\u2022•]/)) {
          return (
            <div key={i} className="flex gap-2">
              <span className="text-rose-600 flex-shrink-0 mt-0.5">•</span>
              <span>{rendered}</span>
            </div>
          );
        }
        if (line.match(/^[#]{1,3} /)) {
          return <p key={i} className="font-bold text-gray-900 mt-2">{rendered}</p>;
        }
        return <p key={i}>{rendered}</p>;
      })}
    </div>
  );
};

const SUGGESTED_QUESTIONS = [
  '💧 Best moisturizers for dry skin?',
  '🧴 How to treat acne at home?',
  '☀️ Best sunscreens for oily skin?',
  '🏃 Facial exercises for better circulation?',
  '🌿 Natural remedies for eczema?',
  '✨ Skincare routine for beginners?',
];

export default function SkinBot() {
  const detectedDisease = (window as any).__skinbotDisease || '';

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'bot',
      text: detectedDisease
        ? `Hi! 👋 I detected **${detectedDisease}**. I'm SkinBot — your AI skin health assistant. I can help you with:\n\n• Treatment options and medications\n• Skincare product recommendations\n• Home care tips\n• When to see a dermatologist\n\nWhat would you like to know?`
        : `Hi! 👋 I'm **SkinBot**, your AI assistant for all things skin health!\n\nI can help you with:\n• 💊 Skin condition treatments & medications\n• 🧴 Real product recommendations (cleansers, moisturizers, SPF, serums)\n• 🌿 Natural remedies & home care tips\n• 🏃 Facial exercises for better skin health\n• 📋 Skincare routines for your skin type\n\nWhat's on your mind?`,
      timestamp: new Date(),
    },
  ]);

  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const buildHistory = (msgs: Message[]) =>
    msgs
      .filter((m) => m.sender === 'user' || m.sender === 'bot')
      .map((m) => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text }));

  const handleSendMessage = async (text: string = inputText) => {
    const trimmed = text.trim();
    if (!trimmed || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: trimmed,
      timestamp: new Date(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputText('');
    setIsTyping(true);
    setError('');

    try {
      const history = buildHistory(messages); // use messages before adding new one as history
      const response = await fetch('http://127.0.0.1:5000/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          disease: detectedDisease || '',
          history,
        }),
      });

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: data.reply || data.error || 'Sorry, I could not get a response.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setError('Could not reach SkinBot server. Make sure the backend is running.');
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: '⚠️ I am having trouble connecting to the server right now. Please make sure the backend is running and try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        id: Date.now().toString(),
        sender: 'bot',
        text: `Chat reset! 🔄 Feel free to ask me anything about skin health, products, or treatments.`,
        timestamp: new Date(),
      },
    ]);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-amber-50/30 to-white flex flex-col">
      <Header />

      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 pt-32 pb-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-indigo-500 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              SkinBot AI
            </h1>
            <p className="text-gray-500 text-sm mt-1">Powered by Groq · llama-3.3-70b · Real skincare expertise</p>
          </div>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 text-sm border border-gray-200 rounded-lg hover:bg-gray-50 text-gray-600 transition"
          >
            <RefreshCw className="w-4 h-4" /> Reset Chat
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* Chat area */}
        <div className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4" style={{ maxHeight: '55vh' }}>
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} gap-3`}>
                {msg.sender === 'bot' && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-500 to-indigo-500 flex items-center justify-center flex-shrink-0 mt-1">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[75%] px-5 py-3 rounded-2xl ${
                    msg.sender === 'user'
                      ? 'bg-gradient-to-br from-rose-600 to-indigo-600 text-white rounded-br-none'
                      : 'bg-gray-50 border border-gray-200 text-gray-900 rounded-bl-none'
                  }`}
                >
                  {msg.sender === 'bot' ? (
                    <FormattedText text={msg.text} />
                  ) : (
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                  )}
                  <p className={`text-xs mt-2 ${msg.sender === 'user' ? 'text-white/60' : 'text-gray-400'}`}>
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
                {msg.sender === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-1 text-xs font-bold text-gray-600">
                    You
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {isTyping && (
              <div className="flex justify-start gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-500 to-indigo-500 flex items-center justify-center flex-shrink-0">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
                <div className="bg-gray-50 border border-gray-200 px-5 py-3 rounded-2xl rounded-bl-none">
                  <div className="flex gap-1 items-center">
                    <div className="w-2 h-2 bg-rose-500 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-rose-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    <div className="w-2 h-2 bg-rose-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                    <span className="ml-2 text-xs text-gray-400">SkinBot is thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Suggested questions */}
          {messages.length <= 1 && (
            <div className="px-6 pb-4">
              <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide mb-3">Try asking:</p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_QUESTIONS.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSendMessage(q)}
                    className="px-3 py-1.5 text-xs bg-rose-50 border border-rose-200 text-rose-700 rounded-full hover:bg-rose-100 transition font-medium"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input area */}
          <div className="border-t border-gray-200 p-4">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex gap-3 items-end"
            >
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Ask about skincare, products, treatments..."
                  disabled={isTyping}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-rose-500 transition text-sm bg-white disabled:opacity-60"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                />
              </div>
              <button
                type="submit"
                disabled={!inputText.trim() || isTyping}
                className="w-11 h-11 flex items-center justify-center bg-gradient-to-br from-rose-600 to-indigo-600 text-white rounded-xl hover:from-rose-700 hover:to-indigo-700 transition shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-xs text-gray-400 mt-2 text-center">
              Educational guidance only · Always consult a dermatologist for medical concerns
            </p>
          </div>
        </div>

        {/* Quick action links */}
        <div className="mt-4 grid grid-cols-3 gap-3">
          <Link
            to="/analyze"
            className="flex flex-col items-center gap-1 p-3 bg-white border border-gray-200 rounded-xl hover:border-rose-500 hover:bg-rose-50 transition text-center group"
          >
            <FlaskConical className="w-5 h-5 text-rose-600" />
            <span className="text-xs font-semibold text-gray-700 group-hover:text-rose-700">Analyze Skin</span>
          </Link>
          <Link
            to="/hospitals"
            className="flex flex-col items-center gap-1 p-3 bg-white border border-gray-200 rounded-xl hover:border-rose-500 hover:bg-rose-50 transition text-center group"
          >
            <MapPin className="w-5 h-5 text-rose-600" />
            <span className="text-xs font-semibold text-gray-700 group-hover:text-rose-700">Find a Clinic</span>
          </Link>
          <Link
            to="/resources"
            className="flex flex-col items-center gap-1 p-3 bg-white border border-gray-200 rounded-xl hover:border-rose-500 hover:bg-rose-50 transition text-center group"
          >
            <BookOpen className="w-5 h-5 text-rose-600" />
            <span className="text-xs font-semibold text-gray-700 group-hover:text-rose-700">Learn More</span>
          </Link>
        </div>
      </main>
    </div>
  );
}
