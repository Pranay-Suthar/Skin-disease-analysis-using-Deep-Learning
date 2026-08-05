import React, { useState, useRef, useEffect } from 'react';
import Header from '../components/Header';
import { Upload, RotateCcw, Brain, Zap, Shield, MapPin, BookOpen, Send, ArrowRight, Sparkles, MessageCircle } from 'lucide-react';

interface AnalysisResult {
  likelyPattern: string;
  confidence: number;
  severity: string;
  description: string;
  treatments: string;
  warning: string;
}

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
}

const MOCK_RESULT: AnalysisResult = {
  likelyPattern: 'Atopic Dermatitis',
  confidence: 88,
  severity: 'Moderate',
  description: 'Characterized by dry, inflamed, and itchy skin with typical eczema patterns.',
  treatments: 'Moisturizers, topical corticosteroids, and anti-inflammatory treatments are recommended.',
  warning: 'Seek medical attention if you notice signs of infection, severe pain, or rapid spreading.',
};

export default function Analyze() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        analyzeImage();
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisResult(null);
    setMessages([]);
    setInputText('');

    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setAnalysisProgress(100);
        setIsAnalyzing(false);
        setAnalysisResult(MOCK_RESULT);
        // Initialize chat
        setMessages([
          {
            id: '1',
            sender: 'bot',
            text: `I detected ${MOCK_RESULT.likelyPattern}. How can I help you understand this condition better?`,
          },
        ]);
      } else {
        setAnalysisProgress(Math.floor(progress));
      }
    }, 300);
  };

  const handleSendMessage = () => {
    if (!inputText.trim() || !analysisResult) return;

    const currentInput = inputText.trim();

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: currentInput,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    setTimeout(() => {
      const userQuery = currentInput.toLowerCase();
      let botResponse = '';

      // Check if the question is skin-related
      const skinKeywords = [
        'skin', 'dermatitis', 'eczema', 'acne', 'rash', 'itchy', 'dry', 'oily', 
        'moisturizer', 'treatment', 'remedy', 'cream', 'lotion', 'cleanser',
        'sunscreen', 'sensitive', 'inflammation', 'irritation', 'breakout',
        'psoriasis', 'fungal', 'infection', 'blister', 'burn', 'scars',
        'wrinkles', 'aging', 'spots', 'moles', 'warts', 'home remedy',
        'natural', 'organic', 'cure', 'prevent', 'care', 'health', 'condition'
      ];

      const isSkinRelated = skinKeywords.some(keyword => userQuery.includes(keyword));

      if (!isSkinRelated) {
        botResponse = "I'm specifically designed to help with skin-related questions. Please ask me about skin conditions, treatments, remedies, or related health topics!";
      } else if (userQuery.includes('remedy') || userQuery.includes('home')) {
        if (analysisResult.likelyPattern.includes('Dermatitis') || analysisResult.likelyPattern.includes('Eczema')) {
          botResponse = `For Atopic Dermatitis, some helpful home remedies include:\n\n• Keep skin moisturized - use fragrance-free creams or ointments daily\n• Avoid hot water - use lukewarm water for bathing\n• Use mild, unscented soap\n• Apply moisturizer within 3 minutes of bathing\n• Wear soft, breathable fabrics like cotton\n• Oatmeal baths - can help soothe itching\n• Avoid triggers like harsh chemicals and perfumes\n• Stay hydrated by drinking plenty of water\n\n⚠️ Always consult a dermatologist for persistent symptoms.`;
        } else {
          botResponse = `For ${analysisResult.likelyPattern}, some general remedies include:\n\n• Keep the area clean and dry\n• Use non-irritating cleansers\n• Apply recommended moisturizers\n• Avoid harsh products and triggers\n• Maintain good hygiene\n• Stay hydrated\n• Get adequate sleep for skin healing\n• Manage stress, which can affect skin health\n\n⚠️ Professional treatment is recommended for your condition.`;
        }
      } else if (userQuery.includes('cause') || userQuery.includes('why')) {
        botResponse = `${analysisResult.likelyPattern} can be caused by several factors:\n\n${analysisResult.likelyPattern === 'Atopic Dermatitis' 
          ? `• Genetic factors (family history)\n• Environmental triggers\n• Immune system dysfunction\n• Skin barrier weakness\n• Allergens and irritants\n• Stress and emotional factors\n• Weather changes\n• Bacterial infections\n\nThese factors can trigger or worsen symptoms.` 
          : `The causes can vary depending on triggers and individual factors. This depends on the specific condition and requires professional diagnosis.`}\n\n💡 For detailed information, consult your dermatologist.`;
      } else if (userQuery.includes('prevent') || userQuery.includes('avoid')) {
        botResponse = `To prevent or reduce ${analysisResult.likelyPattern}:\n\n✓ Maintain a consistent skincare routine\n✓ Use hypoallergenic products\n✓ Keep skin moisturized\n✓ Avoid known triggers\n✓ Manage stress levels\n✓ Maintain proper hygiene\n✓ Protect skin from harsh weather\n✓ Avoid prolonged sun exposure\n✓ Don't scratch or irritate affected areas\n✓ Use protective clothing when needed\n\n📋 Work with your dermatologist to identify your specific triggers.`;
      } else if (userQuery.includes('treatment') || userQuery.includes('medicine') || userQuery.includes('medication')) {
        botResponse = `${analysisResult.likelyPattern} treatment options include:\n\n💊 Medical Treatments:\n• Topical creams and ointments\n• Antihistamines for itching\n• Anti-inflammatory medications\n• Prescription corticosteroids\n• Immunosuppressants for severe cases\n\n🏥 Professional Treatment:\n• Phototherapy\n• Dermatologist prescription treatments\n• Specialized medical procedures\n\n⚠️ Please consult a dermatologist for proper diagnosis and personalized treatment plan.`;
      } else if (userQuery.includes('duration') || userQuery.includes('long')) {
        botResponse = `The duration and recovery time for ${analysisResult.likelyPattern} varies:\n\n• Depends on severity of your condition\n• Individual healing rates vary\n• Treatment compliance affects outcomes\n• Environmental factors play a role\n• Some conditions are chronic and need ongoing management\n\n🔄 Recovery can take weeks to months with proper treatment.\n\n📞 Ask your dermatologist about your specific timeline.`;
      } else if (userQuery.includes('dangerous') || userQuery.includes('serious') || userQuery.includes('risk')) {
        botResponse = `Regarding the seriousness of ${analysisResult.likelyPattern}:\n\n• Most skin conditions are manageable with proper care\n• Early treatment prevents complications\n• Infections can develop if not treated\n• Scarring is possible in severe cases\n• Quality of life impact should be considered\n\n⚠️ Seek medical attention if symptoms worsen or spread rapidly.\n\n🏥 Professional evaluation is important for your health.`;
      } else {
        botResponse = `I can help with questions about ${analysisResult.likelyPattern}. You can ask me about:\n\n• Home remedies and care tips\n• Causes and triggers\n• Prevention methods\n• Available treatments\n• Duration and recovery\n• When to see a doctor\n\n💬 Feel free to ask any skin-related questions!\n\n⚠️ This is educational information only - consult a dermatologist for medical advice.`;
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: botResponse,
      };
      setMessages((prev) => [...prev, botMessage]);
      setIsTyping(false);
    }, 1000);
  };

  const resetAnalysis = () => {
    setUploadedImage(null);
    setIsAnalyzing(false);
    setAnalysisProgress(0);
    setAnalysisResult(null);
    setMessages([]);
    setInputText('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-blue-50 to-white pt-28">
      <Header showAnalyzeButton={false} />

      <main className="mx-auto max-w-6xl px-5 py-12">
        <div className="mb-12">
          <h1 className="page-title mb-3 flex items-center gap-3">
            <Brain className="w-10 h-10 text-teal-600" />
            Analyze Your Skin
          </h1>
          <p className="page-subtitle">Upload an image to get instant AI-powered analysis</p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload & Results */}
          <div className="space-y-6">
            {!uploadedImage ? (
              <div className="space-y-6">
                {/* Main Upload Box */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-teal-300 rounded-2xl p-16 text-center cursor-pointer hover:border-teal-600 hover:bg-teal-50/50 transition-all duration-300 card group"
                >
                  <div className="mb-4 inline-block p-4 bg-gradient-to-br from-teal-100 to-emerald-100 rounded-full group-hover:scale-110 transition-transform duration-300">
                    <Upload className="w-8 h-8 text-teal-600" />
                  </div>
                  <p className="font-bold text-xl mb-2 text-gray-900">Click or drag to upload</p>
                  <p className="text-gray-600 text-sm mb-6">PNG, JPG, or GIF (max 10MB)</p>
                  <div className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-br from-teal-600 to-emerald-600 text-white rounded-lg font-semibold hover:from-teal-700 hover:to-emerald-700 transition-all duration-200 transform hover:scale-105 active:scale-95">
                    <Upload className="w-4 h-4" />
                    <span>Choose Image</span>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>

                {/* Info Cards */}
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="card-gradient">
                    <div className="flex items-center gap-3 mb-2">
                      <Sparkles className="w-5 h-5 text-teal-600" />
                      <h3 className="font-bold text-gray-900">95%+ Accuracy</h3>
                    </div>
                    <p className="text-sm text-gray-600">Advanced AI model trained on dermatological images</p>
                  </div>
                  <div className="card-gradient">
                    <div className="flex items-center gap-3 mb-2">
                      <Shield className="w-5 h-5 text-teal-600" />
                      <h3 className="font-bold text-gray-900">100% Secure</h3>
                    </div>
                    <p className="text-sm text-gray-600">Your images are encrypted and never stored</p>
                  </div>
                  <div className="card-gradient">
                    <div className="flex items-center gap-3 mb-2">
                      <Zap className="w-5 h-5 text-teal-600" />
                      <h3 className="font-bold text-gray-900">Instant Results</h3>
                    </div>
                    <p className="text-sm text-gray-600">Get analysis in seconds, not hours</p>
                  </div>
                  <div className="card-gradient">
                    <div className="flex items-center gap-3 mb-2">
                      <Brain className="w-5 h-5 text-teal-600" />
                      <h3 className="font-bold text-gray-900">Expert AI</h3>
                    </div>
                    <p className="text-sm text-gray-600">Powered by dermatologists' expertise</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6 animate-fade-in">
                <div className="relative rounded-2xl overflow-hidden bg-gray-100 shadow-lg ring-1 ring-white/20">
                  <img
                    src={uploadedImage}
                    alt="Uploaded"
                    className="w-full max-h-96 object-cover"
                  />
                </div>

                {isAnalyzing && (
                  <div className="card-gradient">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-bold text-gray-900">Analyzing your image...</span>
                      <span className="text-sm font-semibold text-teal-600">{analysisProgress}%</span>
                    </div>
                    <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                      <div
                        className="h-full bg-gradient-to-r from-teal-600 via-emerald-500 to-teal-600 transition-all duration-300 shadow-lg"
                        style={{ width: `${analysisProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {analysisResult && (
                  <div className="space-y-4 animate-slide-up">
                    <div className="card-gradient border-2 border-teal-500">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h2 className="text-3xl font-bold text-gray-900 mb-2">{analysisResult.likelyPattern}</h2>
                          <p className="text-gray-700 leading-relaxed">{analysisResult.description}</p>
                        </div>
                        <Brain className="w-8 h-8 text-teal-600 flex-shrink-0" />
                      </div>
                      <div className="flex gap-3 mb-6 flex-wrap">
                        <span className="px-4 py-2 bg-teal-600 text-white rounded-full text-sm font-bold shadow-md flex items-center gap-2">
                          <Zap className="w-4 h-4" />
                          {analysisResult.confidence}% confidence
                        </span>
                        <span className={`px-4 py-2 rounded-full text-sm font-bold shadow-md flex items-center gap-2 ${
                          analysisResult.severity === 'Critical' ? 'bg-red-100 text-red-700' :
                          analysisResult.severity === 'High' ? 'bg-amber-100 text-amber-700' :
                          analysisResult.severity === 'Moderate' ? 'bg-blue-100 text-blue-700' :
                          'bg-green-100 text-green-700'
                        }`}>
                          {analysisResult.severity} Severity
                        </span>
                      </div>

                      <div className="space-y-4 pt-4 border-t border-teal-200">
                        <div>
                          <p className="text-sm font-bold text-teal-700 mb-2 flex items-center gap-2">
                            <Shield className="w-4 h-4" /> Recommended Treatments
                          </p>
                          <p className="text-gray-700 text-sm leading-relaxed">{analysisResult.treatments}</p>
                        </div>
                        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
                          <p className="text-sm font-bold text-red-700 mb-1 flex items-center gap-2">
                            <Shield className="w-4 h-4" /> Important Notice
                          </p>
                          <p className="text-red-700 text-sm leading-relaxed">{analysisResult.warning}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  onClick={resetAnalysis}
                  disabled={isAnalyzing}
                  className="w-full px-4 py-3 border-2 border-teal-600 text-teal-600 font-bold rounded-lg hover:bg-teal-50 transition-all duration-200 disabled:opacity-50 hover:scale-105 active:scale-95 transform flex items-center justify-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Analyze Different Image
                </button>
              </div>
            )}
          </div>

          {/* Chat - Only visible after analysis */}
          {analysisResult && (
            <div className="flex flex-col h-96 lg:h-auto lg:min-h-96 rounded-2xl overflow-hidden shadow-xl ring-1 ring-white/20 bg-white animate-slide-left">
              <div className="bg-gradient-to-r from-teal-50 to-emerald-50 border-b border-teal-200 p-6">
                <h3 className="font-bold text-gray-900 flex items-center gap-2">
                  <MessageCircle className="w-5 h-5 text-teal-600" />
                  Ask About {analysisResult.likelyPattern}
                </h3>
                <p className="text-sm text-gray-600 mt-1">Get educational insights about this condition</p>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-slide-up`}
                  >
                    <div
                      className={`max-w-xs px-4 py-3 rounded-2xl text-sm shadow-md ${
                        msg.sender === 'user'
                          ? 'bg-gradient-to-br from-teal-600 to-emerald-600 text-white rounded-br-none'
                          : 'bg-gray-100 text-gray-900 rounded-bl-none'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 px-4 py-3 rounded-2xl rounded-bl-none">
                      <div className="flex gap-2">
                        <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="border-t border-gray-200 bg-gray-50 p-4">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendMessage();
                  }}
                  className="flex gap-3"
                >
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Ask a question..."
                    className="input-field flex-1 text-sm"
                  />
                  <button
                    type="submit"
                    disabled={!inputText.trim() || isTyping}
                    className="px-4 py-2 bg-gradient-to-r from-teal-600 to-emerald-600 text-white rounded-lg hover:from-teal-700 hover:to-emerald-700 transition-all duration-200 disabled:opacity-50 font-semibold transform hover:scale-105 active:scale-95 flex items-center gap-2"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>
            </div>
          )}
        </div>

        {/* Quick Links */}
        {analysisResult && (
          <div className="mt-12 grid gap-6 md:grid-cols-2 animate-fade-in">
            <a href="/hospitals" className="card group hover:scale-105 transition-all duration-300">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-gray-900 group-hover:text-teal-600 transition-colors mb-2">Find a Dermatologist</h3>
                  <p className="text-sm text-gray-600">Connect with licensed specialists in your area</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-teal-600 font-semibold text-sm">
                <MapPin className="w-4 h-4" />
                View Clinics
              </div>
            </a>
            <a href="/resources" className="card group hover:scale-105 transition-all duration-300">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-gray-900 group-hover:text-teal-600 transition-colors mb-2">Learn More</h3>
                  <p className="text-sm text-gray-600">Educational resources about skin care and health</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-teal-600 font-semibold text-sm">
                <BookOpen className="w-4 h-4" />
                Explore Resources
              </div>
            </a>
          </div>
        )}
      </main>
    </div>
  );
}
