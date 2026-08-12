import React, { useState, useRef, useEffect, useCallback } from 'react';
import Header from '../components/Header';
import { Upload, RotateCcw, Brain, Zap, Shield, MapPin, BookOpen, Send, Sparkles, MessageCircle, Activity, Camera, AlertCircle, X, Aperture, Download, Loader2, Video, CheckCircle2 } from 'lucide-react';
import { fetchHomeRemedies, fetchYouTubeVideos, HomeRemedy, YouTubeVideo } from '../utils/resourcesApi';

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

const FormattedText = ({ text }: { text: string }) => {
  const lines = text.split('\n');
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-2" />;
        const boldParts = line.split(/\*\*(.*?)\*\*/g);
        const rendered = boldParts.map((part, pi) =>
          pi % 2 === 1 ? <strong key={pi} className="font-bold">{part}</strong> : <span key={pi}>{part}</span>
        );
        if (line.startsWith('• ') || line.startsWith('- ') || line.match(/^[\u2022•]/)) {
          return (
            <div key={i} className="flex gap-2 items-start mt-1">
              <span className="text-rose-600 flex-shrink-0 mt-1"><Sparkles className="w-3 h-3" /></span>
              <span>{rendered}</span>
            </div>
          );
        }
        if (line.match(/^[#]{1,3} /)) {
          return <p key={i} className="font-bold text-gray-900 mt-3 mb-1 text-base">{rendered}</p>;
        }
        return <p key={i}>{rendered}</p>;
      })}
    </div>
  );
};

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
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  
  // Report Generation State
  const [reportHospitals, setReportHospitals] = useState<any[]>([]);
  const [reportRemedies, setReportRemedies] = useState<HomeRemedy[]>([]);
  const [reportVideos, setReportVideos] = useState<YouTubeVideo[]>([]);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const pdfTemplateRef = useRef<HTMLDivElement>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
      analyzeImage(file);
    };
    reader.readAsDataURL(file);
  };

  const startCamera = async () => {
    try {
      setIsCameraOpen(true);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Unable to access camera. Please make sure you have granted camera permissions.");
      setIsCameraOpen(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraOpen(false);
  }, []);

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            stopCamera();
            processFile(file);
          }
        }, 'image/jpeg', 0.9);
      }
    }
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const generateReport = async () => {
    if (!analysisResult) return;
    setIsGeneratingPdf(true);

    try {
      // Fetch Hospitals
      let fetchedHospitals: any[] = [];
      try {
        const position = await new Promise<GeolocationPosition>((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
        }).catch(() => null);

        const payload = position 
          ? { lat: position.coords.latitude, lon: position.coords.longitude }
          : { location: "New York, USA" }; 

        const res = await fetch('http://127.0.0.1:5000/api/hospitals/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        fetchedHospitals = (data.hospitals || []).slice(0, 5);
      } catch (err) {
        console.error('Failed to fetch hospitals', err);
      }
      setReportHospitals(fetchedHospitals);

      // Fetch Remedies & Videos
      const remedies = await fetchHomeRemedies(analysisResult.likelyPattern);
      setReportRemedies(remedies || []);

      const videos = await fetchYouTubeVideos(analysisResult.likelyPattern);
      setReportVideos((videos || []).slice(0, 3));

      // Wait a moment for state to update the DOM
      await new Promise(resolve => setTimeout(resolve, 800));

      if (pdfTemplateRef.current) {
        const element = pdfTemplateRef.current;
        element.style.display = 'block'; 
        
        // Dynamically import html2pdf
        const html2pdfModule = await import('html2pdf.js');
        const html2pdf = html2pdfModule.default || html2pdfModule;
        
        const opt = {
          margin: 0.5,
          filename: `DermaAI_Report_${analysisResult.likelyPattern.replace(/\s+/g, '_')}.pdf`,
          image: { type: 'jpeg', quality: 0.98 },
          html2canvas: { scale: 2, useCORS: true },
          jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
        };

        // @ts-ignore
        await html2pdf().from(element).set(opt).save();
        element.style.display = 'none';
      }
    } catch (error) {
      console.error("PDF generation error:", error);
      alert("Failed to generate PDF report.");
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const analyzeImage = async (file: File) => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisResult(null);
    setMessages([]);
    setInputText('');

    // Fake progress bar animation while fetching
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 90) {
        progress = 90; // hold at 90% until fetch completes
      }
      setAnalysisProgress(Math.floor(progress));
    }, 200);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://127.0.0.1:5000/api/predict/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      clearInterval(progressInterval);
      setAnalysisProgress(100);
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze image');
      }

      const result: AnalysisResult = {
        likelyPattern: data.disease || 'Unknown Condition',
        confidence: Math.round((data.confidence || 0) * 100),
        severity: data.severity || 'Unknown',
        description: data.info?.description || 'No description available.',
        treatments: data.info?.treatments || 'Consult a dermatologist for treatment options.',
        warning: data.info?.home_care || 'Seek medical attention if you notice rapid spreading.',
      };

      setAnalysisResult(result);
      setIsAnalyzing(false);

      setMessages([
        {
          id: '1',
          sender: 'bot',
          text: `I detected **${result.likelyPattern}**. How can I help you understand this condition better?`,
        },
      ]);
      
    } catch (err: any) {
      clearInterval(progressInterval);
      setAnalysisProgress(100);
      setIsAnalyzing(false);
      
      setMessages([
        {
          id: '1',
          sender: 'bot',
          text: `⚠️ Analysis failed: ${err.message}. Please try again with a different image.`,
        },
      ]);
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || !analysisResult) return;

    const currentInput = inputText.trim();

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: currentInput,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputText('');
    setIsTyping(true);

    try {
      const history = messages
        .filter((m) => m.sender === 'user' || m.sender === 'bot')
        .map((m) => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text }));

      const response = await fetch('http://127.0.0.1:5000/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: currentInput,
          disease: analysisResult.likelyPattern || '',
          history: history,
        }),
      });

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: data.reply || data.error || 'Sorry, I could not get a response.',
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: '⚠️ I am having trouble connecting to the server right now. Please try again.',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const resetAnalysis = () => {
    setUploadedImage(null);
    setImageFile(null);
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
    <div className="min-h-screen bg-slate-50 overflow-x-hidden pt-28">
      <Header showAnalyzeButton={false} />

      {/* Decorative Background */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-rose-400/10 blur-[100px]" />
        <div className="absolute bottom-[20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-amber-400/10 blur-[120px]" />
      </div>

      <main className="mx-auto max-w-7xl px-5 py-8">
        
        {/* Header Area */}
        <div className="mb-10 animate-fade-in text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-rose-50 text-rose-700 font-bold text-sm mb-6 border border-rose-100 shadow-sm">
            <Sparkles className="w-4 h-4 text-rose-500" />
            AI-Powered Clinical Analysis
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">
            Analyze Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-600 to-amber-600">Skin Health</span>
          </h1>
          <p className="text-lg text-slate-600">
            Upload a clear image of your skin concern for an instant, privacy-first evaluation using our state-of-the-art diagnostic model.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-12 max-w-6xl mx-auto">
          
          {/* Left Column: Upload & Results (Spans 7 cols on large screens) */}
          <div className="lg:col-span-7 space-y-6">
            {!uploadedImage ? (
              <div className="space-y-6 animate-slide-up">
                
                {isCameraOpen ? (
                  /* Camera Viewfinder */
                  <div className="relative overflow-hidden rounded-[2rem] bg-slate-900 border-2 border-rose-200 shadow-2xl shadow-rose-900/10 group">
                    <div className="flex items-center justify-between px-6 py-4 bg-slate-900/50 absolute top-0 w-full z-10 backdrop-blur-md">
                      <div className="flex items-center gap-2 text-white font-bold">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                        Camera Active
                      </div>
                      <button onClick={stopCamera} className="p-2 bg-white/10 hover:bg-white/20 rounded-full text-white transition-colors">
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                    
                    <video 
                      ref={videoRef}
                      autoPlay 
                      playsInline
                      className="w-full h-[400px] object-cover scale-x-[-1]"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    
                    <div className="absolute bottom-0 w-full p-6 bg-gradient-to-t from-slate-900 to-transparent flex justify-center pb-8">
                      <button 
                        onClick={capturePhoto}
                        className="w-20 h-20 bg-rose-500 hover:bg-rose-400 border-4 border-white rounded-full shadow-[0_0_20px_rgba(244,63,94,0.5)] transition-all duration-200 transform hover:scale-105 active:scale-95 flex items-center justify-center"
                      >
                        <Aperture className="w-10 h-10 text-white" />
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Premium Upload Dropzone */
                  <div className="relative overflow-hidden border-2 border-dashed border-rose-200 rounded-[2rem] p-12 text-center hover:border-rose-400 hover:shadow-2xl hover:shadow-rose-900/5 transition-all duration-300 bg-white group shadow-xl shadow-slate-200/40">
                    {/* Hover gradient effect inside dropzone */}
                    <div className="absolute inset-0 bg-gradient-to-br from-rose-50 to-orange-50 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                    
                    <div className="relative z-10">
                      <div className="mb-6 inline-flex p-5 bg-gradient-to-br from-rose-100 to-orange-100 rounded-2xl group-hover:scale-110 transition-transform duration-500 shadow-inner">
                        <Camera className="w-10 h-10 text-rose-600" />
                      </div>
                      <h3 className="font-extrabold text-2xl mb-2 text-slate-900">Upload Skin Image</h3>
                      <p className="text-slate-500 text-sm mb-8 font-medium">PNG, JPG, or HEIC (max 10MB)</p>
                      
                      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <button 
                          onClick={() => fileInputRef.current?.click()}
                          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-4 bg-slate-900 text-white rounded-xl font-bold hover:bg-slate-800 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg shadow-slate-900/20"
                        >
                          <Upload className="w-5 h-5" />
                          Browse Files
                        </button>
                        
                        <button 
                          onClick={startCamera}
                          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-4 bg-rose-50 text-rose-700 border border-rose-200 rounded-xl font-bold hover:bg-rose-100 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg shadow-rose-900/5"
                        >
                          <Camera className="w-5 h-5" />
                          Take Photo
                        </button>
                      </div>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </div>
                )}

                {/* Feature Grid */}
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { icon: Sparkles, title: "95%+ Accuracy", desc: "Trained on millions of clinical images", color: "rose" },
                    { icon: Shield, title: "100% Private", desc: "Zero-retention encryption policy", color: "indigo" },
                    { icon: Zap, title: "Instant Analysis", desc: "Results delivered in under 5 seconds", color: "blue" },
                    { icon: Brain, title: "Expert Model", desc: "Verified by top dermatologists", color: "violet" }
                  ].map((feat, idx) => (
                    <div key={idx} className="bg-white p-5 rounded-2xl shadow-md shadow-slate-200/50 border border-slate-100 hover:border-rose-200 transition-colors">
                      <div className={`w-10 h-10 rounded-xl bg-${feat.color}-50 flex items-center justify-center mb-3 text-${feat.color}-600`}>
                        <feat.icon className="w-5 h-5" />
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm mb-1">{feat.title}</h4>
                      <p className="text-xs text-slate-500 font-medium leading-relaxed">{feat.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-6 animate-fade-in">
                {/* Image Preview Window */}
                <div className="relative rounded-[2rem] overflow-hidden bg-slate-900 shadow-2xl ring-4 ring-white/50 aspect-[4/3] group">
                  <img
                    src={uploadedImage}
                    alt="Uploaded for analysis"
                    className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-500"
                  />
                  {/* Scanner overlay effect when analyzing */}
                  {isAnalyzing && (
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="w-full h-2 bg-rose-400 blur-sm animate-scan opacity-70"></div>
                      <div className="absolute inset-0 bg-rose-900/20 animate-pulse"></div>
                    </div>
                  )}
                </div>

                {isAnalyzing && (
                  <div className="bg-white rounded-2xl p-6 shadow-xl shadow-slate-200/50 border border-slate-100">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <Activity className="w-5 h-5 text-rose-600 animate-spin-slow" />
                        <span className="font-bold text-slate-900 text-lg">Running Neural Network...</span>
                      </div>
                      <span className="text-lg font-black text-rose-600">{analysisProgress}%</span>
                    </div>
                    <div className="w-full h-4 bg-slate-100 rounded-full overflow-hidden shadow-inner border border-slate-200">
                      <div
                        className="h-full bg-gradient-to-r from-rose-500 via-indigo-400 to-rose-500 transition-all duration-300 relative"
                        style={{ width: `${analysisProgress}%` }}
                      >
                        <div className="absolute inset-0 bg-white/20 animate-shimmer"></div>
                      </div>
                    </div>
                  </div>
                )}

                {analysisResult && (
                  <div className="bg-white rounded-3xl p-8 shadow-2xl shadow-slate-200/50 border border-rose-100 relative overflow-hidden animate-slide-up">
                    {/* Decorative abstract shape in results */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-rose-50 rounded-full blur-3xl -mr-32 -mt-32 opacity-70 pointer-events-none"></div>
                    
                    <div className="relative z-10">
                      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-6">
                        <div>
                          <p className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-1">Detection Result</p>
                          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 leading-tight">
                            {analysisResult.likelyPattern}
                          </h2>
                        </div>
                        <div className="flex gap-2 flex-wrap">
                          <span className="px-4 py-2 bg-slate-900 text-white rounded-xl text-sm font-bold shadow-md flex items-center gap-2">
                            <Sparkles className="w-4 h-4 text-rose-400" />
                            {analysisResult.confidence}% match
                          </span>
                          <span className={`px-4 py-2 rounded-xl text-sm font-bold shadow-md flex items-center gap-2 border ${
                            analysisResult.severity === 'Critical' ? 'bg-red-50 text-red-700 border-red-200' :
                            analysisResult.severity === 'High' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                            analysisResult.severity === 'Moderate' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                            'bg-indigo-50 text-indigo-700 border-indigo-200'
                          }`}>
                            <AlertCircle className="w-4 h-4" />
                            {analysisResult.severity} Risk
                          </span>
                        </div>
                      </div>

                      <p className="text-slate-600 text-lg leading-relaxed mb-8 font-medium">
                        {analysisResult.description}
                      </p>

                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="bg-amber-50/50 p-5 rounded-2xl border border-amber-100">
                          <p className="text-sm font-bold text-amber-800 mb-2 flex items-center gap-2">
                            <Shield className="w-4 h-4" /> Recommended Action
                          </p>
                          <p className="text-slate-700 text-sm leading-relaxed">{analysisResult.treatments}</p>
                        </div>
                        <div className="bg-red-50/50 p-5 rounded-2xl border border-red-100">
                          <p className="text-sm font-bold text-red-800 mb-2 flex items-center gap-2">
                            <AlertCircle className="w-4 h-4" /> Medical Warning
                          </p>
                          <p className="text-slate-700 text-sm leading-relaxed">{analysisResult.warning}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex gap-4">
                  <button
                    onClick={resetAnalysis}
                    disabled={isAnalyzing}
                    className="flex-1 px-4 py-4 bg-white border-2 border-slate-200 text-slate-700 font-bold rounded-2xl hover:border-rose-400 hover:text-rose-700 transition-all duration-200 disabled:opacity-50 hover:shadow-lg active:scale-95 flex items-center justify-center gap-2 group"
                  >
                    <RotateCcw className="w-5 h-5 group-hover:-rotate-90 transition-transform duration-300" />
                    Scan Another Image
                  </button>
                  
                  {analysisResult && (
                    <button
                      onClick={generateReport}
                      disabled={isGeneratingPdf}
                      className="flex-1 px-4 py-4 bg-rose-50 text-rose-700 font-bold rounded-2xl hover:bg-rose-100 transition-all duration-200 disabled:opacity-50 hover:shadow-lg active:scale-95 flex items-center justify-center gap-2 border border-rose-200"
                    >
                      {isGeneratingPdf ? (
                        <><Loader2 className="w-5 h-5 animate-spin" /> Generating PDF...</>
                      ) : (
                        <><Download className="w-5 h-5" /> Download Report</>
                      )}
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Chatbot (Spans 5 cols on large screens) */}
          <div className="lg:col-span-5 relative">
            {analysisResult ? (
              <div className="sticky top-28 flex flex-col h-[700px] bg-white/80 backdrop-blur-xl rounded-[2rem] shadow-2xl shadow-slate-200/50 border border-white overflow-hidden animate-slide-left ring-1 ring-slate-900/5">
                
                {/* Chat Header */}
                <div className="bg-slate-900 p-5 flex items-center gap-4">
                  <div className="relative">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-rose-400 to-amber-500 flex items-center justify-center shadow-lg">
                      <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div className="absolute bottom-0 right-0 w-3 h-3 bg-indigo-400 border-2 border-slate-900 rounded-full"></div>
                  </div>
                  <div>
                    <h3 className="font-bold text-white text-lg flex items-center gap-2">
                      DermaBot AI
                    </h3>
                    <p className="text-xs text-slate-400 font-medium">Expert on {analysisResult.likelyPattern}</p>
                  </div>
                </div>

                {/* Chat Messages Area */}
                <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar bg-slate-50/50">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
                    >
                      {msg.sender === 'bot' && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-400 to-amber-500 flex items-center justify-center flex-shrink-0 mr-3 mt-1 shadow-sm">
                          <Brain className="w-4 h-4 text-white" />
                        </div>
                      )}
                      
                      <div
                        className={`max-w-[80%] px-5 py-3.5 rounded-2xl text-sm shadow-sm ${
                          msg.sender === 'user'
                            ? 'bg-slate-900 text-white rounded-tr-sm'
                            : 'bg-white border border-slate-100 text-slate-700 rounded-tl-sm shadow-slate-200/50'
                        }`}
                      >
                        {msg.sender === 'bot' ? (
                          <FormattedText text={msg.text} />
                        ) : (
                          <span className="font-medium">{msg.text}</span>
                        )}
                      </div>
                    </div>
                  ))}
                  
                  {isTyping && (
                    <div className="flex justify-start items-end">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-400 to-amber-500 flex items-center justify-center flex-shrink-0 mr-3 shadow-sm">
                        <Brain className="w-4 h-4 text-white" />
                      </div>
                      <div className="bg-white border border-slate-100 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm flex gap-1.5 items-center">
                        <div className="w-2 h-2 bg-rose-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-rose-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-rose-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} className="h-2" />
                </div>

                {/* Chat Input Area */}
                <div className="p-4 bg-white border-t border-slate-100">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleSendMessage();
                    }}
                    className="relative flex items-center"
                  >
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="Ask about symptoms, treatments..."
                      className="w-full pl-5 pr-14 py-4 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:border-rose-500 transition-all font-medium text-slate-700 placeholder-slate-400"
                    />
                    <button
                      type="submit"
                      disabled={!inputText.trim() || isTyping}
                      className="absolute right-2 p-2.5 bg-slate-900 text-white rounded-lg hover:bg-rose-600 transition-all duration-200 disabled:opacity-50 disabled:hover:bg-slate-900 shadow-md"
                    >
                      <Send className="w-4 h-4 ml-0.5" />
                    </button>
                  </form>
                </div>
              </div>
            ) : (
              // Empty State for Chat (Before Analysis)
              <div className="h-full min-h-[400px] lg:h-[700px] bg-white/40 backdrop-blur-sm rounded-[2rem] border-2 border-dashed border-slate-200 flex flex-col items-center justify-center p-8 text-center">
                <div className="w-20 h-20 bg-slate-100 rounded-3xl flex items-center justify-center mb-6 shadow-inner">
                  <MessageCircle className="w-10 h-10 text-slate-300" />
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2">AI Assistant Standby</h3>
                <p className="text-slate-500 max-w-xs leading-relaxed">
                  Upload an image first. Once the analysis is complete, you can chat with DermaBot to learn more about the condition.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* HIDDEN PDF TEMPLATE */}
      <div 
        ref={pdfTemplateRef} 
        style={{ display: 'none', width: '680px', backgroundColor: 'white', padding: '30px', color: '#1e293b', fontFamily: 'sans-serif' }}
      >
        <div style={{ borderBottom: '2px solid #e11d48', paddingBottom: '15px', marginBottom: '25px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ fontSize: '32px', fontWeight: 'bold', margin: 0, color: '#e11d48' }}>DermaAI Report</h1>
          <p style={{ fontSize: '14px', color: '#64748b' }}>Generated on: {new Date().toLocaleDateString()}</p>
        </div>

        <div style={{ display: 'flex', gap: '30px', marginBottom: '40px' }}>
          {uploadedImage && (
            <div style={{ flex: '1' }}>
              <p style={{ fontWeight: 'bold', marginBottom: '10px' }}>Uploaded Image</p>
              <img src={uploadedImage} alt="Uploaded Skin" style={{ width: '100%', borderRadius: '12px', border: '1px solid #cbd5e1' }} />
            </div>
          )}
          
          <div style={{ flex: '1', backgroundColor: '#fff1f2', padding: '20px', borderRadius: '12px', border: '1px solid #fecdd3' }}>
            <p style={{ fontSize: '12px', textTransform: 'uppercase', color: '#be123c', fontWeight: 'bold', marginBottom: '5px' }}>Detection Result</p>
            <h2 style={{ fontSize: '28px', fontWeight: 'bold', margin: '0 0 10px 0' }}>{analysisResult?.likelyPattern}</h2>
            
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
              <span style={{ padding: '4px 10px', backgroundColor: '#1e293b', color: 'white', borderRadius: '6px', fontSize: '12px', fontWeight: 'bold' }}>
                {analysisResult?.confidence}% Match
              </span>
              <span style={{ padding: '4px 10px', backgroundColor: '#fff', color: '#be123c', borderRadius: '6px', fontSize: '12px', fontWeight: 'bold', border: '1px solid #fecdd3' }}>
                {analysisResult?.severity} Risk
              </span>
            </div>
            
            <p style={{ fontSize: '14px', lineHeight: '1.6' }}>{analysisResult?.description}</p>
          </div>
        </div>

        {reportRemedies.length > 0 && (
          <div style={{ marginBottom: '40px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: 'bold', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px', marginBottom: '15px' }}>Recommended Home Remedies</h3>
            {reportRemedies.map((remedy, idx) => (
              <div key={idx} style={{ marginBottom: '15px', padding: '15px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <h4 style={{ margin: '0 0 5px 0', fontSize: '16px', color: '#0f172a' }}>{remedy.title}</h4>
                <p style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#475569' }}>{remedy.description}</p>
                {remedy.warning && (
                  <p style={{ margin: 0, fontSize: '13px', color: '#b91c1c', fontWeight: 'bold' }}>⚠️ {remedy.warning}</p>
                )}
              </div>
            ))}
          </div>
        )}

        {reportHospitals.length > 0 && (
          <div style={{ marginBottom: '40px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: 'bold', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px', marginBottom: '15px' }}>Nearby Dermatology Clinics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              {reportHospitals.map((clinic, idx) => {
                const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(clinic.name + ' ' + clinic.address)}`;
                return (
                  <div key={idx} style={{ padding: '15px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                    <h4 style={{ margin: '0 0 5px 0', fontSize: '14px', color: '#0f172a' }}>
                      <a href={mapsUrl} target="_blank" rel="noreferrer" style={{ color: '#0284c7', textDecoration: 'none' }}>
                        {clinic.name} ↗
                      </a>
                    </h4>
                    <p style={{ margin: '0 0 5px 0', fontSize: '12px', color: '#64748b' }}>{clinic.address}</p>
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#e11d48' }}>{clinic.distance} km away</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {reportVideos.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: 'bold', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px', marginBottom: '15px' }}>Recommended Video Resources</h3>
            {reportVideos.map((video, idx) => (
              <div key={idx} style={{ marginBottom: '10px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#e11d48' }}>▶ {video.channel}</span>
                <span style={{ fontSize: '14px', color: '#334155' }}>— {video.title}</span>
                <span style={{ fontSize: '12px', color: '#64748b' }}>(Search on YouTube)</span>
              </div>
            ))}
          </div>
        )}
        
        <div style={{ marginTop: '50px', textAlign: 'center', fontSize: '12px', color: '#94a3b8', borderTop: '1px solid #e2e8f0', paddingTop: '20px' }}>
          Disclaimer: This report is generated by AI and is for educational purposes only. It is not a substitute for professional medical advice.
        </div>
      </div>

    </div>
  );
}
