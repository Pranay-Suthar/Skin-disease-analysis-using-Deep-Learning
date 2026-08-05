import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Microscope, History, Activity, Map, MessageSquare, LogIn, ExternalLink } from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import HospitalLocator from './components/HospitalLocator';
import SkinBotChat from './components/SkinBotChat';

const Navbar = () => (
  <nav className="flex justify-between items-center px-6 py-4 bg-surface/90 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
    <Link to="/" className="flex items-center gap-2 text-xl font-bold text-white hover:text-primary transition-colors">
      <Microscope className="w-6 h-6 text-primary" />
      <span>AI Skin Checker</span>
    </Link>
    <div className="flex items-center gap-4">
      <a href="https://github.com/Pranay-Suthar" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors px-3 py-1.5 rounded-lg hover:bg-slate-800 text-sm font-medium">
        <ExternalLink className="w-4 h-4" />
        <span>Contact Developer</span>
      </a>
    </div>
  </nav>
);

const Home = () => {
  const [results, setResults] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary mb-4 flex justify-center items-center gap-3">
            <Microscope className="w-10 h-10 text-primary" /> AI Skin Disease Checker
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Upload a skin image for instant AI analysis, find nearby dermatologists, and chat with SkinBot for guidance.
          </p>  
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="glass-card">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                <Activity className="w-5 h-5 text-primary" /> Upload Image
              </h2>
              <ImageUploader onResults={setResults} onImagePreview={setImagePreview} />
            </div>
            {results && (
              <div className="glass-card">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                  <Map className="w-5 h-5 text-primary" /> Find Nearby Hospitals
                </h2>
                <HospitalLocator diseaseName={results.disease} />
              </div>
            )}
          </div>
          
          <div className="space-y-6">
            <div className="glass-card h-full min-h-[400px]">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                <Microscope className="w-5 h-5 text-secondary" /> Analysis Results
              </h2>
              {results ? (
                <ResultsDisplay results={results} imagePreview={imagePreview} />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-4 min-h-[300px]">
                  <Activity className="w-12 h-12 opacity-50" />
                  <p>Upload an image to view diagnostic results.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
      
      {results && <SkinBotChat disease={results.disease} info={results.info} />}
      
      <footer className="text-center py-6 text-slate-500 text-sm border-t border-slate-800 mt-auto bg-surface/50">
        <p>MEDICAL DISCLAIMER: This AI tool is for educational purposes only. Always consult a qualified healthcare professional.</p>
      </footer>
    </div>
  );
};

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </BrowserRouter>
  );
}
