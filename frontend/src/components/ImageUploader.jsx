import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, X, Loader2 } from 'lucide-react';

const API_URL = 'http://127.0.0.1:5000/api';

export default function ImageUploader({ onResults, onImagePreview }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    setError('');
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      onImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
    uploadImage(file);
  };

  const uploadImage = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(`${API_URL}/predict/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onResults(response.data);
    } catch (err) {
      console.error(err);
      setError('Failed to analyze image. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setPreview(null);
    onImagePreview(null);
    onResults(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex flex-col items-center">
      {!preview ? (
        <div 
          className="w-full border-2 border-dashed border-slate-600 rounded-xl p-10 flex flex-col items-center justify-center bg-slate-800/30 hover:bg-slate-800/50 transition-colors cursor-pointer"
          onClick={() => fileInputRef.current.click()}
        >
          <Upload className="w-12 h-12 text-slate-400 mb-4" />
          <p className="text-slate-300 font-medium mb-1">Click to upload skin image</p>
          <p className="text-slate-500 text-sm">PNG, JPG or JPEG (max. 10MB)</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="image/png, image/jpeg, image/jpg" 
            onChange={handleFileChange}
          />
        </div>
      ) : (
        <div className="relative w-full rounded-xl overflow-hidden bg-slate-900 border border-slate-700 aspect-video flex items-center justify-center">
          <img src={preview} alt="Uploaded" className="max-w-full max-h-full object-contain" />
          <button 
            onClick={clearImage}
            className="absolute top-2 right-2 p-1.5 bg-black/50 hover:bg-black/80 rounded-full text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          {loading && (
            <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center">
              <Loader2 className="w-10 h-10 text-primary animate-spin mb-2" />
              <p className="text-white font-medium">Analyzing Image...</p>
            </div>
          )}
        </div>
      )}
      {error && <p className="mt-4 text-danger font-medium text-sm">{error}</p>}
    </div>
  );
}
