import React from 'react';
import { AlertTriangle, AlertCircle, CheckCircle, Info, Thermometer, ShieldAlert } from 'lucide-react';

export default function ResultsDisplay({ results, imagePreview }) {
  if (!results) return null;

  const { disease, confidence, top_3, severity, info } = results;

  const getSeverityColor = (sev) => {
    switch (sev) {
      case 'Critical': return 'text-danger bg-danger/10 border-danger/30';
      case 'High': return 'text-warning bg-warning/10 border-warning/30';
      case 'Moderate': return 'text-secondary bg-secondary/10 border-secondary/30';
      case 'Low': return 'text-success bg-success/10 border-success/30';
      default: return 'text-slate-400 bg-slate-800 border-slate-700';
    }
  };

  const getSeverityIcon = (sev) => {
    switch (sev) {
      case 'Critical': return <ShieldAlert className="w-5 h-5" />;
      case 'High': return <AlertTriangle className="w-5 h-5" />;
      case 'Moderate': return <AlertCircle className="w-5 h-5" />;
      case 'Low': return <CheckCircle className="w-5 h-5" />;
      default: return <Info className="w-5 h-5" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className={`p-4 rounded-xl border flex items-start gap-4 ${getSeverityColor(severity)}`}>
        <div className="mt-1">{getSeverityIcon(severity)}</div>
        <div>
          <h3 className="text-xl font-bold mb-1">{info?.name || disease}</h3>
          <p className="opacity-90 text-sm mb-2">{info?.description || 'Detected condition.'}</p>
          <div className="flex items-center gap-2 font-semibold">
            <span>Confidence: {(confidence * 100).toFixed(1)}%</span>
            <span className="px-2 py-0.5 rounded-full text-xs border border-current uppercase">
              {severity} Severity
            </span>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Top Differential Diagnoses</h4>
        <div className="space-y-3">
          {top_3.map((item, idx) => (
            <div key={idx}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-200">{item.disease}</span>
                <span className="text-slate-400">{(item.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div 
                  className="bg-primary h-2 rounded-full" 
                  style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-4 pt-4 border-t border-slate-800">
        <div>
          <h4 className="text-sm font-semibold text-primary uppercase tracking-wider mb-2 flex items-center gap-2">
            <Thermometer className="w-4 h-4" /> Causes & Risk Factors
          </h4>
          <p className="text-sm text-slate-300">{info?.causes || 'Unknown causes. Consult a professional.'}</p>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
            <h4 className="text-xs font-semibold text-secondary uppercase mb-1">Medical Treatments</h4>
            <p className="text-sm text-slate-300">{info?.treatments || 'Consult a dermatologist'}</p>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
            <h4 className="text-xs font-semibold text-success uppercase mb-1">Home Care</h4>
            <p className="text-sm text-slate-300">{info?.home_care || 'Monitor for changes'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
