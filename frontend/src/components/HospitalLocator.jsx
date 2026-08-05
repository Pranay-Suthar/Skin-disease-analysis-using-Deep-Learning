import React, { useState } from 'react';
import axios from 'axios';
import { Search, MapPin, Navigation, Phone, ExternalLink, Loader2 } from 'lucide-react';

const API_URL = 'http://127.0.0.1:5000/api';

export default function HospitalLocator({ diseaseName }) {
  const [location, setLocation] = useState('');
  const [loading, setLoading] = useState(false);
  const [hospitals, setHospitals] = useState([]);
  const [userLoc, setUserLoc] = useState(null);
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!location.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_URL}/hospitals/`, { location });
      setHospitals(response.data.hospitals);
      setUserLoc(response.data.user_location);
    } catch (err) {
      setError('Could not locate hospitals. Try a more specific city or address.');
      setHospitals([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="relative flex-1">
          <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input 
            type="text" 
            placeholder="Enter city or address (e.g. New York, NY)" 
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary placeholder-slate-500"
          />
        </div>
        <button 
          type="submit" 
          disabled={loading}
          className="btn-primary whitespace-nowrap"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
          <span>Search</span>
        </button>
      </form>

      {error && <p className="text-sm text-danger">{error}</p>}

      {userLoc && (
        <div className="text-sm text-success flex items-center gap-2 mb-2">
          <MapPin className="w-4 h-4" /> Showing results near: <span className="font-semibold">{userLoc.display_name}</span>
        </div>
      )}

      {hospitals.length > 0 && (
        <div className="grid gap-3">
          {hospitals.map((hosp, idx) => {
            const gmapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${userLoc.lat},${userLoc.lon}&destination=${hosp.lat},${hosp.lon}&travelmode=driving`;
            return (
              <div key={idx} className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex justify-between items-start gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-xs font-bold text-white">{idx + 1}</span>
                    <h4 className="font-semibold text-slate-200 leading-tight">{hosp.name}</h4>
                  </div>
                  <div className="flex items-center gap-2 text-xs mb-2">
                    <span className="bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">{hosp.type}</span>
                    <span className="text-primary font-medium flex items-center gap-1"><Navigation className="w-3 h-3" /> {hosp.distance_km} km</span>
                  </div>
                  <p className="text-xs text-slate-400">{hosp.address}</p>
                </div>
                <a 
                  href={gmapsUrl} 
                  target="_blank" 
                  rel="noreferrer"
                  className="bg-primary/10 hover:bg-primary/20 text-primary p-2 rounded-lg transition-colors flex-shrink-0"
                  title="Get Directions"
                >
                  <ExternalLink className="w-5 h-5" />
                </a>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
