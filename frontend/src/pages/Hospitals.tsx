import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { MapPin, Phone, Search, Navigation, Loader, AlertCircle, Building2, ShieldCheck, Clock, ExternalLink, Activity } from 'lucide-react';
import { getUserLocation } from '../utils/geolocation';

export default function Hospitals() {
  const [clinics, setClinics] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userLocation, setUserLocation] = useState<string>('Detecting location...');
  const [manualLocation, setManualLocation] = useState('');

  const fetchClinics = async (lat?: number, lon?: number, locStr?: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/hospitals/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: lat,
          lon: lon,
          location: locStr
        })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch clinics');
      }

      setClinics(data.hospitals || []);
      setUserLocation(data.user_location?.display_name || 'Your Location');
      
      if (data.hospitals?.length === 0) {
        setError('No clinics found in this area within 100km.');
      }
    } catch (err: any) {
      console.error('Error fetching clinics:', err);
      setError(err.message || 'Unable to fetch clinics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initLocation = async () => {
      try {
        const location = await getUserLocation();
        await fetchClinics(location.latitude, location.longitude);
      } catch (e) {
        setUserLocation('Unknown Location');
        setError('Location detection failed. Please enter a manual location.');
      }
    };
    initLocation();
  }, []);

  const handleManualSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (manualLocation.trim()) {
      fetchClinics(undefined, undefined, manualLocation);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 pt-28 overflow-x-hidden">
      <Header />

      <main className="mx-auto max-w-7xl px-5 py-12">
        {/* Header Section */}
        <div className="mb-16 relative">
          {/* Decorative background blur */}
          <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 bg-rose-100 rounded-full blur-3xl opacity-50 pointer-events-none"></div>
          
          <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-8">
            <div className="max-w-2xl animate-slide-up">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 text-amber-600 text-sm font-bold mb-4 border border-amber-100">
                <ShieldCheck className="w-4 h-4" /> Trusted Healthcare Partners
              </div>
              <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">
                Find Local <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-600 to-amber-600">Dermatologists</span>
              </h1>
              <p className="text-lg text-slate-600">
                Instantly discover specialized skin clinics and verified healthcare professionals near you.
              </p>
            </div>
            
            {/* Search Box */}
            <div className="w-full md:w-[400px] animate-slide-left">
              <form onSubmit={handleManualSearch} className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-rose-600" />
                </div>
                <input 
                  type="text" 
                  placeholder="Enter city or zip code..." 
                  value={manualLocation}
                  onChange={(e) => setManualLocation(e.target.value)}
                  className="block w-full pl-12 pr-32 py-4 text-base border-2 border-white rounded-2xl bg-white shadow-xl shadow-slate-200/50 focus:outline-none focus:ring-4 focus:ring-rose-500/20 focus:border-rose-500 transition-all font-medium text-slate-700"
                />
                <button type="submit" className="absolute inset-y-2 right-2 bg-gradient-to-r from-rose-600 to-indigo-500 text-white px-6 py-2 rounded-xl font-bold hover:shadow-lg hover:scale-105 transition-all active:scale-95 text-sm flex items-center gap-2">
                  Search
                </button>
              </form>
              
              <div className="mt-4 flex items-center gap-2 px-2 text-sm text-slate-500 bg-white/50 backdrop-blur-md rounded-lg py-2 inline-flex border border-slate-200/50">
                <Navigation className="w-4 h-4 text-rose-600 animate-pulse" />
                <span className="font-medium text-slate-700 truncate max-w-[300px]">{userLocation}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Status Indicators */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-32 bg-white rounded-3xl border border-dashed border-slate-300 shadow-sm">
            <div className="relative mb-4">
              <div className="absolute inset-0 bg-rose-200 rounded-full blur-xl animate-pulse"></div>
              <Loader className="w-12 h-12 text-rose-600 animate-spin relative z-10" />
            </div>
            <p className="text-slate-600 font-semibold text-lg">Scanning for nearby clinics...</p>
            <p className="text-slate-400 text-sm mt-1">This may take a few moments</p>
          </div>
        )}

        {error && !loading && (
          <div className="p-8 bg-white rounded-3xl border border-red-100 shadow-xl shadow-red-100/50 flex flex-col items-center text-center max-w-2xl mx-auto">
            <div className="w-16 h-16 bg-red-50 text-red-500 rounded-2xl flex items-center justify-center mb-4">
              <AlertCircle className="w-8 h-8" />
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-2">Search Interrupted</h3>
            <p className="text-slate-600 mb-6">{error}</p>
            <button 
              onClick={() => {
                setManualLocation('');
                const initLocation = async () => {
                  try {
                    const location = await getUserLocation();
                    await fetchClinics(location.latitude, location.longitude);
                  } catch (e) {
                    setError('Auto-detect failed. Please type a location.');
                  }
                };
                initLocation();
              }}
              className="px-6 py-3 bg-slate-900 text-white font-bold rounded-xl hover:bg-slate-800 transition-colors flex items-center gap-2"
            >
              <Navigation className="w-4 h-4" /> Try Auto-Detect Again
            </button>
          </div>
        )}

        {/* Results Grid */}
        {!loading && clinics.length > 0 && (
          <div className="animate-fade-in">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                <Building2 className="w-6 h-6 text-rose-600" />
                Available Facilities
              </h2>
              <span className="bg-rose-50 text-rose-700 font-bold px-4 py-1.5 rounded-full text-sm border border-rose-100">
                {clinics.length} Results
              </span>
            </div>
            
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {clinics.map((clinic, idx) => {
                const distanceVal = clinic.distance !== undefined ? clinic.distance.toFixed(1) : (clinic.distance_km !== undefined ? clinic.distance_km.toFixed(1) : '?');
                
                return (
                <div key={clinic.id || idx} className="bg-white rounded-3xl p-6 shadow-lg shadow-slate-200/40 border border-slate-100 hover:shadow-2xl hover:border-rose-200 transition-all duration-300 flex flex-col relative group overflow-hidden">
                  
                  {/* Subtle top gradient bar */}
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-rose-400 to-amber-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  
                  <div className="flex justify-between items-start mb-4">
                    <div className="p-3 bg-amber-50 text-amber-600 rounded-2xl group-hover:scale-110 transition-transform duration-300">
                      <Activity className="w-6 h-6" />
                    </div>
                    {clinic.openNow !== undefined && (
                      <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                        clinic.openNow 
                          ? 'bg-indigo-50 text-indigo-700 border border-indigo-100' 
                          : 'bg-rose-50 text-rose-700 border border-rose-100'
                      }`}>
                        <div className={`w-2 h-2 rounded-full animate-pulse ${clinic.openNow ? 'bg-indigo-500' : 'bg-rose-500'}`}></div>
                        {clinic.openNow ? 'Open Now' : 'Closed'}
                      </div>
                    )}
                  </div>
                  
                  <h3 className="font-extrabold text-xl text-slate-900 mb-2 leading-tight group-hover:text-rose-700 transition-colors">
                    {clinic.name}
                  </h3>
                  
                  <div className="space-y-3 mb-8 flex-grow">
                    <div className="flex items-start gap-3 text-sm text-slate-600">
                      <MapPin className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                      <span className="leading-relaxed">{clinic.address}</span>
                    </div>
                    
                    <div className="flex items-center gap-3 text-sm font-semibold text-amber-600 bg-amber-50/50 p-2 rounded-lg inline-flex">
                      <Navigation className="w-4 h-4" />
                      {distanceVal} km away
                    </div>
                    
                    {clinic.phone && (
                      <div className="flex items-center gap-3 text-sm font-medium text-slate-700">
                        <Phone className="w-4 h-4 text-rose-600" />
                        <a href={`tel:${clinic.phone}`} className="hover:text-rose-600 hover:underline transition-all">
                          {clinic.phone}
                        </a>
                      </div>
                    )}
                  </div>

                  <div className="flex gap-3 mt-auto">
                    <a 
                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(clinic.name)}+${encodeURIComponent(clinic.address || '')}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-1 bg-slate-900 text-white px-4 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 hover:bg-slate-800 hover:shadow-lg transition-all active:scale-95"
                    >
                      <MapPin className="w-4 h-4" /> Directions
                    </a>
                    {clinic.phone && (
                      <a 
                        href={`tel:${clinic.phone}`}
                        className="bg-rose-50 text-rose-700 border border-rose-100 px-4 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 hover:bg-rose-100 transition-all active:scale-95"
                      >
                        <Phone className="w-4 h-4" /> Call
                      </a>
                    )}
                  </div>
                </div>
              )})}
            </div>
          </div>
        )}

        {!loading && clinics.length === 0 && !error && (
          <div className="text-center py-32 bg-white rounded-3xl border border-dashed border-slate-300">
            <Building2 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-600 text-xl font-bold">No facilities found</p>
            <p className="text-slate-400 text-sm mt-2">Try searching for a different city or zip code</p>
          </div>
        )}
      </main>
    </div>
  );
}
