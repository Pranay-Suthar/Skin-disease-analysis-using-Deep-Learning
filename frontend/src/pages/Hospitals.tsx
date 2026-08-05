import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { MapPin, Phone, Star, Search, Navigation, Loader, AlertCircle } from 'lucide-react';
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
    <div className="min-h-screen bg-gradient-to-b from-white via-blue-50 to-white pt-28">
      <Header />

      <main className="mx-auto max-w-7xl px-5 py-12">
        <div className="mb-12">
          <h1 className="page-title mb-3 flex items-center gap-3">
            <MapPin className="w-10 h-10 text-teal-600" />
            Find Dermatology Care
          </h1>
          <p className="page-subtitle">Discover dermatologists and clinics near you (within 100km)</p>
          
          <form onSubmit={handleManualSearch} className="mt-6 flex max-w-md gap-2">
            <input 
              type="text" 
              placeholder="Enter manual location (e.g. London)" 
              value={manualLocation}
              onChange={(e) => setManualLocation(e.target.value)}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:outline-none"
            />
            <button type="submit" className="bg-teal-600 text-white px-4 py-2 rounded-lg hover:bg-teal-700 transition flex items-center gap-2">
              <Search className="w-4 h-4"/> Search
            </button>
          </form>

          <p className="text-sm text-gray-600 mt-4 flex items-center gap-2">
            <Navigation className="w-4 h-4 text-teal-600" />
            Showing clinics near: <span className="font-semibold text-gray-900">{userLocation}</span>
          </p>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader className="w-12 h-12 text-teal-600 animate-spin mb-4" />
            <p className="text-gray-600">Finding nearby dermatology clinics...</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="mb-8 p-6 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-bold text-red-900 mb-1">Unable to Load Clinics</h3>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Clinics List */}
        {!loading && clinics.length > 0 && (
          <div className="space-y-6">
            <div className="text-sm font-semibold text-teal-600 mb-4">
              Found {clinics.length} dermatology clinics near you
            </div>
            
            {clinics.map((clinic) => (
              <div key={clinic.id} className="card group hover:shadow-xl transition-all duration-300 border-l-4 border-teal-500">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="font-bold text-xl text-gray-900 group-hover:text-teal-600 transition-colors mb-2">
                      {clinic.name}
                    </h3>
                    <div className="flex flex-wrap gap-4 text-sm text-gray-600 mb-3">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-teal-600" />
                        <span>{clinic.address}</span>
                      </div>
                      <div className="flex items-center gap-2 text-blue-600 font-semibold">
                        <Navigation className="w-4 h-4" />
                        <span>{clinic.distance !== undefined ? clinic.distance.toFixed(1) : (clinic.distance_km !== undefined ? clinic.distance_km.toFixed(1) : '?')} km away</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <div className="flex items-center gap-1 bg-gradient-to-r from-yellow-100 to-amber-100 px-3 py-2 rounded-lg">
                      <Star className="w-4 h-4 text-yellow-600 fill-yellow-600" />
                      <span className="font-bold text-yellow-700">
                        {clinic.rating !== undefined ? clinic.rating.toFixed(1) : 'N/A'}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">{clinic.reviews || 0} reviews</span>
                  </div>
                </div>

                {clinic.openNow !== undefined && (
                  <div className={`mb-3 inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold ${
                    clinic.openNow 
                      ? 'bg-green-100 text-green-700' 
                      : 'bg-red-100 text-red-700'
                  }`}>
                    <div className={`w-2 h-2 rounded-full ${clinic.openNow ? 'bg-green-700' : 'bg-red-700'}`}></div>
                    {clinic.openNow ? 'Open Now' : 'Closed'}
                  </div>
                )}

                {clinic.phone && (
                  <div className="flex items-center gap-2 text-sm text-gray-600 mb-6 p-3 bg-blue-50 rounded-lg">
                    <Phone className="w-4 h-4 text-teal-600" />
                    <a href={`tel:${clinic.phone}`} className="hover:text-teal-600 transition-colors">
                      {clinic.phone}
                    </a>
                  </div>
                )}

                <div className="flex gap-3">
                  <a 
                    href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(clinic.name)}+${encodeURIComponent(clinic.address || '')}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 px-4 py-3 bg-gradient-to-r from-teal-600 to-emerald-600 text-white font-bold rounded-lg hover:from-teal-700 hover:to-emerald-700 transition-all duration-200 hover:scale-105 active:scale-95 transform flex items-center justify-center gap-2"
                  >
                    <MapPin className="w-4 h-4" />
                    View on Maps
                  </a>
                  <a 
                    href={`tel:${clinic.phone}`}
                    className="flex-1 px-4 py-3 border-2 border-teal-600 text-teal-600 font-bold rounded-lg hover:bg-teal-50 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <Phone className="w-4 h-4" />
                    Call Now
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && clinics.length === 0 && !error && (
          <div className="text-center py-20">
            <MapPin className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600 text-lg">No clinics found nearby</p>
            <p className="text-gray-500 text-sm mt-2">Please enable location services and try again</p>
          </div>
        )}
      </main>
    </div>
  );
}
