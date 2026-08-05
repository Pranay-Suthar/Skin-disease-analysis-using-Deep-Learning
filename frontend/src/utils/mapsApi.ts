// Google Maps API service for finding dermatology clinics
// Note: You need to set up your Google Places API key in .env file

export interface Clinic {
  id: string;
  name: string;
  location: string;
  distance: number; // in km
  rating: number;
  reviews: number;
  phone?: string;
  address: string;
  latitude: number;
  longitude: number;
  website?: string;
  openNow?: boolean;
}

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

// Haversine formula to calculate distance between two coordinates
const calculateDistance = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number => {
  const R = 6371; // Earth's radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

export const findDermatologyClinics = async (
  latitude: number,
  longitude: number,
  radiusKm: number = 100
): Promise<Clinic[]> => {
  try {
    if (!GOOGLE_MAPS_API_KEY) {
      console.warn('Google Maps API key not configured. Using mock data.');
      return getMockClinics(latitude, longitude);
    }

    const radiusMeters = radiusKm * 1000;

    // Search for dermatologists and skin clinics
    const searchTerms = [
      'dermatologist',
      'dermatology clinic',
      'skin clinic',
      'skin care center',
    ];

    let allClinics: Clinic[] = [];

    for (const term of searchTerms) {
      const response = await fetch(
        `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${latitude},${longitude}&radius=${radiusMeters}&keyword=${term}&key=${GOOGLE_MAPS_API_KEY}`
      );

      const data = await response.json();

      if (data.results) {
        const clinics = data.results
          .filter((place: any) => place.rating && place.rating > 3.5) // Filter by rating
          .map((place: any) => ({
            id: place.place_id,
            name: place.name,
            location: place.vicinity,
            distance: calculateDistance(
              latitude,
              longitude,
              place.geometry.location.lat,
              place.geometry.location.lng
            ),
            rating: place.rating,
            reviews: place.user_ratings_total || 0,
            address: place.vicinity,
            latitude: place.geometry.location.lat,
            longitude: place.geometry.location.lng,
            openNow: place.opening_hours?.open_now,
          }));

        allClinics = [...allClinics, ...clinics];
      }
    }

    // Remove duplicates and sort by distance
    const uniqueClinics = Array.from(
      new Map(allClinics.map((clinic) => [clinic.id, clinic])).values()
    );

    return uniqueClinics
      .filter((clinic) => clinic.distance <= radiusKm)
      .sort((a, b) => {
        // Sort by rating first, then by distance
        if (b.rating !== a.rating) {
          return b.rating - a.rating;
        }
        return a.distance - b.distance;
      })
      .slice(0, 20); // Return top 20
  } catch (error) {
    console.error('Error fetching clinics from Google Maps:', error);
    return getMockClinics(latitude, longitude);
  }
};

// Mock data for testing (when API key is not available)
const getMockClinics = (latitude: number, longitude: number): Clinic[] => {
  return [
    {
      id: '1',
      name: 'Premier Dermatology Center',
      location: 'City Center',
      distance: 2.5,
      rating: 4.8,
      reviews: 342,
      phone: '(555) 123-4567',
      address: '123 Main St, City Center',
      latitude: latitude + 0.02,
      longitude: longitude + 0.02,
      website: 'https://example.com',
      openNow: true,
    },
    {
      id: '2',
      name: 'Advanced Skin & Laser Clinic',
      location: 'Downtown',
      distance: 5.3,
      rating: 4.6,
      reviews: 289,
      phone: '(555) 234-5678',
      address: '456 Oak Ave, Downtown',
      latitude: latitude + 0.05,
      longitude: longitude - 0.03,
      website: 'https://example.com',
      openNow: true,
    },
    {
      id: '3',
      name: 'Dermatology Excellence',
      location: 'Medical District',
      distance: 8.7,
      rating: 4.7,
      reviews: 156,
      phone: '(555) 345-6789',
      address: '789 Park Blvd, Medical District',
      latitude: latitude - 0.06,
      longitude: longitude + 0.04,
      website: 'https://example.com',
      openNow: false,
    },
    {
      id: '4',
      name: 'Skincare Specialists',
      location: 'North Ridge',
      distance: 12.1,
      rating: 4.5,
      reviews: 198,
      phone: '(555) 456-7890',
      address: '321 Elm St, North Ridge',
      latitude: latitude + 0.08,
      longitude: longitude + 0.07,
      website: 'https://example.com',
      openNow: true,
    },
    {
      id: '5',
      name: 'Total Skin Care Clinic',
      location: 'Westside',
      distance: 15.4,
      rating: 4.4,
      reviews: 267,
      phone: '(555) 567-8901',
      address: '654 Pine Rd, Westside',
      latitude: latitude - 0.1,
      longitude: longitude - 0.08,
      website: 'https://example.com',
      openNow: true,
    },
  ];
};

export const getClinicDetails = async (placeId: string): Promise<any> => {
  if (!GOOGLE_MAPS_API_KEY) return null;

  try {
    const response = await fetch(
      `https://maps.googleapis.com/maps/api/place/details/json?place_id=${placeId}&key=${GOOGLE_MAPS_API_KEY}`
    );
    return await response.json();
  } catch (error) {
    console.error('Error fetching clinic details:', error);
    return null;
  }
};
