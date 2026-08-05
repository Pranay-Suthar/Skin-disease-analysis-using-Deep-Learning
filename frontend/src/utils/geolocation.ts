// Geolocation utility for getting user's current location
export interface LocationCoordinates {
  latitude: number;
  longitude: number;
  accuracy: number;
}

export interface LocationData {
  coordinates: LocationCoordinates;
  city: string;
  country: string;
}

export const getUserLocation = (): Promise<LocationCoordinates> => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by this browser.'));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        resolve({ latitude, longitude, accuracy });
      },
      (error) => {
        // Default to a major city if geolocation fails (can be customized)
        console.warn('Geolocation error:', error);
        // Using coordinates for a common location as fallback
        resolve({
          latitude: 40.7128, // New York latitude
          longitude: -74.006, // New York longitude
          accuracy: 1000,
        });
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0,
      }
    );
  });
};

export const getLocationName = async (
  latitude: number,
  longitude: number
): Promise<{ city: string; country: string }> => {
  try {
    // Using reverse geocoding API (can use Google's Geocoding API or another service)
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
    );
    const data = await response.json();
    
    const city = data.address?.city || data.address?.town || 'Unknown';
    const country = data.address?.country || 'Unknown';
    
    return { city, country };
  } catch (error) {
    console.error('Error getting location name:', error);
    return { city: 'Unknown', country: 'Unknown' };
  }
};
