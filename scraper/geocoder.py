from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationGeocoder:
    def __init__(self, user_agent: str = "ezerScraper"):
        """
        Initialize the geocoder with a custom user agent.
        
        Args:
            user_agent (str): Custom user agent string for the geocoding service
        """
        self.geolocator = Nominatim(user_agent=user_agent)
        self.cache = {}  # Simple in-memory cache to avoid repeated geocoding

    def geocode_address(self, address: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
        """
        Geocode an address to get its latitude and longitude coordinates.
        
        Args:
            address (str): The address to geocode
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Optional[Tuple[float, float]]: Tuple of (latitude, longitude) or None if geocoding failed
        """
        if not address:
            return None

        # Check cache first
        if address in self.cache:
            return self.cache[address]

        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(address)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.cache[address] = coords  # Cache the result
                    return coords
                break  # If no location found, don't retry
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"Geocoding attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to geocode address after {max_retries} attempts: {address}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error while geocoding address {address}: {e}")
                return None

        return None

    def geocode_locations(self, locations: List[Dict]) -> List[Dict]:
        """
        Add geocoded coordinates to a list of location dictionaries.
        
        Args:
            locations (List[Dict]): List of location dictionaries containing address information
            
        Returns:
            List[Dict]: List of location dictionaries with added latitude and longitude
        """
        geocoded_locations = []
        
        for location in locations:
            address = location.get('address')
            if not address:
                continue
                
            coords = self.geocode_address(address)
            if coords:
                location['latitude'] = coords[0]
                location['longitude'] = coords[1]
                geocoded_locations.append(location)
            else:
                logger.warning(f"Could not geocode address: {address}")
                # Still add the location without coordinates
                geocoded_locations.append(location)
        
        return geocoded_locations

def geocode_locations_data(locations: List[Dict]) -> List[Dict]:
    """
    Convenience function to geocode a list of locations.
    
    Args:
        locations (List[Dict]): List of location dictionaries
        
    Returns:
        List[Dict]: List of location dictionaries with added coordinates
    """
    geocoder = LocationGeocoder()
    return geocoder.geocode_locations(locations) 