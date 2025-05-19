from geopy.geocoders import Nominatim, GoogleV3, ArcGIS, Photon
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import logging
from typing import Dict, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationGeocoder:
    def __init__(self, user_agent: str = "ezerScraper", google_api_key: str = None):
        """
        Initialize the geocoder with multiple providers for better reliability.
        
        Args:
            user_agent (str): Custom user agent string for the geocoding services
            google_api_key (str, optional): Google Maps API key for better results
        """
        # Initialize geocoders with rate limiting
        self.nominatim = RateLimiter(
            Nominatim(user_agent=user_agent).geocode,
            min_delay_seconds=1.5
        )
        self.arcgis = RateLimiter(
            ArcGIS().geocode,
            min_delay_seconds=1.0
        )
        self.photon = RateLimiter(
            Photon().geocode,
            min_delay_seconds=1.0
        )
        if google_api_key:
            self.google = RateLimiter(
                GoogleV3(api_key=google_api_key).geocode,
                min_delay_seconds=0.5
            )
        else:
            self.google = None
            
        self.cache = {}  # Simple in-memory cache

    def preprocess_address(self, address: str) -> str:
        """
        Clean and standardize address format.
        """
        if not address:
            return ""
            
        # Convert to string and clean basic issues
        address = str(address).strip()
        
        # Remove extra whitespace
        address = re.sub(r'\s+', ' ', address)
        
        # Add Tunisia if not present (since we're working with Tunisian addresses)
        if 'tunisia' not in address.lower() and 'tunisie' not in address.lower():
            address += ', Tunisia'
            
        # Replace common abbreviations
        replacements = {
            'ave ': 'avenue ',
            'ave. ': 'avenue ',
            'bd ': 'boulevard ',
            'bd. ': 'boulevard ',
            'st ': 'street ',
            'st. ': 'street ',
            'apt ': 'apartment ',
            'apt. ': 'apartment ',
            'n° ': 'number ',
            'no. ': 'number ',
        }
        
        for old, new in replacements.items():
            address = re.sub(rf'\b{old}\b', new, address, flags=re.IGNORECASE)
            
        return address

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate if coordinates are within valid ranges.
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180

    def geocode_with_provider(self, address: str, provider, provider_name: str) -> Optional[Tuple[float, float]]:
        """
        Try geocoding with a specific provider.
        """
        try:
            location = provider(address)
            if location:
                lat, lon = location.latitude, location.longitude
                if self.validate_coordinates(lat, lon):
                    logger.info(f"Successfully geocoded with {provider_name}: {address}")
                    return (lat, lon)
                else:
                    logger.warning(f"{provider_name} returned invalid coordinates: {lat}, {lon}")
        except Exception as e:
            logger.warning(f"{provider_name} geocoding failed for {address}: {str(e)}")
        return None

    def geocode_address(self, address: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
        """
        Try geocoding with multiple providers in sequence.
        """
        if not address:
            return None

        # Check cache first
        if address in self.cache:
            return self.cache[address]

        # Preprocess the address
        clean_address = self.preprocess_address(address)
        if not clean_address:
            return None

        # Try each provider in sequence
        providers = [
            (self.google, "Google") if self.google else None,
            (self.nominatim, "Nominatim"),
            (self.photon, "Photon"),
            (self.arcgis, "ArcGIS")
        ]
        
        for provider_tuple in providers:
            if not provider_tuple:
                continue
                
            provider, name = provider_tuple
            for attempt in range(max_retries):
                try:
                    coords = self.geocode_with_provider(clean_address, provider, name)
                    if coords:
                        self.cache[address] = coords
                        return coords
                    break  # If no location found, try next provider
                except (GeocoderTimedOut, GeocoderUnavailable) as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"{name} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed with {name} after {max_retries} attempts: {address}")

        logger.warning(f"Could not geocode: {address}")
        return None

    def geocode_locations(self, locations: List[Dict]) -> List[Dict]:
        """
        Add geocoded coordinates to a list of location dictionaries.
        """
        geocoded_locations = []
        total = len(locations)
        
        for idx, location in enumerate(locations, 1):
            address = location.get('address')
            if not address:
                continue
                
            logger.info(f"Geocoding address {idx}/{total}: {address}")
            coords = self.geocode_address(address)
            
            if coords:
                location['latitude'] = coords[0]
                location['longitude'] = coords[1]
                logger.info(f"Successfully geocoded: {address} -> {coords}")
            else:
                logger.warning(f"Could not geocode: {address}")
            
            geocoded_locations.append(location)
            
            # Add a small delay between requests
            time.sleep(0.5)
        
        return geocoded_locations

def geocode_locations_data(locations: List[Dict], google_api_key: str = None) -> List[Dict]:
    """
    Convenience function to geocode a list of locations.
    
    Args:
        locations (List[Dict]): List of location dictionaries
        google_api_key (str, optional): Google Maps API key for better results
        
    Returns:
        List[Dict]: List of location dictionaries with added coordinates
    """
    geocoder = LocationGeocoder(google_api_key=google_api_key)
    return geocoder.geocode_locations(locations) 