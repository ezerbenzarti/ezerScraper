import pandas as pd
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def has_valid_phone(phone):
    """
    Check if the phone number is valid (not empty and has digits).
    Returns True if valid, False otherwise.
    """
    if not phone or pd.isna(phone):
        return False
    
    # Check if it contains at least 5 digits (minimum reasonable length)
    digits = re.sub(r'\D', '', str(phone))
    return len(digits) >= 5

def has_valid_email(email):
    """
    Enhanced email validation with better checks and filtering.
    Returns True if valid, False otherwise.
    """
    if not email or pd.isna(email):
        return False
    
    email = str(email).strip().lower()
    
    # Basic format validation with stricter rules
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]{0,63}@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,63}$', email):
        return False
    
    # Check for valid TLD length
    tld = email.split('.')[-1]
    if len(tld) < 2 or len(tld) > 63:
        return False
    
    # Check for consecutive special characters
    if re.search(r'[._%+-]{2,}', email):
        return False
    
    # Check for valid local part length
    local_part = email.split('@')[0]
    if len(local_part) > 64:
        return False
    
    # Check for placeholder emails
    placeholder_patterns = [
        "example", "exemple", "sample", "test", "demo",
        "your.email", "your-email", "your_email",
        "email@", "mail@", "contact@",
        "info@", "support@", "admin@",
        "user@", "username@", "name@",
        "someone@", "someone@example.com",
        "ton-email@", "votre-mail@", "votre-email@",
        "votre.email@", "votre_mail@", "votre_email@",
        "no-reply@", "noreply@", "no.reply@",
        "donotreply@", "do-not-reply@", "do.not.reply@",
        "postmaster@", "webmaster@", "hostmaster@",
        "emailaddress@", "email.address@",
        "myemail@", "my.email@", "my-email@",
        "votreadresse@", "votre.adresse@",
        "adresse.mail@", "adressemail@",
        # Arabic placeholders
        "بريد@", "بريدك@", "عنوان@", "عنوانك@"
    ]
    
    # Check for placeholder domains
    placeholder_domains = [
        "example.com", "exemple.com", "sample.com", "test.com",
        "domain.com", "domaine.com", "site.com", "website.com",
        "email.com", "mail.com", "yoursite.com", "votresite.com"
    ]
    
    # Check if email contains any placeholder pattern
    if any(p in email for p in placeholder_patterns):
        return False
    
    # Check domain
    domain = email.split('@')[-1]
    if any(d in domain for d in placeholder_domains):
        return False
    
    # Check for temporary/disposable email services
    temp_email_services = [
        "temp", "disposable", "throwaway", "tempmail",
        "10minutemail", "mailinator", "guerrillamail", "yopmail"
    ]
    if any(service in domain for service in temp_email_services):
        return False
    
    # Additional validation for common patterns
    if re.match(r'^[0-9]+@', email):  # Emails starting with numbers are often fake
        return False
    
    if len(re.findall(r'[0-9]', local_part)) > len(local_part) / 2:  # Too many numbers in local part
        return False
    
    return True

def has_valid_address(address):
    """
    Enhanced address validation with better checks for Tunisian addresses.
    Returns True if valid, False otherwise.
    """
    if not address or pd.isna(address):
        return False
    
    address = str(address).strip().lower()
    
    # Check if address has reasonable length
    if len(address) < 5:
        return False
        
    # Common Tunisian address keywords and patterns
    address_keywords = [
        # French terms
        'rue', 'avenue', 'boulevard', 'route', 'place', 'quartier',
        'résidence', 'immeuble', 'appartement', 'étage', 'bloc',
        'cité', 'zone', 'centre', 'complexe', 'lotissement',
        
        # Arabic transliterated terms
        'nahj', 'charaa', 'hay', 'madina', 'borj',
        
        # Common abbreviations
        'ave', 'blvd', 'rte', 'res', 'apt', 'ctr',
        'lot', 'bp', 'cp', 'km',
        
        # Numbers (with variations)
        'n°', 'numero', 'numéro', 'num',
        
        # Cities and regions
        'tunis', 'sfax', 'sousse', 'kairouan', 'bizerte',
        'gabes', 'ariana', 'gafsa', 'monastir', 'ben arous',
        'kasserine', 'medenine', 'nabeul', 'hammamet', 'tataouine',
        'beja', 'jendouba', 'siliana', 'zaghouan', 'kebili',
        'mahdia', 'sidi bouzid', 'tozeur', 'manouba'
    ]
    
    # Check for presence of address keywords
    if not any(keyword in address for keyword in address_keywords):
        return False
    
    # Check for number patterns (building numbers, postal codes, etc.)
    has_numbers = bool(re.search(r'\d', address))
    
    # Check for postal code pattern (common in Tunisia)
    has_postal_code = bool(re.search(r'\b\d{4}\b', address))
    
    # Scoring system
    score = 0
    
    # Basic checks
    if has_numbers:
        score += 1
    if has_postal_code:
        score += 2
    if len(address) > 20:  # Reasonable length for a complete address
        score += 1
    
    # Keyword checks
    keyword_count = sum(1 for keyword in address_keywords if keyword in address)
    score += min(keyword_count, 3)  # Cap at 3 to avoid over-counting
    
    # Check for common Tunisian address patterns
    if re.search(r'rue .{3,}', address):  # Street name pattern
        score += 1
    if re.search(r'(cp|code postal|bp)\s*\d{4}', address):  # Postal code pattern
        score += 1
    
    # Consider valid if score is high enough
    return score >= 3  # Adjust threshold as needed

def categorize_data(df):
    """
    Categorize the dataframe into three types:
    1. Raw data (all scraped data)
    2. Contact data (has valid phone or email)
    3. Location data (has valid phone and address)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (raw_df, contact_df, location_df)
    """
    if df.empty:
        logger.warning("Empty dataframe provided to categorize_data")
        return df, df, df
    
    # Create copies to avoid modifying the original
    raw_df = df.copy()
    contact_df = df.copy()
    location_df = df.copy()
    
    # Add validation columns
    contact_df['has_valid_phone'] = contact_df['phone'].apply(has_valid_phone)
    contact_df['has_valid_email'] = contact_df['email'].apply(has_valid_email)
    location_df['has_valid_phone'] = location_df['phone'].apply(has_valid_phone)
    location_df['has_valid_address'] = location_df['address'].apply(has_valid_address)
    
    # Filter contact data (has phone OR email)
    contact_df = contact_df[
        (contact_df['has_valid_phone']) | 
        (contact_df['has_valid_email'])
    ].drop(['has_valid_phone', 'has_valid_email'], axis=1).reset_index(drop=True)
    
    # Filter location data (has phone AND address)
    location_df = location_df[
        (location_df['has_valid_phone']) & 
        (location_df['has_valid_address'])
    ].drop(['has_valid_phone', 'has_valid_address'], axis=1).reset_index(drop=True)
    
    logger.info(f"Total rows: {len(raw_df)}")
    logger.info(f"Rows with contact info: {len(contact_df)}")
    logger.info(f"Rows with location info: {len(location_df)}")
    
    return raw_df, contact_df, location_df

def save_categorized_data(raw_df, contact_df, location_df, output_dir="output"):
    """
    Save the three categorized dataframes to separate CSV files.
    
    Args:
        raw_df (pd.DataFrame): Raw scraped data
        contact_df (pd.DataFrame): Data with contact information
        location_df (pd.DataFrame): Data with location information
        output_dir (str): Directory to save the files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each dataframe
    raw_df.to_csv(os.path.join(output_dir, "raw_data.csv"), index=False)
    contact_df.to_csv(os.path.join(output_dir, "contact_data.csv"), index=False)
    location_df.to_csv(os.path.join(output_dir, "location_data.csv"), index=False)
    
    logger.info(f"Saved data to {output_dir}/")
    logger.info(f"- raw_data.csv: {len(raw_df)} rows")
    logger.info(f"- contact_data.csv: {len(contact_df)} rows")
    logger.info(f"- location_data.csv: {len(location_df)} rows") 