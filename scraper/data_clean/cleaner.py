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
    Check if the email is valid (not empty and has @ and .).
    Returns True if valid, False otherwise.
    """
    if not email or pd.isna(email):
        return False
    
    email = str(email).strip().lower()
    
    # Basic email validation
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
    
    # Check for placeholder emails
    placeholder_patterns = [
        "example", "exemple", "sample", "test", "demo",
        "your.email", "your-email", "your_email",
        "email@", "mail@", "contact@",
        "info@", "support@", "admin@",
        "user@", "username@", "name@",
        "someone@", "someone@example.com"
    ]
    
    return not any(pattern in email for pattern in placeholder_patterns)

def has_valid_address(address):
    """
    Check if the address is valid (not empty and has some content).
    Returns True if valid, False otherwise.
    """
    if not address or pd.isna(address):
        return False
    
    address = str(address).strip()
    
    # Check if address has reasonable length and content
    if len(address) < 10:
        return False
    
    # Check for common address keywords
    address_keywords = [
        "rue", "avenue", "bp", "quartier", "route", "lot", "zone", "imm",
        "street", "road", "blvd", "boulevard", "zip", "postal", "cedex"
    ]
    
    return any(keyword in address.lower() for keyword in address_keywords)

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