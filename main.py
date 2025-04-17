#!/usr/bin/env python
import argparse
import json
import torch
import logging
import pandas as pd
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from scraper import qa_model, crawler
from scraper.data_clean import categorize_data, save_categorized_data
from scraper.utils import save_results
from scraper.geocoder import geocode_locations_data
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store the data
raw_data = []
contact_data = []
location_data = []
geocoded_location_data = []  # New global variable for geocoded locations

def update_global_data(result):
    """Update the global data variables with new scraped data"""
    global raw_data, contact_data, location_data, geocoded_location_data
    
    # Convert result to DataFrame
    df = pd.DataFrame(result)
    
    # Categorize the data
    raw_df, contact_df, location_df = categorize_data(df)
    
    # Update global variables
    raw_data = raw_df.to_dict(orient='records')
    contact_data = contact_df.to_dict(orient='records')
    location_data = location_df.to_dict(orient='records')
    
    # Geocode the location data
    geocoded_location_data = geocode_locations_data(location_data)
    
    # Also save to CSV files for backup
    save_categorized_data(raw_df, contact_df, location_df, output_dir="output")

def generate_scraping_response(start_url, prompt, qa_pipe, crawl_detail):
    try:
        # Initialize progress
        yield json.dumps({"progress": 0, "message": "Starting scraping process..."}) + "\n"
        
        # Start crawling
        result = crawler.crawl_site(
            start_url=start_url,
            prompt=prompt,
            depth=1,  # Fixed depth of 1
            max_pages=None,  # No limit on pages
            qa_pipe=qa_pipe,
            crawl_detail=crawl_detail
        )
        
        # Ensure result is a list of dictionaries with required fields
        if not isinstance(result, list):
            result = [result] if result else []
            
        # Add missing fields to each item
        for item in result:
            if not isinstance(item, dict):
                continue
            for field in ['name', 'phone', 'email', 'address', 'domain', 'poste']:
                if field not in item:
                    item[field] = ''
        
        # Update global data
        update_global_data(result)
        
        # Update progress
        yield json.dumps({"progress": 100, "message": "Scraping completed!"}) + "\n"
        
        # Send final result
        yield json.dumps({"result": result}) + "\n"
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        yield json.dumps({"error": str(e)}) + "\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data_view():
    return render_template('data_view.html')

@app.route('/api/scrape', methods=['POST'])
def scrape():
    data = request.json
    
    try:
        # Load QA model
        qa_pipe = qa_model.load_model()
        
        return Response(
            stream_with_context(
                generate_scraping_response(
                    start_url=data['url'],
                    prompt=data['prompt'],
                    qa_pipe=qa_pipe,
                    crawl_detail=data['crawl_detail']
                )
            ),
            mimetype='text/event-stream'
        )
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}")
        return Response(
            json.dumps({"error": str(e)}),
            status=500,
            mimetype='application/json'
        )

@app.route('/api/data/raw', methods=['GET'])
def get_raw_data():
    """Get all raw scraped data"""
    try:
        return jsonify(raw_data)
    except Exception as e:
        logger.error(f"Error getting raw data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/contact', methods=['GET'])
def get_contact_data():
    """Get data with valid contact information"""
    try:
        return jsonify(contact_data)
    except Exception as e:
        logger.error(f"Error getting contact data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/location', methods=['GET'])
def get_location_data():
    """Get data with valid location information"""
    try:
        return jsonify(geocoded_location_data)  # Return geocoded data instead
    except Exception as e:
        logger.error(f"Error getting location data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/stats', methods=['GET'])
def get_data_stats():
    """Get statistics about the available data"""
    try:
        stats = {
            "raw": len(raw_data),
            "contact": len(contact_data),
            "location": len(location_data)
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Using CPU.")
    
    parser = argparse.ArgumentParser(description="Prompt-Guided Multimodal Web Scraper")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()

    try:
        app.run(host=args.host, port=args.port, debug=True)
    except OSError as e:
        if "address already in use" in str(e):
            logger.error(f"Port {args.port} is already in use. Please try a different port using --port argument.")
            logger.info("Try running with: python main.py --port 8081")
        else:
            logger.error(f"Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()