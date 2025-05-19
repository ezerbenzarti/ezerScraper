#!/usr/bin/env python
import argparse
import json
import torch
import logging
import pandas as pd
import requests
import threading
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, send_file
from scraper import qa_model, crawler
from scraper.data_clean import categorize_data, save_categorized_data
from scraper.utils import save_results
from scraper.geocoder import geocode_locations_data
from scraper.cv_scraper import cv_crawl_site
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
geocoded_location_data = []

def update_global_data(result, workflow_id):
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
    
    # Save to workflow-specific CSV files
    output_dir = os.path.join("output", f"workflow_{workflow_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    raw_file = os.path.join(output_dir, "raw_data.csv")
    contact_file = os.path.join(output_dir, "contact_data.csv")
    location_file = os.path.join(output_dir, "location_data.csv")
    
    raw_df.to_csv(raw_file, index=False)
    contact_df.to_csv(contact_file, index=False)
    location_df.to_csv(location_file, index=False)
    
    logger.info(f"Saved data to {output_dir}/")
    logger.info(f"- raw_data.csv: {len(raw_df)} rows")
    logger.info(f"- contact_data.csv: {len(contact_df)} rows")
    logger.info(f"- location_data.csv: {len(location_df)} rows")
    
    return raw_file

def send_webhook(workflow_id, status, data=None, error=None):
    """Send webhook to Laravel with scraping results"""
    logger.info(f"Sending webhook for workflow {workflow_id}", {
        "status": status,
        "has_data": bool(data),
        "has_error": bool(error)
    })
    
    try:
        webhook_url = "http://127.0.0.1:8000/api/webhook/scraper-callback"
        webhook_data = {
            "workflow_id": workflow_id,
            "status": status,
            "data": data if data else {}
        }
        
        # Add number of rows and CSV file path if data exists
        if data and isinstance(data, dict):
            webhook_data["data"]["extracted_rows"] = len(data.get("raw_results", []))
            webhook_data["data"]["output_file"] = os.path.join("output", f"workflow_{workflow_id}", "raw_data.csv")
            
        if error:
            webhook_data["data"]["error"] = str(error)
            
        logger.debug(f"Webhook payload: {json.dumps(webhook_data, indent=2)}")
        
        response = requests.post(webhook_url, json=webhook_data)
        logger.info(f"Webhook response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to send webhook: {response.text}")
        else:
            logger.info(f"Successfully sent webhook for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"Error sending webhook: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")

def process_scraping(start_url, prompt, qa_pipe, crawl_detail, workflow_id, scraping_method='legacy'):
    """Process scraping in a separate thread"""
    logger.info(f"Starting scraping process for workflow {workflow_id}", {
        "url": start_url,
        "prompt": prompt,
        "crawl_detail": crawl_detail,
        "method": scraping_method
    })
    
    try:
        # Choose scraping method
        if scraping_method == 'computer_vision':
            logger.info(f"Using computer vision scraping for workflow {workflow_id}")
            result = cv_crawl_site(
                start_url=start_url,
                prompt=prompt,
                qa_pipe=qa_pipe,
                crawl_detail=crawl_detail
            )
        else:
            # Legacy scraping method
            logger.info(f"Using legacy scraping for workflow {workflow_id}")
            result = crawler.crawl_site(
                start_url=start_url,
                prompt=prompt,
                depth=1,
                max_pages=None,
                qa_pipe=qa_pipe,
                crawl_detail=crawl_detail
            )
        
        logger.info(f"Crawl completed for workflow {workflow_id}", {
            "result_count": len(result) if isinstance(result, list) else 1
        })
        
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
        
        logger.debug(f"Processed result for workflow {workflow_id}: {json.dumps(result, indent=2)}")
        
        # Update global data
        logger.info(f"Updating global data for workflow {workflow_id}")
        update_global_data(result, workflow_id)
        
        # Send success webhook
        logger.info(f"Sending success webhook for workflow {workflow_id}")
        send_webhook(
            workflow_id=workflow_id,
            status="COMPLETED",
            data={
                "raw_results": raw_data,
                "contact_results": contact_data,
                "location_results": geocoded_location_data
            }
        )
        
    except Exception as e:
        logger.error(f"Error during scraping for workflow {workflow_id}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        
        # Send error webhook
        logger.error(f"Sending error webhook for workflow {workflow_id}")
        send_webhook(
            workflow_id=workflow_id,
            status="FAILED",
            error=str(e)
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data_view():
    return render_template('data_view.html')

@app.route('/api/scrape', methods=['POST'])
def scrape():
    data = request.json
    workflow_id = data.get('workflow_id')
    scraping_method = data.get('scraping_method', 'legacy')  # Default to legacy method
    
    logger.info(f"Received scrape request for workflow {workflow_id}", {
        "url": data.get('url'),
        "has_prompt": bool(data.get('prompt')),
        "crawl_detail": data.get('crawl_detail'),
        "method": scraping_method
    })
    
    try:
        # Load QA model
        logger.info(f"Loading QA model for workflow {workflow_id}")
        qa_pipe = qa_model.load_model()
        
        # Start scraping in a separate thread
        thread = threading.Thread(
            target=process_scraping,
            args=(
                data['url'],
                data['prompt'],
                qa_pipe,
                data.get('crawl_detail', False),
                workflow_id,
                scraping_method
            )
        )
        thread.start()
        
        return jsonify({
            "status": "started",
            "message": f"Scraping started with {scraping_method} method",
            "workflow_id": workflow_id
        })
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint for workflow {workflow_id}: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
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

@app.route('/api/data/raw/csv/<workflow_id>', methods=['GET'])
def get_raw_data_csv(workflow_id):
    """Download the raw data CSV file for a specific workflow"""
    try:
        csv_path = os.path.join("output", f"workflow_{workflow_id}", "raw_data.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "CSV file not found"}), 404
            
        return send_file(
            csv_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'raw_data_workflow_{workflow_id}.csv'
        )
    except Exception as e:
        logger.error(f"Error downloading CSV file: {e}")
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
        return jsonify(geocoded_location_data)
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