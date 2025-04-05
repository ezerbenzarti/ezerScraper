#!/usr/bin/env python
import argparse
import json
from scraper import qa_model, crawler

def main():
    parser = argparse.ArgumentParser(description="Prompt-Guided Multimodal Web Scraper")
    parser.add_argument("url", help="Starting URL to scrape")
    parser.add_argument("prompt", help='Prompt like "Extract all names, phone, addresses, emails"')
    parser.add_argument("--depth", type=int, default=0, help="Depth of internal link crawling (default: 0)")
    parser.add_argument("--max_pages", type=int, default=10, help="Max pages for detail crawling (default: 10)")
    parser.add_argument("--output", default="output.json", help="Output JSON filename (default: output.json)")
    parser.add_argument("--crawl_detail", action="store_true", help="Follow candidate links one level deep")
    args = parser.parse_args()

    print("Loading QA model...")
    qa_pipe = qa_model.load_model()
    print("Starting crawl...")

    # depth not heavily used in the code snippet above, but you can pass it if you want
    result = crawler.crawl_site(
        start_url=args.url,
        prompt=args.prompt,
        depth=args.depth,
        max_pages=args.max_pages,
        qa_pipe=qa_pipe,
        crawl_detail=args.crawl_detail
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if isinstance(result, list):
        print(f"Extracted {len(result)} items. Output -> {args.output}")
    else:
        print("No results extracted or single-answer mode not implemented in detail here.")

if __name__ == "__main__":
    main()