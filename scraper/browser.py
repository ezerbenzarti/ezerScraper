from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def init_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--ignore-certificate-errors")  # Ignore SSL errors
    # Optionally, set a window size to improve screenshot quality:
    options.add_argument("--window-size=1280,2000")
    driver = webdriver.Chrome(options=options)
    return driver

def render_page(url, timeout=10):
    """
    Renders a webpage using Selenium and returns a tuple:
    (html, visible_text, screenshot_path)
    
    - html: the page source.
    - visible_text: text from the <body> element.
    - screenshot_path: path to the saved screenshot (always 'screenshot.png' here).
    
    In case of errors, it returns empty strings.
    """
    driver = init_driver()
    driver.set_page_load_timeout(timeout)
    try:
        driver.get(url)
    except Exception as e:
        print(f"Error loading {url}: {e}")
        driver.quit()
        return "", "", ""
    
    # Wait for dynamic content to load.
    time.sleep(3)
    
    html = driver.page_source
    try:
        visible_text = driver.find_element("tag name", "body").text
    except Exception:
        visible_text = html
    screenshot_path = "screenshot.png"
    try:
        driver.save_screenshot(screenshot_path)
    except Exception as e:
        print(f"Error saving screenshot for {url}: {e}")
        screenshot_path = ""
    driver.quit()
    return html, visible_text, screenshot_path
