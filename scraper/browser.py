from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_driver():
    options = Options()
    options.add_argument("--headless=new")  # Use new headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    return driver

def render_page(url, timeout=60):
    """
    Renders a webpage using Selenium and returns a tuple:
    (html, visible_text, screenshot_path)
    """
    driver = None
    try:
        driver = init_driver()
        driver.set_page_load_timeout(timeout)
        
        logger.info(f"Loading page: {url}")
        driver.get(url)
        
        # Wait for the page to be interactive
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        
        # Wait for body to be present and visible
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        time.sleep(3)
        
        # Scroll to load dynamic content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Get page source and visible text
        html = driver.page_source
        try:
            visible_text = driver.find_element(By.TAG_NAME, "body").text
        except NoSuchElementException:
            logger.warning("Could not find body element, using page source as text")
            visible_text = html
            
        # Take screenshot
        screenshot_path = "screenshot.png"
        try:
            driver.save_screenshot(screenshot_path)
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            screenshot_path = ""
            
        logger.info("Page loaded successfully")
        return html, visible_text, screenshot_path
        
    except TimeoutException:
        logger.error(f"Timeout while loading {url}")
        return "", "", ""
    except WebDriverException as e:
        logger.error(f"WebDriver error while loading {url}: {e}")
        return "", "", ""
    except Exception as e:
        logger.error(f"Unexpected error while loading {url}: {e}")
        return "", "", ""
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
