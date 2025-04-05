from urllib.parse import urlparse, urljoin

def get_domain(url):
    """
    Returns the domain (netloc) of the URL.
    """
    parsed = urlparse(url)
    return parsed.netloc

def is_internal(start_url, new_url):
    """
    Checks whether new_url is in the same domain as start_url.
    """
    return get_domain(start_url) == get_domain(new_url)
