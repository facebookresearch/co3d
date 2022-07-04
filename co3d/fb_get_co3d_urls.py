"""
copy-paste from https://github.com/facebookresearch/co3d/issues/8#issuecomment-914503533
"""

import argparse
import time
import requests
import sys
import selenium.webdriver.firefox.options
from selenium import webdriver


CO3D_WEBPAEGE_URL = "https://ai.facebook.com/datasets/co3d-downloads/"
GECKO_DRIVER_PATH = "/private/home/dnovotny/bin/geckodriver/geckodriver"

# sys.path.insert(0, GECKO_DRIVER_DIR)


def fetch_url_by_span_text(driver, query_text):
    text_elem = driver.find_element_by_xpath("//span[contains(text(),'{}')]".format(query_text))
    a_elm = text_elem.find_element_by_xpath("..")
    url = a_elm.get_attribute("href")
    return url


def get_category_ids(driver):
    cur_list_url = fetch_url_by_span_text(driver, "Download all links")
    response = requests.get(cur_list_url)
    data = response.text
    lines = data.split('\n')[1:]
    category_ids = [elm.split()[0].strip() for elm in lines]
    return category_ids


def get_co3d_urls(page_path):
    options = selenium.webdriver.firefox.options.Options()
    options.headless = True
    firefox_profile = webdriver.FirefoxProfile()
    firefox_profile.set_preference("browser.privatebrowsing.autostart", True)
    with webdriver.Firefox(options=options, firefox_profile=firefox_profile, executable_path=GECKO_DRIVER_PATH) as driver:
        driver.get(page_path)
        time.sleep(1)  # Some delay to let the webpage populate
        category_ids = get_category_ids(driver)
        item_path_pairs = []
        for category_id in category_ids:
            url = fetch_url_by_span_text(driver, category_id)
            item_path_pairs.append((category_id, url))
        return item_path_pairs


def dump_default_co3d_urls(download_files_list):
    co3d_item_urls = get_co3d_urls(CO3D_WEBPAEGE_URL)
    with open(download_files_list, 'w') as f_out:
        f_out.write("file_name\tcdn_link\n")
        for i, (item, url) in enumerate(co3d_item_urls):
            f_out.write(item)
            f_out.write('\t')
            f_out.write(url)
            if i < len(co3d_item_urls) - 1:
                f_out.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_files_list", type=str, required=False, help="Where the downloadable list will be generated", default="./downloadpaths.txt")
    args = parser.parse_args()
    co3d_item_urls = get_co3d_urls(CO3D_WEBPAEGE_URL)
    with open(args.download_files_list, 'w') as f_out:
        f_out.write("file_name\tcdn_link\n")
        for i, (item, url) in enumerate(co3d_item_urls):
            f_out.write(item)
            f_out.write('\t')
            f_out.write(url)
            if i < len(co3d_item_urls) - 1:
                f_out.write('\n')

# /private/home/dnovotny/bin/geckodriver --websocket-port 56291 --port 56133