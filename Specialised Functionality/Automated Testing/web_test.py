from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def visit_report():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")
    client = webdriver.Chrome(chrome_options=chrome_options,
        executable_path=r"/run/media/hacker/Windows/Users/zunmu/Documents/Stuff/Linux Tools/chromedriver")
		#executable_path=r"D:\\chromedriver.exe")
    client.set_page_load_timeout(10)
    client.set_script_timeout(10)

    client.get('https://www.w3schools.com/html/html_forms.asp')
    time.sleep(3)
    ### Fill up text fields
    client.find_element_by_id("fname").send_keys("Data")
    client.find_element_by_id("lname").click()
    ### Traversing the DOM
    button = (
        client.find_element_by_id("fname")
        .find_element(by=By.XPATH,value='..')
        .find_elements_by_xpath(".//*")[-1]
    )
    button.click()
    time.sleep(300)
    client.quit()

visit_report()
