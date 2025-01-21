import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the faculty page
BASE_URL = "https://pure.bit.edu.cn/zh/organisations/school-of-aerospace-engineering/persons/?page="

# Function to scrape data from one page
def scrape_page(page_number):
    url = f"{BASE_URL}{page_number}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page {page_number}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    faculty_data = []

    # Find all faculty entries on the page
    for person in soup.find_all('h3', class_='title'):
        name = person.find('span').text.strip() if person.find('span') else "N/A"
        print(name)
        profile_url = person.find('a', href=True)['href'] if person.find('a', href=True) else "N/A"
        # Fetch research interests from the profile page
        research_interests = get_research_interests(profile_url) if profile_url else "N/A"
        # fetch title
        title = get_title(profile_url) if profile_url else "N/A"

        faculty_data.append({
            "Name": name,
            "Title": title,
            "Profile URL": profile_url,
            "Research Interests": research_interests,
        })
    return faculty_data

# Function to scrape research interests from a faculty profile
def get_research_interests(profile_url):
    response = requests.get(profile_url)
    if response.status_code != 200:
        print(f"Failed to fetch profile: {profile_url}")
        return "N/A"

    soup = BeautifulSoup(response.text, 'html.parser')
    research_section = soup.find('h3', string="研究领域和方向")  # Header for research areas
    if research_section:
        text_block = research_section.find_next('div', class_='textblock')
        if text_block:
            return text_block.text.strip().replace("\n", " | ").replace(",", " |")
    return "N/A"

# Function to scrape title from a faculty profile
def get_title(profile_url):
    response = requests.get(profile_url)
    if response.status_code != 200:
        print(f"Failed to fetch profile: {profile_url}")
        return "N/A"

    soup = BeautifulSoup(response.text, 'html.parser')
    intro_section = soup.find('h3', string="个人简介")  # Header for personal intro section
    if intro_section:
        text_block = intro_section.find_next('div', class_='textblock')
        if text_block:
            # Look for the text after "职 称："
            title_info = text_block.text
            # Extract text after "职 称：" (it may have some leading or trailing spaces)
            title = title_info.split("职 称：")[-1].split("\n")[0].strip()
            return title
    return "N/A"
# Scrape data from all pages
all_faculty_data = []
for page in range(4):  # Adjust range based on the total number of pages
    all_faculty_data.extend(scrape_page(page))

# Convert data to a DataFrame
df = pd.DataFrame(all_faculty_data)

# Save the data to a CSV file
df.to_csv("faculty_data.csv", index=False, encoding='utf-8')
print("Data saved to faculty_data.csv")