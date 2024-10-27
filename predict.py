import torch
from flask import Flask,  jsonify
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import pandas as pd
import re
from bs4 import BeautifulSoup
import tldextract
import requests

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # 512 Nodes on 1 hidden Layer
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("phishingDetection/model_weights.pth", weights_only=True), strict=False)
model.eval()

# Code Taken from HTML_extraction.ipynb

url_regex = r'https?://(?:www\.)?[^\s/$.?#].[^\s]*'
href_regex = r'href=["\'](https?://[^\s"\'<>]+)["\']'
ip_regex = r'(https?://)?\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#Create the function helper for the HTML extraction

#1) Check if there is a @ in the url (Bool)
def has_at_in_urls(urls):
    email_pattern = re.compile(email_regex, re.IGNORECASE)
    #Store all the email pattern to detect if it is a emal addresses
    for url in urls:
        if "@" in url and not email_pattern.search(url):
            return True
    return False


def number_attachments(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Initialize the count of attachments
    total_attachments = 0

    # Count <a> tags that link to downloadable files (including images if needed)
    total_attachments += len(soup.find_all('a', href=lambda href: href and href.endswith(('.pdf', '.doc', '.ppt', '.zip', '.txt', '.jpg', '.png', '.gif'))))

    # Count <embed> tags (if they link to downloadable content)
    total_attachments += len(soup.find_all('embed', src=lambda src: src and src.endswith(('.pdf', '.doc', '.ppt', '.zip', '.txt', '.jpg', '.png', '.gif'))))

    # Count <object> tags (commonly used for embedding downloadable multimedia)
    total_attachments += len(soup.find_all('object', data=lambda data: data and data.endswith(('.pdf', '.doc', '.ppt', '.zip', '.txt', '.jpg', '.png', '.gif'))))

    return total_attachments
#Used the find_all to find all the tags relates to attachments like: <a>, <img>, <embed>, <iframe>, <object>

#3) Check CSS in header (value)
def count_css_links(html):
    soup = BeautifulSoup(html, "html.parser")
    link_count = len(soup.find_all("link", rel="stylesheet"))
    style_count = len(soup.find_all("style"))
    return link_count + style_count

#Used the parsed HTML to find all the <link> tag set to rel=stylesheet or <style> 


#4) Check for external resources (value)
def count_external_resources(html):
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all(src=True)) + len(soup.find_all("link", href=True))
#Find all the src tag or link tag with href to count the number of external resources


#5) Check for HTML content (bool)
def html_content_str(html):
    return bool(re.search(r'html', html, re.IGNORECASE))
#r'html' is the expression pattern for the string <html will look into it and re.search will scans through the string html looking for a match
#the ignorecase is to make the search case-insensitive

#6) Check for HTML form (bool)
def html_form(html):
    return bool(re.search(r'<\s?\/?\s?form\s?>', html, re.IGNORECASE))
#searching for the form tag <form> by looking for the opening <, \s?(optional spaces), |/? to allow for </form> and closing angle>

#7) Check for iframe Form (bool)
def iframe(html):
    return bool(re.search(r'<\s?\/?\s?iframe\s?>', html, re.IGNORECASE))
#same type of searching as the form but looking for <iframe> and </iframe>

#8) Check for IPs in URLS (bool)
def ips_in_urls(urls):
    for url in urls:
        if re.search(ip_regex, url):
            return True
    return False
#loop into the urls list if found in the url a pattern defined by ip_regenex then return True

#9) Check for Javascript Block (value)
def count_javascript_blocks(html):
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("script"))
#Using the search to find all the <script> tags and counts them

#10) Check for URLs in the email
def extract_urls_from_html(html):
    return re.findall(url_regex, html)
    
#re.findall searches the entire HTML string to find the pattern: url_regex that contains all the patern for matching URLs
#re is tool for defining complex patterns fro string matching (works a plain text) versus beautifulSoup that works as a DOM-like parser that treats HTML as a structured documents

def process_file(h):
    html = "<html>\n" + h + "\n</html>"

    # Extract URLs from HTML using regex
    urls = extract_urls_from_html(html)

    # Initialize feature dictionary
    feature_dict = {
        '@ in URLs': has_at_in_urls(urls),
        'Attachments': number_attachments(html),
        'Css': count_css_links(html),
        'External Resources': count_external_resources(html),
        'HTML content': html_content_str(html),
        'Html Form': html_form(html),
        'Html iFrame': iframe(html),
        'IPs in URLs': ips_in_urls(urls),
        'Javascript': count_javascript_blocks(html),
        'URLs': len(urls)
    }

    # Convert to DataFrame
    df = pd.DataFrame(columns=feature_dict.keys())  # Create empty DataFrame with columns as feature names
    df.loc[0] = feature_dict.values()  # Set the first row with values

    # Create an empty DataFrame with feature names as columns and populate the first row with values from the feature dictionary.
    return df

# End of Code from HTML_extraction.ipynb

app = Flask(__name__)
@app.route('/send-string', methods=['POST'])
def index():
    return jsonify({ "prediction": max_probability })
if __name__ == '__main__':
    app.run(port=5000, debug=True)

emailContents = ""
def fetch_data():
    input = requests.get('http://localhost:5000/send-string')
    if input.status_code == 200:
        data = input.json()
        return data
if  __name__ == '__main__':
    emailContents = fetch_data()
    

    

row = process_file(emailContents)

atInUrl =  1 if row['@ in URLs'].iloc[0] else 0
Attachments = row['Attachments'].iloc[0]
Css = row['Css']
Ext = row['External Resources']
htmlcont = 1 if row['HTML content'].iloc[0] else 0
htmlform = 1 if row['Html Form'].iloc[0] else 0
htmliframe = 1 if row['Html iFrame'].iloc[0] else 0
ip = 1 if row['IPs in URLs'].iloc[0] else 0
js = row['Javascript']
urls = row['URLs']
features = [atInUrl,Attachments,Css,Ext,htmlcont,htmlform,htmliframe,ip,js,urls]
features_tensor = torch.tensor(features, dtype=torch.float32)

new_data = TensorDataset(features_tensor.unsqueeze(0))
new_dataloader = DataLoader(new_data, batch_size = 1)


with torch.no_grad():
    for batch in new_dataloader:
        X = batch[0]
        prediction = model(X)
        prob = nn.functional.softmax(prediction, dim=1)
        max_index = torch.argmax(prob, dim=1)
        max_probability = prediction[0,max_index].item()

# print(prediction.argmax(1))
    

