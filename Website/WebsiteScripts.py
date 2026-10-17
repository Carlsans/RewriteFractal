import io
import os
from argparse import Namespace

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
#import opentimestamps
import otsclient.cmds

class WebsiteScripts:


    def fetch_and_timestamp_images(self,webpage_url, extensions=['.jpg', '.png', '.gif', '.jpeg']):
        # Create directories for images and timestamps
        os.makedirs('downloaded_images', exist_ok=True)
        os.makedirs('ots_timestamps', exist_ok=True)

        # Fetch the webpage
        try:
            response = requests.get(webpage_url)
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching webpage: {e}")
            return

        # Find all image tags
        images = soup.find_all('img')



        # Process images
        for img in images:
            # Get image source
            img_src = img.get('src')
            if not img_src:
                continue

            # Convert relative URLs to absolute
            full_img_url = urljoin(webpage_url, img_src)

            # Parse the URL to get filename
            parsed_url = urlparse(full_img_url)
            filename = os.path.basename(parsed_url.path)

            # Check file extension
            if any(filename.lower().endswith(ext) for ext in extensions):
                try:
                    # Download image
                    img_response = requests.get(full_img_url)

                    # Save image
                    local_filename = os.path.join('downloaded_images', filename)
                    with open(local_filename, 'wb') as f:
                        f.write(img_response.content)

                    # Create OTS timestamp
                    ots_filename = os.path.join('ots_timestamps', f'{filename}.ots')
                    #ots_client.stamp(local_filename)
                    file_objects = []
                    file_obj = open(local_filename, 'rb')
                    f = io.open(local_filename, mode="r", encoding="utf-8")
                    args = {}
                    args['files'] = file_obj
                    arguments = Namespace(files=f)
                    otsclient.cmds.stamp_command(arguments)

                    print(f"Timestamped: {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
webscript = WebsiteScripts()
webscript.fetch_and_timestamp_images('http://moduloinfo.ca/2024/11/26/rewriting-system/')