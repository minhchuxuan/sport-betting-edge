"""
Download historical football match data from Football-Data.co.uk
Simple script - downloads 5 seasons of Premier League data
"""

import requests
from pathlib import Path

def download_football_data():
    """Download Premier League data for last 5 seasons"""
    seasons = ["2324", "2223", "2122", "2021", "1920"]
    base_url = "https://www.football-data.co.uk/mmz4281"
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Premier League data...")
    
    for season in seasons:
        url = f"{base_url}/{season}/E0.csv"
        output_file = output_dir / f"E0_{season}.csv"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"[OK] Season {season}: {output_file.name}")
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Season {season}: {e}")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    download_football_data()
