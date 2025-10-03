import asf_search as asf
import os
from netrc import netrc, NetrcParseError
from tqdm import tqdm
import requests
from pathlib import Path

### USER SETTINGS ###

# 1. Define your Area of Interest using a Well-Known Text (WKT) string.
#    You can get this easily from https://wktmap.com/ by drawing a box.
#    This example covers a large part of the Tien Shan mountains in Kyrgyzstan.
wkt_aoi = 'POLYGON ((74.487648 42.479187, 74.473057 42.475769, 74.474258 42.463613, 74.487648 42.436127, 74.510994 42.433846, 74.523869 42.448035, 74.503269 42.467032, 74.487648 42.479187))'

# 2. Define the start and end dates for your entire time series.
start_date = '2017-01-01'
end_date = '2020-12-31'

# 3. Define the months you are interested in (e.g., end of melt season).
#    This will find images only from August and September for every year.
start_month = 8
end_month = 9

# 4. Choose the satellite pass direction (stick to one!).
#    Note: Constants live under FLIGHT_DIRECTION
pass_direction = asf.FLIGHT_DIRECTION.ASCENDING  # or asf.FLIGHT_DIRECTION.DESCENDING

# 5. Set the folder where you want to save the data.
download_folder = 'kyrgyzstan_glacier_data'

### END OF USER SETTINGS ###


def build_session_from_netrc():
    """
    Try to authenticate to Earthdata using ~/.netrc and return an ASFSession.
    Returns None if credentials are unavailable.
    """
    try:
        creds = None
        try:
            creds = netrc().authenticators('urs.earthdata.nasa.gov')
        except (FileNotFoundError, NetrcParseError):
            creds = None
        if creds is None:
            return None
        login, account, password = creds
        session = asf.ASFSession().auth_with_creds(login, password)
        return session
    except Exception as e:
        print("Could not authenticate with Earthdata using ~/.netrc.")
        print("Reason:", e)
        return None


def download_file_with_progress(url, output_path, session):
    """
    Download a single file with progress bar.
    """
    try:
        # Use the underlying requests session from ASFSession
        # ASFSession wraps a requests session, so we need to access it properly
        response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
            desc=Path(output_path).name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"\nError downloading {Path(output_path).name}: {e}")
        return False


def download_results_with_progress(results, download_folder, session):
    """
    Download all results with progress tracking.
    """
    Path(download_folder).mkdir(parents=True, exist_ok=True)

    # Prepare download list
    downloads = []
    for r in results:
        try:
            url = r.properties.get('url')
            scene_name = r.properties.get('sceneName', 'unknown')
            if url and scene_name:
                filename = f"{scene_name}.zip"
                output_path = Path(download_folder) / filename

                # Skip if already downloaded
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    if file_size > 0:
                        continue

                downloads.append((url, output_path))
        except Exception:
            continue

    if not downloads:
        print("\nAll files already downloaded or no valid URLs found.")
        return

    print(f"\n{len(downloads)} file(s) to download...")
    print(f"Already downloaded: {len(results) - len(downloads)} file(s)\n")

    # Download with overall progress
    successful = 0
    failed = 0

    for url, output_path in tqdm(downloads, desc="Overall progress", unit="file"):
        if download_file_with_progress(url, output_path, session):
            successful += 1
        else:
            failed += 1

    print(f"\n✓ Successfully downloaded: {successful} file(s)")
    if failed > 0:
        print(f"✗ Failed: {failed} file(s)")


def main():
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    print("Searching for Sentinel-1 data...")

    # Base search kwargs
    base_kwargs = dict(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD, asf.PRODUCT_TYPE.GRD_MS],  # Exclude OPERA products
        beamMode=asf.BEAMMODE.IW,
        flightDirection=pass_direction,
        intersectsWith=wkt_aoi,
        start=start_date,
        end=end_date,
        season=(start_month, end_month),  # Seasonal filter (Aug–Sep)
    )

    # Perform the search using your criteria
    results = []
    try:
        search_results = asf.search(**base_kwargs)
        # Convert generator to list more safely
        for r in search_results:
            try:
                results.append(r)
            except Exception:
                continue
    except asf.ASFSearchError as e:
        print(f"Search failed with ASFSearchError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during search: {e}")

    print(f"Found {len(results)} images matching your criteria.")

    # Fallback 1: remove season filter if no results
    if len(results) == 0:
        print("No results with seasonal filter. Retrying without season...")
        try:
            kwargs2 = base_kwargs.copy()
            kwargs2.pop('season', None)
            search_results = asf.search(**kwargs2)
            results = []
            for r in search_results:
                try:
                    results.append(r)
                except Exception:
                    continue
            print(f"Found {len(results)} images after removing season filter.")
        except asf.ASFSearchError as e:
            print(f"Retry without season failed with ASFSearchError: {e}")
            results = []
        except Exception as e:
            print(f"An unexpected error occurred during retry without season: {e}")
            results = []

    # Fallback 2: also remove flightDirection if still no results
    if len(results) == 0:
        print("Still no results. Retrying without flight direction constraint...")
        try:
            kwargs3 = base_kwargs.copy()
            kwargs3.pop('season', None)
            kwargs3.pop('flightDirection', None)
            search_results = asf.search(**kwargs3)
            results = []
            for r in search_results:
                try:
                    results.append(r)
                except Exception:
                    continue
            print(f"Found {len(results)} images after removing season and flight direction.")
        except asf.ASFSearchError as e:
            print(f"Retry without flight direction failed with ASFSearchError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during retry without flight direction: {e}")

    # Preview first few scene IDs for feedback
    if not results:
        print("No results found after all attempts.")
        return

    # Try to preview scene names with better error handling
    print("\nSample results:")
    for i, r in enumerate(results[:5]):
        try:
            scene_name = r.properties.get('sceneName', 'Unknown')
            print(f" - {scene_name}")
        except Exception as e:
            print(f" - Result {i+1} (unable to get scene name)")

    if len(results) > 0:
        # Authenticate with Earthdata using ~/.netrc if available
        session = build_session_from_netrc()
        if session is None:
            print("\nNo Earthdata credentials found in ~/.netrc; skipping download.")
            print("Create a ~/.netrc with your Earthdata credentials to enable downloads.")
        else:
            # Download all the found images with progress bars
            download_results_with_progress(results, download_folder, session)
    else:
        print("\nNo results found. Try adjusting your AOI, date range, or filters.")


if __name__ == '__main__':
    main()
