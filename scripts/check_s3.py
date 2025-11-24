"""import code from the slides_extractor package

and verify things work

just use a different bucket name `tests_video` instead of `video`
"""

import os
import sys
from datetime import datetime

import boto3
import cv2
import numpy as np
import requests
from botocore.client import Config

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from slides_extractor.settings import S3_ACCESS_KEY, S3_BUCKET_NAME, S3_ENDPOINT
from slides_extractor.video_service import upload_to_s3


def main() -> None:
    print("Checking S3 configuration...")
    print(f"Endpoint: {S3_ENDPOINT}")
    print(f"Bucket: {S3_BUCKET_NAME}")
    masked_key = (
        f"{S3_ACCESS_KEY[:4]}...{S3_ACCESS_KEY[-4:]}" if S3_ACCESS_KEY else "None"
    )
    print(f"Access Key: {masked_key}")

    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        print("Error: S3_ENDPOINT and S3_ACCESS_KEY must be set.")
        sys.exit(1)

    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # Red
    success, buffer = cv2.imencode(".png", img)
    if not success:
        print("Error: Failed to encode image.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"tests_video/check_{timestamp}.png"

    print(f"\nUploading to key: {key}")

    try:
        public_url = upload_to_s3(
            data=buffer.tobytes(),
            key=key,
            content_type="image/png",
            metadata={"test": "true", "timestamp": timestamp},
        )
        print("Upload successful!")
        print(f"Public URL: {public_url}")

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )

        # Verify download via public URL
        print(f"\nVerifying public access at {public_url}...")
        r = requests.head(public_url)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            print("SUCCESS: Public URL works.")
        else:
            print("FAIL: Public URL failed (likely permissions issue).")

        # Generate presigned URL as backup check
        presigned_url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET_NAME, "Key": key}, ExpiresIn=3600
        )
        print(f"\nPresigned URL: {presigned_url}")

        r_pre = requests.get(presigned_url)
        print(f"Presigned Access Status: {r_pre.status_code}")

        if r_pre.status_code == 200:
            print("SUCCESS: Presigned URL works.")
        else:
            print("FAIL: Presigned URL failed.")

    except Exception as e:
        print(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
