import os
from datetime import datetime

import boto3
import numpy as np
import pytest
import requests
import cv2
from botocore.client import Config

from slides_extractor.settings import S3_ACCESS_KEY, S3_BUCKET_NAME, S3_ENDPOINT
from slides_extractor.video_service import upload_to_s3


@pytest.mark.integration
def test_s3_upload_flow():
    """Integration test for S3 upload flow.

    Verifies:
    1. Upload works (and bypasses corruption issues using requests)
    2. Authenticated download works
    3. File integrity (exact bytes match)
    4. Public access fails
    5. Presigned URL works
    """
    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        pytest.skip("S3 configuration not available, skipping integration test")

    # Type narrowing for static analysis
    assert S3_ENDPOINT is not None
    assert S3_ACCESS_KEY is not None

    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # Red
    success, buffer = cv2.imencode(".png", img)
    assert success, "Failed to encode image"
    expected_bytes = buffer.tobytes()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"tests_video/integration_test_{timestamp}.png"

    # 1. Test Upload
    s3_uri = upload_to_s3(
        data=expected_bytes,
        key=key,
        content_type="image/png",
        metadata={"test": "true", "timestamp": timestamp},
    )
    assert s3_uri.startswith(f"s3://{S3_BUCKET_NAME}")
    assert key in s3_uri

    # Setup boto3 client for verification
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    # 2. Verify authenticated download
    response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    content = response["Body"].read()

    # 3. Verify Integrity
    # We now expect exact byte match because we fixed the chunked encoding corruption
    # by using requests + presigned URL for upload.
    assert len(content) == len(expected_bytes), (
        f"Size mismatch: got {len(content)}, expected {len(expected_bytes)}"
    )
    assert content == expected_bytes, "Content mismatch"

    # Optional: Save file to verify content (and cleanup)
    tmp_path = f"/tmp/integration_test_{timestamp}.png"
    with open(tmp_path, "wb") as f:
        f.write(content)
    assert os.path.exists(tmp_path)
    os.remove(tmp_path)

    # 4. Verify public access is denied
    public_http_url = f"{S3_ENDPOINT.rstrip('/')}/{S3_BUCKET_NAME}/{key}"
    r = requests.head(public_http_url)
    assert r.status_code != 200, "Public URL should not be accessible"

    # 5. Verify presigned URL access
    presigned_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET_NAME, "Key": key}, ExpiresIn=60
    )
    r_pre = requests.get(presigned_url)
    assert r_pre.status_code == 200, "Presigned URL should be accessible"
    assert r_pre.content == expected_bytes, "Presigned URL content mismatch"

    # Cleanup S3 object
    s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
