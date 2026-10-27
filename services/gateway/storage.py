"""Write uploads to S3-compatible object storage (SeaweedFS locally, S3 in a cloud).

boto3 is synchronous, so callers in async code run these methods in a thread.
"""

import os
import time


class S3Store:
    def __init__(self):
        import boto3

        self.bucket = os.getenv("S3_BUCKET", "claimlens")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),  # None means AWS S3
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )

    def put(self, key, data, content_type):
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    def ensure_bucket(self, attempts=10, delay_s=1.0):
        """Create the bucket if it's missing. Retries because storage may still
        be starting when the gateway does."""
        from botocore.exceptions import ClientError, EndpointConnectionError

        for attempt in range(attempts):
            try:
                self.client.head_bucket(Bucket=self.bucket)
                return
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    self.client.create_bucket(Bucket=self.bucket)
                    return
                raise
            except EndpointConnectionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay_s)
