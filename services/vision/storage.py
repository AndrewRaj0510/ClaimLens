"""Read images from S3-compatible object storage (SeaweedFS locally, S3 in a cloud).

boto3 speaks the S3 API, so switching backends is a change of S3_ENDPOINT only.
"""

import os


class ObjectNotFound(KeyError):
    pass


class S3Store:
    def __init__(self, bucket=None, endpoint=None):
        import boto3

        self.bucket = bucket or os.getenv("S3_BUCKET", "claimlens")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint or os.getenv("S3_ENDPOINT"),  # None means AWS S3
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )

    def get(self, key, max_bytes):
        """Read at most max_bytes + 1, so an oversized object can be detected
        without pulling the whole thing into memory."""
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey as e:
            raise ObjectNotFound(key) from e
        with obj["Body"] as body:
            return body.read(max_bytes + 1)
