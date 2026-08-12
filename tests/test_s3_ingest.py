from pathlib import Path

from cluster.s3_ingest import S3Ingestor
from cluster.store import ClusterStore


class NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class FakePaginator:
    def paginate(self, **_):
        return [{"Contents": [
            {"Key": "incoming/a.mp4", "Size": 123, "ETag": "etag-a"},
            {"Key": "incoming/readme.txt", "Size": 3, "ETag": "etag-text"},
        ]}]


class FakeS3:
    def __init__(self, region="us-east-2"):
        self.region = region

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **_):
        raise NotFound()

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://signed.{self.region}.example/{operation}/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def get_bucket_location(self, Bucket):
        return {"LocationConstraint": self.region}


def test_s3_scan_is_idempotent_and_claim_urls_are_fresh(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", presign_seconds=3600, client=FakeS3())

    assert ingestor.scan() == 1
    assert ingestor.scan() == 0
    task = store.list_tasks()[0]
    assert task["source_url"] == "s3://source-bucket/incoming/a.mp4"
    assert task["output_object_key"] == "masked/masked_a.mp4"

    claimed = ingestor.materialize(task)
    assert claimed["source_url"].startswith("https://signed.us-east-2.example/get_object/source-bucket/incoming/a.mp4")
    assert claimed["output_upload_url"].startswith("https://signed.us-east-2.example/put_object/output-bucket/masked/masked_a.mp4")
    assert task["source_url"].startswith("s3://")
    assert task["output_upload_url"] == ""
    store.close()


def test_s3_separate_output_region_uses_its_own_client(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", output_region="ap-southeast-1",
                          client=FakeS3("us-east-2"), output_client=FakeS3("ap-southeast-1"))
    ingestor.validate_bucket_regions()
    assert ingestor.scan() == 1
    claimed = ingestor.materialize(store.list_tasks()[0])
    assert claimed["source_url"].startswith("https://signed.us-east-2.example/")
    assert claimed["output_upload_url"].startswith("https://signed.ap-southeast-1.example/")
    store.close()
