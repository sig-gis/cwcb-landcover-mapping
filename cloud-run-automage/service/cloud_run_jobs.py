"""Cloud Run Jobs execution helper."""
from __future__ import annotations


class JobRunner:
    def __init__(self, client=None):
        if client is None:
            from google.cloud import run_v2
            client = run_v2.JobsClient()
        self.client = client

    def run(self, job_name: str, manifest_uri: str):
        from google.cloud import run_v2

        request = run_v2.RunJobRequest(
            name=job_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        args=["--manifest", manifest_uri]
                    )
                ]
            ),
        )
        return self.client.run_job(request=request)
