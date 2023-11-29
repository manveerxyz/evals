import logging
import dataclasses
import requests
from typing import Any, Dict, List
from lm_eval.api.task import TaskConfig
from lm_eval.config import Config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Event:
    run_id: str
    event_id: int
    sample_id: str
    type: str
    data: Any
    created_by: str
    created_at: str


@dataclasses.dataclass
class RunConfig:
    model: str
    model_args: str
    batch_size: int
    batch_sizes: List[int]
    device: str
    use_cache: bool
    limit: int
    bootstrap_iters: int


@dataclasses.dataclass
class FinalReport:
    results: Dict[
        str, Dict[str, float]
    ]  # dict of task name to dict of metric name to value
    versions: Dict[str, int]  # dict of task name to version
    configs: Dict[str, TaskConfig]
    config: RunConfig
    git_hash: str


class HttpRecorder:
    def __init__(
        self,
        run_id: str,
        config: Config = None,
    ):
        self._events = []
        self.failed_requests = 0
        self.run_id = run_id
        self.base_url = "http://localhost:5001"
        if config is not None:
            self.bearer_auth_token = config.api_key
        logger.info("HttpRecorder initialized")

    def _send_event(self, events: List[Event]):
        # Convert the events to dictionaries
        events_dict = [dataclasses.asdict(event) for event in events]

        logger.debug(f"Sending events: {events_dict}")

        try:
            # Send the events to the specified URL
            headers = {}
            if self.bearer_auth_token is not None:
                headers = {"Authorization": f"Bearer {self.bearer_auth_token}"}
            response = requests.post(self.base_url, json=events_dict, headers=headers)

            # If the request succeeded, log a success message
            if response.ok:
                logger.debug("Events sent successfully")

            # If the request failed, log a warning and increment failed_requests
            else:
                logger.warning(f"Failed to send events: {response.text}")
                self.failed_requests += len(
                    events
                )  # Increase the count by the number of events in the failed request

        except Exception as e:
            logger.warning(f"Failed to send events: {str(e)}")
            self.failed_requests += len(
                events
            )  # Increase the count by the number of events in the failed request

    def record_final_report(self, final_report: FinalReport) -> str:
        """
        Send a final report to the server.

        Args:
            final_report: A FinalReport instance or a dictionary with the same keys as a FinalReport instance.

        Returns:
            The results_url
        """
        if isinstance(final_report, FinalReport):
            config_data = dataclasses.asdict(final_report.config)
            results_data = final_report.results
            evals_data = final_report.versions
        elif isinstance(final_report, dict):
            # Assuming the dictionary has keys 'config', 'results', and 'versions'
            config_data = final_report.get("config", {})
            results_data = final_report.get("results", {})
            evals_data = final_report.get("versions", {})
        else:
            raise TypeError(
                "final_report must be either a FinalReport instance or a dictionary"
            )

        data = {
            "run_id": self.run_id,
            "config": config_data,
            "results": results_data,
            "evals": evals_data,
        }
        url = f"{self.base_url}/v1/evals/runs/report"
        headers = {}
        if self.bearer_auth_token is not None:
            headers = {"Authorization": f"Bearer {self.bearer_auth_token}"}

        logger.debug(f"Sending final report: {data}")

        try:
            # Send the events to the specified URL
            response = requests.post(url, json=data, headers=headers)

            # If the request succeeded, log a success message
            if response.ok:
                logger.debug("Final report sent successfully")

                # Return the results_url
                d = response.json()
                return d.get("results_url", None)

            # If the request failed, log a warning and increment failed_requests
            else:
                logger.warning(f"Failed to send final report: {response.text}")
                self.failed_requests += 1
        except Exception as e:
            logger.warning(f"Failed to send final report: {str(e)}")
            self.failed_requests += 1
