import time
import json
import atexit
import threading
import dataclasses
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
from lm_eval.api.task import TaskConfig
from lm_eval.config import Config
from print_color import print
from lm_eval.logger import eval_logger


MIN_FLUSH_EVENTS = 100
MIN_FLUSH_SECONDS = 10
MEGABYTE = 1024 * 1024
MAX_REQUEST_BODY_SIZE = 5 * MEGABYTE


@dataclasses.dataclass
class Output:
    doc_id: Union[str, int]
    doc: Any
    target: Any
    arguments: List[Any]
    resps: List[Any]
    filtered_resps: List[Any]

    # additional fields are metrics/scores. e.g.
    # bleu: 0.5
    # f1: 0.6
    # acc: 0.7
    # etc.


@dataclasses.dataclass
class Event:
    run_id: str
    eval_name: str
    event_id: int
    type: str
    data: dict
    created_at: str


@dataclasses.dataclass
class CommandLineConfig:
    schema_version: str = "0.0.2"
    cli_version: str = None

    model: str = None
    model_sha: Optional[str] = None  # if it's a local model
    model_args: Dict[str, Any] = None
    evals: List[str] = None
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None

    accelerate: Optional[bool] = None
    parallelize: Optional[bool] = None


@dataclasses.dataclass
class FinalReportBody:
    run_id: str
    project_slug: str
    results: Dict[str, Dict[str, float]]

    # evals: eval name ->
    #   TaskConfig
    #   & {
    #       version: string | int,
    #       schema_version: string // semver, e.g. "1.0.0"
    #     }
    evals: Dict[str, Dict[str, Any]]
    config: CommandLineConfig


class ScholarAPI:
    def __init__(
        self,
        config: Config = None,
        batch_size: int = 100,
    ):
        self._events: List[Event] = []
        self._last_flush_time = time.time()
        self._flushes_done = 0
        self._written_events = 0
        self._flushes_started = 0
        self._event_lock = threading.Lock()
        self._paused_ids: List[str] = []
        # atexit.register(self.flush_events)

        self.failed_requests = 0
        self.batch_size = batch_size
        self.frontend_url = "https://usescholar.org"
        self.base_url = "https://research-replicator.usescholar.org"
        # self.base_url = "http://localhost:5001"
        if config is not None:
            self.bearer_auth_token = config.api_key
            self.check_token_expiry()
        else:
            self.bearer_auth_token = None

        self.run_id = None
        self.project_id = None

        eval_logger.info("HttpRecorder initialized")

    def has_user_auth_token(self) -> bool:
        """
        Returns:
            True if the user has an auth token, False otherwise.
        """
        return self.bearer_auth_token is not None

    def get_user_projects(self) -> Tuple[List[Dict[str, str]], bool]:
        """
        Returns:
            A list of dicts, each containing a slug and display_name for a project.
            and a boolean indicating whether the request succeeded.
        """
        if not self.has_user_auth_token():
            return [], False

        url = f"{self.base_url}/v1/projects/list"
        headers = {}
        if self.bearer_auth_token is not None:
            headers = {"Authorization": f"Bearer {self.bearer_auth_token}"}

        try:
            # Send the events to the specified URL
            response = requests.get(url, headers=headers)

            # If the request succeeded, log a success message
            if response.ok:
                d = response.json()
                return d.get("projects", []), True
            else:
                eval_logger.warning(f"Failed to get projects: {response.text}")
                self.failed_requests += 1
                return [], False
        except Exception as e:
            eval_logger.warning(f"Failed to get projects: {str(e)}")
            self.failed_requests += 1

        return [], False

    def check_token_expiry(self) -> Tuple[bool, str]:
        """
        Check if the token has expired, and if so, prompt the user for a new one.

        Returns:
            A tuple containing a boolean indicating whether the token is valid, and a string
            containing an error message if the token is invalid.
        """
        url = f"{self.base_url}/v1/auth/api-credentials/verify"
        body = {"token": self.bearer_auth_token}
        headers = {}

        try:
            # Send the events to the specified URL
            response = requests.post(url, json=body, headers=headers)

            if not response.ok:
                return False, f"Invalid Response: {response.text}"

            # Get body
            d = response.json()
            if d.get("is_expired", False):
                print(
                    f"\nAPI Key expired. Get a new one from {self.frontend_url}/api-keys",
                    color="red",
                )
                exit(1)

            return True, None
        except Exception as e:
            eval_logger.warning(f"Failed to verify token: {str(e)}")
            self.failed_requests += 1
            return False, str(e)

    def init_run(self, project_id) -> Tuple[str, Union[str, None]]:
        """
        Send an init run event to the server.

        Args:
            project_id: The project_id to use for this run

        Returns:
            Run ID and Error Message (if any)
        """
        self.project_id = project_id

        data = {
            "project_slug": project_id,
        }
        url = f"{self.base_url}/v1/evals/runs"
        headers = {}
        if self.bearer_auth_token is not None:
            headers = {"Authorization": f"Bearer {self.bearer_auth_token}"}

        eval_logger.debug(f"Sending init run: {data}")

        try:
            # Send the events to the specified URL
            response = requests.post(url, json=data, headers=headers)

            # If the request succeeded, log a success message
            if response.ok:
                d = response.json()
                run_id = d.get("id", None)
                eval_logger.info(f"Init run sent successfully: {run_id}")
                self.run_id = run_id
                return run_id, None

            # If the request failed, log a warning and increment failed_requests
            else:
                eval_logger.warning(f"Failed to send init run: {response.text}")
                self.failed_requests += 1

            return None, response.text
        except Exception as e:
            eval_logger.warning(f"Failed to send init run: {str(e)}")
            self.failed_requests += 1
            return None, str(e)

    def record_final_report(
        self,
        versions: Dict[str, Union[int, str]],
        configs: Dict[str, TaskConfig],
        config: CommandLineConfig,
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Send a final report to the server.

        Args:
            versions: dict of task name to version
            configs: dict of task name to config
            config: CommandLineConfig
            results: dict of task name to dict of metric name to value

        Returns:
            The results_url
        """

        # if we haven't flushed events yet, do so now
        self.flush_events()

        evals = {}
        for task_name, task_config in configs.items():
            evals[task_name] = {
                "version": versions[task_name],
                "schema_version": task_config.get("schema_version", "0.0.2"),
                **task_config,
            }

        b = FinalReportBody(
            run_id=self.run_id,
            project_slug=self.project_id,
            results=results,
            evals=evals,
            config=config,
        )

        data = dataclasses.asdict(b)
        url = f"{self.base_url}/v1/evals/runs/report"
        headers = {}
        if self.bearer_auth_token is not None:
            headers = {"Authorization": f"Bearer {self.bearer_auth_token}"}

        eval_logger.debug(f"Sending final report: {data}")

        try:
            # Send the events to the specified URL
            response = requests.post(url, json=data, headers=headers)

            # If the request succeeded, log a success message
            if response.ok:
                eval_logger.debug("Final report sent successfully")

                # Return the results_url
                d = response.json()
                return d.get("results_url", None)

            # If the request failed, log a warning and increment failed_requests
            else:
                eval_logger.warning(f"Failed to send final report: {response.text}")
                self.failed_requests += 1
        except Exception as e:
            eval_logger.warning(f"Failed to send final report: {str(e)}")
            self.failed_requests += 1

    def _send_event(self, events: List[Event]):
        # Convert the events to dictionaries
        events_dict = []
        for event in events:
            if isinstance(event, Event):
                events_dict.append(dataclasses.asdict(event))
            else:
                events_dict.append(event)

        eval_logger.debug(f"Sending events, total: {len(events_dict)}")

        url = f"{self.base_url}/v1/evals/events"
        data = {
            "run_id": self.run_id,
            "events": events_dict,
        }

        try:
            # Send the events to the specified URL
            response = requests.post(url, json=data)

            # If the request succeeded, log a success message
            if response.ok:
                eval_logger.debug(f"Events sent successfully")

            # If the request failed, log a warning and increment failed_requests
            else:
                eval_logger.warning(f"Failed to send events: {response.text}")
                self.failed_requests += len(
                    events
                )  # Increase the count by the number of events in the failed request

        except Exception as e:
            eval_logger.warning(f"Failed to send events: {str(e)}")
            self.failed_requests += len(
                events
            )  # Increase the count by the number of events in the failed request

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        # batch_size = self.batch_size
        # for i in range(0, len(events_to_write), batch_size):
        #     batch = list(events_to_write[i : i + batch_size])
        #     try:
        #         self._send_event(batch)
        #     except RuntimeError as e:
        #         eval_logger.warning(f"Failed to send events: {str(e)}")
        #         self.failed_requests += len(batch)
        #         break

        current_batch_size_bytes = 0
        batch = []

        for event in events_to_write:
            event_json = None
            if isinstance(event, Event):
                event_json = json.dumps(dataclasses.asdict(event))
            else:
                event_json = json.dumps(event)
            event_size_bytes = len(event_json.encode("utf-8"))

            if current_batch_size_bytes + event_size_bytes > MAX_REQUEST_BODY_SIZE:
                if batch:
                    # Send the current batch if it's not empty
                    try:
                        self._send_event(batch)
                    except RuntimeError as e:
                        eval_logger.warning(f"Failed to send events: {str(e)}")
                        self.failed_requests += len(batch)
                        # Break if a batch fails to send
                        break

                # Start a new batch
                batch = [event]
                current_batch_size_bytes = event_size_bytes
            else:
                # Add event to the current batch
                batch.append(event)
                current_batch_size_bytes += event_size_bytes

        # Send any remaining events in the final batch
        if batch:
            try:
                self._send_event(batch)
            except RuntimeError as e:
                eval_logger.warning(f"Failed to send events: {str(e)}")
                self.failed_requests += len(batch)

    def flush_events(self):
        with self._event_lock:
            if len(self._events) == self._written_events:
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
        self._flush_events_internal(events_to_write)

    def record_event(self, type, data=None, eval_name=None):
        with self._event_lock:
            event = Event(
                run_id=self.run_id,
                eval_name=eval_name,
                event_id=len(self._events),
                type=type,
                data=data,
                created_at=str(datetime.now(timezone.utc)),
            )
            self._events.append(event)
            if (
                self._flushes_done < self._flushes_started
                or len(self._events) < self._written_events + MIN_FLUSH_EVENTS
                or time.time() < self._last_flush_time + MIN_FLUSH_SECONDS
            ):
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
            self._flush_events_internal(events_to_write)


##################################################
### Helper methods which use the global record ###
##################################################


scholar_api = None


def init_scholar_api(config: Config = None):
    global scholar_api
    scholar_api = ScholarAPI(config=config)


def get_scholar_api():
    global scholar_api
    return scholar_api


def record_output(eval_name: str, output: Output):
    global scholar_api
    if scholar_api is None:
        return

    if isinstance(output, Output):
        data = dataclasses.asdict(output)
        scholar_api.record_event(
            type="output",
            data=data,
            eval_name=eval_name,
        )
    elif isinstance(output, dict):
        scholar_api.record_event(
            type="output",
            data=output,
            eval_name=eval_name,
        )
    else:
        raise ValueError("Invalid output type")
