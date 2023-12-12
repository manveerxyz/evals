import random
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("dummy")
class DummyLM(LM):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests):
        res = []

        for _ in requests:
            res.append((-random.random(), False))

        return res

    def generate_until(self, requests):
        res = []

        for request in requests:
            # Assuming `request` is an object with an attribute that holds the context
            # Adjust the attribute names according to your actual `Instance` object structure
            ctx = request.args[0]  # or whatever the correct way to access context is
            res.append("this is a dummy response")  # Your dummy response
            assert ctx.strip() != ""

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for _ in requests:
            res.append(-random.random())

        return res
